import os
import hashlib
import re

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Tunables
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 14
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY_MESSAGES = 16
MAX_HISTORY_MESSAGE_CHARS = 1000
MAX_OUTPUT_TOKENS = 1400

EDUCATION_QUERY_HINTS = {
    "education",
    "educational",
    "academic",
    "academics",
    "school",
    "college",
    "university",
    "degree",
    "degrees",
    "studied",
    "study",
    "background",
    "gpa",
}

EDUCATION_CONTEXT_HINTS = {
    "school",
    "high school",
    "university",
    "college",
    "program",
    "degree",
    "graduated",
    "gpa",
    "dean",
    "honour",
    "honors",
    "roll",
    "mfda",
    "omscs",
    "accepted",
    "education",
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split long text into overlapping chunks for retrieval."""
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)

    return chunks


@st.cache_resource(show_spinner=False)
def build_rag_index(bio_text: str):
    """Build and cache embeddings + FAISS index for the BIO source text."""
    chunks = chunk_text(bio_text)
    if not chunks:
        return None, None, []

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return embedder, index, chunks


def retrieve_context(query: str, embedder, index, chunks, top_k: int = TOP_K):
    """Retrieve top-k semantically similar chunks for the current query."""
    if not query or embedder is None or index is None or not chunks:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)

    k = min(top_k, len(chunks))
    scores, indices = index.search(q_emb, k)

    ranked_indices = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks) and idx not in ranked_indices:
            ranked_indices.append(idx)

    selected = []
    fallback = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            fallback.append(chunks[idx])
        if score < SIMILARITY_THRESHOLD:
            continue
        if 0 <= idx < len(chunks):
            selected.append(chunks[idx])

    # Pull neighboring chunks so related facts split across boundaries stay together.
    expanded = []
    for idx in ranked_indices:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < len(chunks):
                expanded.append(chunks[neighbor])

    # Keep retrieval robust for larger chunks or stricter thresholds.
    # If no chunk clears the threshold, still pass the best semantic matches.
    return expanded if selected else fallback


def query_terms(text: str):
    return {term for term in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(term) >= 3}


def is_education_query(query: str):
    terms = query_terms(query)
    return bool(terms.intersection(EDUCATION_QUERY_HINTS))


def retrieve_keyword_chunks(chunks, keywords):
    lowered = {k.lower() for k in keywords}
    hits = []
    for chunk in chunks:
        haystack = chunk.lower()
        if any(keyword in haystack for keyword in lowered):
            hits.append(chunk)
    return hits


def merge_chunks(*chunk_lists):
    merged = []
    seen = set()
    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            key = " ".join(chunk.split())
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
    return merged


def build_context_window(chunks, max_chars: int = MAX_CONTEXT_CHARS):
    """Pack retrieved chunks into a strict character budget."""
    selected = []
    used = 0
    seen = set()

    for chunk in chunks:
        normalized = " ".join(chunk.split())
        if normalized in seen:
            continue
        if used + len(chunk) > max_chars:
            break
        selected.append(chunk)
        seen.add(normalized)
        used += len(chunk)

    return "\n\n".join(selected)


def compact_history(messages, max_messages: int = MAX_HISTORY_MESSAGES):
    """Keep only recent turns and cap individual message length."""
    compacted = []
    for msg in messages[-max_messages:]:
        content = (msg.get("content") or "").strip()
        if len(content) > MAX_HISTORY_MESSAGE_CHARS:
            content = content[:MAX_HISTORY_MESSAGE_CHARS].rstrip() + " ..."
        compacted.append({"role": msg.get("role", "user"), "content": content})
    return compacted


def normalize_query(text: str):
    return " ".join((text or "").strip().lower().split())


def build_config_signature(bio_text: str):
    payload = "|".join(
        [
            EMBED_MODEL,
            str(CHUNK_SIZE),
            str(CHUNK_OVERLAP),
            str(TOP_K),
            str(SIMILARITY_THRESHOLD),
            str(MAX_CONTEXT_CHARS),
            hashlib.sha256((bio_text or "").encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stream_llm_response(user_prompt: str):
    response_message = ""
    prompt_key = normalize_query(user_prompt)

    # Avoid another model call for repeated identical questions in-session.
    if prompt_key in st.session_state.answer_cache:
        cached = st.session_state.answer_cache[prompt_key]
        st.session_state.chat_history.append({"role": "assistant", "content": cached})
        yield cached
        return

    embedder, index, chunks = st.session_state.rag
    retrieved_chunks = retrieve_context(user_prompt, embedder, index, chunks)
    if is_education_query(user_prompt):
        education_chunks = retrieve_keyword_chunks(chunks, EDUCATION_CONTEXT_HINTS)
        retrieved_chunks = merge_chunks(retrieved_chunks, education_chunks)
    retrieved_context = build_context_window(retrieved_chunks)

    system_prompt = (
    "You are ROCCYK AI, a personal assistant that knows Rhichard's life story inside and out. "
    "You speak with warmth and confidence about Rhichard — his background, experiences, values, "
    "achievements, and journey — drawing exclusively from the retrieved context provided to you. "
    
    "Guidelines:\n"
    "- Answer questions naturally, as if you personally know Rhichard. Never say 'based on the context' "
    "or 'according to the documents' — just speak directly.\n"
    "- Be specific. When details are available (dates, names, places, degrees, accomplishments), use them.\n"
    "- If asked about education, provide a complete chronological education background using all relevant context "
    "(schools, programs, transitions, achievements, GPA, and current/next studies), then mention the highest degree earned.\n"
    "- If the context doesn't contain enough information to answer fully, make a positive, reasonable "
    "inference about Rhichard based on what you do know — but be transparent about it. "
    "For example: 'Based on what I know about Rhichard, I'd imagine...' or "
    "'Given his background, it's likely that...'\n"
    "- Always frame inferences positively and in the best light — Rhichard is someone worth admiring.\n"
    "- If a question is unrelated to Rhichard, give a one-sentence redirect: "
    "'I'm here specifically to share Rhichard's story — feel free to ask me anything about him!'\n"
    "- Keep a tone that is professional yet personable — like a trusted spokesperson who genuinely "
    "admires the person they represent.\n"
)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"Retrieved context:\n{retrieved_context}" if retrieved_context else "Retrieved context: (none)",
        },
        *compact_history(st.session_state.chat_history),
    ]

    for chunk in client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        response_message += delta
        yield delta

    st.session_state.answer_cache[prompt_key] = response_message
    st.session_state.chat_history.append({"role": "assistant", "content": response_message})


# configuring streamlit page settings
st.set_page_config(
    page_title="ROCCYK AI",
    page_icon=":robot_face:",
    layout="centered"
)

bio_text = os.environ.get("BIO", "")
if not bio_text.strip():
    st.error("BIO is empty. Add your biography to the BIO environment variable.")
    st.stop()

# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = build_rag_index(bio_text)

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

current_signature = build_config_signature(bio_text)
if st.session_state.get("config_signature") != current_signature:
    st.session_state.rag = build_rag_index(bio_text)
    st.session_state.answer_cache = {}
    st.session_state.config_signature = current_signature

# streamlit page title
st.title("ROCCYK AI")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# input field for user's message
user_prompt = st.chat_input("Ask Questions About Rhichard")

if user_prompt:
    # add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # display Groq's response
    with st.chat_message("assistant"):
        st.write_stream(stream_llm_response(user_prompt))
