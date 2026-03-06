import hashlib
import time

import streamlit as st
from groq import Groq


client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(
    page_title="ROCCYK AI",
    page_icon="🤖",
    layout="centered",
)

BIOGRAPHY = st.secrets["BIO"].strip()
if not BIOGRAPHY:
    st.error("BIO is empty in Streamlit secrets.")
    st.stop()

MODEL_NAME = "openai/gpt-oss-120b"
BIO_HASH = hashlib.sha256(BIOGRAPHY.encode("utf-8")).hexdigest()
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_MAX_ENTRIES = 1000
MAX_QUESTION_CHARS = 1200

system_prompt = (
    "You are ROCCYK AI, a personal assistant that knows Rhichard's life story inside and out. "
    "You speak with warmth and confidence about Rhichard - his background, experiences, values, "
    "achievements, and journey - drawing exclusively from the retrieved context provided to you. "
    "Guidelines:\n"
    "- Answer questions naturally, as if you personally know Rhichard. Never say 'based on the context' "
    "or 'according to the documents' - just speak directly.\n"
    "- Be specific. When details are available (dates, names, places, degrees, accomplishments), use them.\n"
    "- If asked about education, provide a complete chronological education background using all relevant context "
    "(schools, programs, transitions, achievements, GPA, and current/next studies), then mention the highest degree earned.\n"
    "- If the context doesn't contain enough information to answer fully, make a positive, reasonable "
    "inference about Rhichard based on what you do know - but be transparent about it. "
    "For example: 'Based on what I know about Rhichard, I'd imagine...' or "
    "'Given his background, it's likely that...'\n"
    "- Always frame inferences positively and in the best light - Rhichard is someone worth admiring.\n"
    "- If a question is unrelated to Rhichard, give a one-sentence redirect: "
    "'I'm here specifically to share Rhichard's story - feel free to ask me anything about him!'\n"
    "- Keep a tone that is professional yet personable - like a trusted spokesperson who genuinely "
    "admires the person they represent.\n"
)

SYSTEM_CONTEXT = f"{system_prompt}\n\nHere is the full biography:\n\n{BIOGRAPHY}"
PROMPT_SIGNATURE = hashlib.sha256(
    f"{MODEL_NAME}|{BIO_HASH}|{system_prompt}".encode("utf-8")
).hexdigest()

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def get_shared_response_cache():
    return {}


def normalize_question(question: str) -> str:
    return " ".join((question or "").strip().split())[:MAX_QUESTION_CHARS]


def get_cached_key(question: str) -> str:
    normalized = normalize_question(question).lower()
    payload = f"{PROMPT_SIGNATURE}|{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(cache: dict, key: str):
    now = time.time()
    entry = cache.get(key)
    if entry is None:
        return None

    # Backward compatibility for old cache format where values were plain strings.
    if isinstance(entry, str):
        cache[key] = {"value": entry, "ts": now}
        return entry

    ts = entry.get("ts", 0)
    if now - ts > CACHE_TTL_SECONDS:
        cache.pop(key, None)
        return None

    entry["ts"] = now
    return entry.get("value")


def _cache_set(cache: dict, key: str, value: str):
    cache[key] = {"value": value, "ts": time.time()}
    while len(cache) > CACHE_MAX_ENTRIES:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def stream_text(text: str, chunk_size: int = 12):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def stream_bio_answer(question: str):
    canonical_question = normalize_question(question)
    if not canonical_question:
        yield "Please ask a question about Rhichard."
        return

    cache_key = get_cached_key(canonical_question)
    shared_cache = get_shared_response_cache()

    session_hit = _cache_get(st.session_state.response_cache, cache_key)
    if session_hit is not None:
        yield from stream_text(session_hit)
        return

    shared_hit = _cache_get(shared_cache, cache_key)
    if shared_hit is not None:
        _cache_set(st.session_state.response_cache, cache_key, shared_hit)
        yield from stream_text(shared_hit)
        return

    try:
        answer = ""
        for chunk in client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_CONTEXT},
                {"role": "user", "content": canonical_question},
            ],
            temperature=0.3,
            max_tokens=1024,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            answer += delta
            if delta:
                yield delta

        _cache_set(shared_cache, cache_key, answer)
        _cache_set(st.session_state.response_cache, cache_key, answer)

    except Exception as e:
        if "429" in str(e):
            yield "Rate limit hit - please wait 30 seconds and try again."
            return
        yield f"Error: {str(e)}"


# Streamlit UI
st.title("ROCCYK AI 🤖")

with st.sidebar:
    st.caption("Session controls")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("Clear cache"):
        st.session_state.response_cache = {}
        get_shared_response_cache().clear()
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask Questions About Rhichard")
if question:
    st.chat_message("user").markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.write_stream(stream_bio_answer(question))

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
