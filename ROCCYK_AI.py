import hashlib

import streamlit as st
from groq import Groq


client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(
    page_title="ROCCYK AI",
    page_icon=":robot_face:",
    layout="centered",
)

BIOGRAPHY = st.secrets["BIO"].strip()
if not BIOGRAPHY:
    st.error("BIO is empty in Streamlit secrets.")
    st.stop()
MODEL_NAME = "openai/gpt-oss-120b"
BIO_HASH = hashlib.sha256(BIOGRAPHY.encode("utf-8")).hexdigest()

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


if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_cached_key(question: str) -> str:
    normalized = " ".join((question or "").strip().lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


@st.cache_data(ttl=86400, max_entries=1000, show_spinner=False)
def cached_completion(question: str, prompt_hash: str) -> str:
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"Here is the full biography:\n\n{BIOGRAPHY}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def query_bio(question: str):
    canonical_question = " ".join((question or "").strip().split())
    cache_key = get_cached_key(canonical_question)

    if cache_key in st.session_state.response_cache:
        return st.session_state.response_cache[cache_key]

    try:
        # Shared cache invalidates automatically when model/prompt/bio changes.
        prompt_hash = hashlib.sha256(
            f"{MODEL_NAME}|{system_prompt}|{BIO_HASH}".encode("utf-8")
        ).hexdigest()
        answer = cached_completion(canonical_question, prompt_hash)

        st.session_state.response_cache[cache_key] = answer
        return answer

    except Exception as e:
        if "429" in str(e):
            return "Rate limit hit — please wait 30 seconds and try again."
        return f"Error: {str(e)}"


# Streamlit UI
st.title("ROCCYK AI")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask Questions About Rhichard")
if question:
    st.chat_message("user").markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = query_bio(question)
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
