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


@st.cache_resource
def get_shared_response_cache():
    return {}


def get_cached_key(question: str) -> str:
    normalized = " ".join((question or "").strip().lower().split())
    payload = f"{MODEL_NAME}|{BIO_HASH}|{system_prompt}|{normalized}"
    return hashlib.md5(payload.encode()).hexdigest()


def stream_text(text: str, chunk_size: int = 12):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def stream_bio_answer(question: str):
    canonical_question = " ".join((question or "").strip().split())
    cache_key = get_cached_key(canonical_question)
    shared_cache = get_shared_response_cache()

    if cache_key in st.session_state.response_cache:
        yield from stream_text(st.session_state.response_cache[cache_key])
        return
    if cache_key in shared_cache:
        answer = shared_cache[cache_key]
        st.session_state.response_cache[cache_key] = answer
        yield from stream_text(answer)
        return

    try:
        answer = ""
        for chunk in client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        f"Here is the full biography:\n\n{BIOGRAPHY}"
                    ),
                },
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

        shared_cache[cache_key] = answer
        st.session_state.response_cache[cache_key] = answer

    except Exception as e:
        if "429" in str(e):
            yield "Rate limit hit — please wait 30 seconds and try again."
            return
        yield f"Error: {str(e)}"
        return


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
        answer = st.write_stream(stream_bio_answer(question))

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
