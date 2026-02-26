import os
import streamlit as st
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

def stream_llm_response():
    response_message = ""

    for chunk in openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": os.environ.get('BIO')
},
            *st.session_state.chat_history
        ],
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
    st.session_state.chat_history.append({"role": "assistant", "content": response_message})

# configuring streamlit page settings
st.set_page_config(
    page_title="ROCCYK AI",
    page_icon=":robot_face:",
    layout="centered"
)

# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# streamlit page title
st.title("ROCCYK AI :robot_face:")

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

    # display GPT-4o's response
    with st.chat_message("assistant"):
        st.write_stream(stream_llm_response())
