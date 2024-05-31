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
            {"role": "system", "content": """Rhichard Oliver Christian Yang Koh is a Chinese-Filipino-Canadian, born on May 24, 2000. Rhichard was born in Makati, Philippines and was nine years old when he moved to Toronto, Canada with his mother and sister. At a young age, Rhichard had an interest in technology through his family.
Rhichard attended Blessed Cardinal Newman Catholic High School, which is now known as St. John Henry Newman Catholic School. During high school, he tailored his subjects towards the medical field - taking courses such as Biology and Chemistry. By the time he graduated in 2018, he was accepted to York University for Health Sciences. After three days, Rhichard realized that this was not the correct career for him and dropped out of York University. Koh took a 3 year gap, where he spent his time working at Online Trading Academy as a Student support Specialist/ Center Administrator. His responsibilities were to teach and guide students on how to trade stocks, options, commodities, and currency. He learned the valuable skills of teamwork, time management, responsibility, and patience.
In the beginning of 2020, just after the pandemic, Rhichard was laid off, due to a provincial shut down. This time allowed him to supplement his income through trading options, specifically selling credit spreads/iron condors on the SPX.
In the beginning of 2021, Rhichard decided he wanted to apply to college again. He applied to *list all the placed he applied*. Ultimately, Rhichard decided to accept an offer from Durham College for Artificial Intelligence, a newer program at this college. His program consists of *the number of people in his class*, and he currently holds a *his GPA*. During his first year at college, he *insert the different projects he has done*. He took *insert his courses for first year*, where he exceeded all expectations. In the summer of 2023, Rhichard got a date scientist internship at Bell. During his time at Bell, he worked on *his projects* with his team.
Rhichard continues to find new ways to grow and flourish in his future career path into Artificial Intelligence.
"""},
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
