from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

import streamlit as st


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",streaming=True
)


# -----------------------------
# Google Search Tool
# -----------------------------

search = GoogleSerperAPIWrapper()

tools = [search.run]


# -----------------------------
# Session Memory
# -----------------------------

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


# -----------------------------
# Create Agent
# -----------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a cool AI that can search from Google",
    checkpointer=st.session_state.memory,
)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Chatbot : Answers at the speed of thought")


# -----------------------------
# Display Previous Chat History
# -----------------------------

for message in st.session_state.history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

prompt = st.chat_input("Ask Anything...")


if prompt:

    # Display user message
    st.chat_message("user").markdown(prompt)

    # Save user message
    st.session_state.history.append({
        "role": "user",
        "content": prompt
    })


    # -------------------------
    # Agent Response
    # -------------------------

    res = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        },
        stream_mode="messages"  
)


    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""
        for chunk in res:
            message+=chunk[0].content
            space.write(message)


    # Save AI response
    st.session_state.history.append({
        "role": "ai",
        "content": message
    })