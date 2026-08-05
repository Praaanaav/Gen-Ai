from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

out = StrOutputParser()

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

st.title("🐍 PyGuru: Python Coach AI")
st.markdown("💻 Ask Python.&nbsp;&nbsp;&nbsp;Learn Faster.&nbsp;&nbsp;&nbsp;Build Better.", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



query = st.chat_input("Ask me anything...")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            You are PyGuru, an expert Python coach.

            Rules:
            - Answer ONLY Python-related questions.
            - Help with Python syntax, OOP, libraries, debugging, DSA in Python, projects, and best practices.
            - If the user asks about another programming language, politely say:
            "I'm designed to help only with Python. Please ask a Python-related question."
            - Keep explanations beginner-friendly unless the user asks for advanced details.
        """
    ),
    ("human", "{query}")
])

chain = prompt | llm | out 

if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("User").markdown(query)
    res = chain.invoke(query)
    st.chat_message("assistant").markdown(res)
    st.session_state.messages.append({"role":"AI", "content":res})
