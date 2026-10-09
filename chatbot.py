
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Help me to run the query of the user. Answer the query of the user."),
    ("human", "Question: {query}")
])


def generate_response(query, llm, temperature, max_tokens, api_key):
    model = ChatGroq(model_name=llm, temperature=temperature, api_key=api_key, max_tokens=max_tokens)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"query": query})


st.title("Q and A Chatbot Using Groq with Any API KEY Supported ")
st.caption("Powered by Groq ")

query = st.text_input("What is your Query ?")

st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Provide the Api key ", type='password')
llm = st.sidebar.selectbox(
    'Select The LLm :',
    [
        # Production / Standard Text Models
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",

        # Preview & Specialized Models
        "qwen/qwen3.8-27b",
        "minimaxai/minimax-m2.7",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-safeguard-20b",

        # Speech Recognition (Audio STT)
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    ],
)
temperature = st.sidebar.slider("Select the value of the temperature :", min_value=0.0, max_value=2.0, value=0.1)
max_token = st.sidebar.slider("Select thge value of max number of token :", min_value=50, max_value=200, value=100)

if st.button("Answer"):
    if query and api_key:
        response = generate_response(query, llm, temperature, max_token, api_key)
        st.write(response)
    else:
        st.warning("Please enter a query and API key.")
