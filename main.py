import streamlit as st
from langchain_ollama.chat_models import ChatOllama

st.title("🦜🔗 InfoSec Training Generator")
st.write("This app helps you to organize advanced training methods for cybersecurity teams.")
st.write("It allows you to input a security incident and generates a training exercise based on Llama 3.2.")

llama_api_key = st.sidebar.text_input("Llama API Key", type="password")

def generate_exercise(text):
    model = ChatOllama(
        model="llama3.2",
        temperature=0,
        api_key=llama_api_key
    )
    st.info(model.invoke(text))

with st.form("main_form"):
    text = st.text_area("Security Incident :", height=200)
    submitted = st.form_submit_button("Generate Exercise")

    if not submitted and not llama_api_key:
        st.warning("Please enter your Llama API Key in the sidebar.")
    elif submitted and not text:
        st.warning("Please enter a security incident.")
    elif submitted and text:
        generate_exercise(text)
