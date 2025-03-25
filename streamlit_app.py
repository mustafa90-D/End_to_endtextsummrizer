# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("📝 Text Summarization App")

input_text = st.text_area("Enter text to summarize", height=200)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            response = requests.post(
                "http://localhost:8080/predict",
                params={"text": input_text}
            )
            if response.status_code == 200:
                st.subheader("Summary")
                st.success(response.text)
            else:
                st.error("Failed to summarize. Try again.")
        except Exception as e:
            st.error(f"Error: {e}")
