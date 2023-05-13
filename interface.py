import streamlit as st
import time


def get_query(query):
    print(query)


st.title("👨‍💻 Chat with your CSV")

st.write("Please upload your csv file below.")

data = st.file_uploader("Upload a CSV")

query = st.text_area("Insert your query")

sbt_btn = st.button("Submit Query", on_click=get_query(query), type="primary")
