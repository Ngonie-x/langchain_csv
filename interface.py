import streamlit as st
import pandas as pd

from app import query_agent


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """

    print(response)
    # Remove the surrounding single quotes from the string representation
    return eval("{" + response + "}")


def write_response(response_dict: dict):
    if (
        response_dict.get("bar")
        or response_dict.get("line")
        or response_dict.get("table")
    ):
        if response_dict.get("bar"):
            # Create the DataFrame from the evaluated dictionary
            data = response_dict.get("bar")
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)
        elif response_dict.get("line"):
            data = response_dict.get("line")
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.line_chart(df)
        else:
            data = response_dict.get("table")
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)

    else:
        st.write(response_dict.get("answer"))


def get_query():
    query_text = query
    response = query_agent(query_text)
    decoded_response = decode_response(response)
    write_response(decoded_response)


st.title("👨‍💻 Chat with your CSV")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV")

query = st.text_area("Insert your query")

sbt_btn = st.button("Submit Query", on_click=get_query, type="primary")
