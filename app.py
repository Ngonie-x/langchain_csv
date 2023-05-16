from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")


def create_agent(filename: str):
    llm = OpenAI(openai_api_key=API_KEY)
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm, df, verbose=False)


def query_agent(agent, query):
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            "table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}

            If the query requires creating a bar chart, reply as follows:
            "bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}
            
            If the query requires creating a line chart, reply as follows:
            "line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}
            
            There can only be two types of chart, "bar" and "line".
            
            If it is just asking a question that does that requires neither, reply as follows:
            "answer": "answer"
            Example:
            "answer": "The title with the highest rating is 'Gilead'"
            
            If you do not know the answer, reply as follows:
            "answer": "I do not know."
            
            Return all output as a string.
            
            Lets think step by step.
            
            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)

    return response.__str__()
