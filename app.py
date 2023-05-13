from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

llm = OpenAI(openai_api_key=API_KEY)
df = pd.read_csv("book_sample.csv")

agent = create_pandas_dataframe_agent(llm, df, verbose=False)


def query_agent(query):
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            "table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}
            
            If it requires drawing a graph, reply as follows:
            "chart_type": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}

            Example:
            "bar": {"columns": ["Name", "Age", "City"], "data": [["John", 25, "New York"], ["Lisa", 32, "London"], ...]}
            
            There can only be two types of chart, "bar" and "line".
            
            If it is just asking a question that does that requires neither, reply as follows:
            "answer": "answer"
            Example:
            "answer": "There a 5 book in the thriller genre"
            
            If you do not know the answer, reply as follows:
            "answer": "I do not know."
            
            Return all output as a string.
            
            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)

    return response.__str__()
