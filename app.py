from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

llm = OpenAI(openai_api_key=API_KEY)
df = pd.read_csv("books.csv")

agent = create_pandas_dataframe_agent(llm, df, verbose=True)


def query_agent(query):
    prompt = f"""
    For the following query, if it requires drawing a table, reply as follows:
    'table': '[heading1, heading2, heading3], [row1data1, row1data2, row1data3], [row2data1, row2data2, row2data3]...'
    
    If it requires drawing a graph, reply as follows:
    'graph': '[(type, bar), (x-axis-label, y-axis-label), [(x-value1, y-value1), (x-value2, y-value2), ...]]'
    
    If it is just asking a question that does that requires neither, reply as follows:
    'Answer': 'Answer'
    
    Query: {query}
    """

    response = agent.run(prompt)

    return response.__str__()
