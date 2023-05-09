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

query = "How many rows of data are in the document?"

print(agent.run(query))
