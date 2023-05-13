from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=True,
    device_map="auto",
)


pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2,
)

local_llm = HuggingFacePipeline(pipeline=pipe)


df = pd.read_csv("book_sample.csv")

agent = create_pandas_dataframe_agent(local_llm, df, verbose=True)


def query_agent(query):
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
            
            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)

    return response


print(query_agent("How many books are in the document?"))
