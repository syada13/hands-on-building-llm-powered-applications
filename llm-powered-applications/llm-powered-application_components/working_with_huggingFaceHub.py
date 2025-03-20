import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub

load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
question = "What was the first Disney movie?"
template = """Question: {question}
Answer: give a direct answer"""

prompt = PromptTemplate(template=template, input_variables=["question"])
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    repo_id = repo_id,
    model_kwargs={"temperature": 0.5, "max_length": 1000},
    huggingfacehub_api_token=huggingfacehub_api_token
)

print(llm("what was the first disney movie?"))
