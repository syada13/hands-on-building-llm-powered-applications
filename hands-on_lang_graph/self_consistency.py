"""
Self-consistency:
The idea behind self-consistency is simple: let’s increase an LLM’s temperature, sample the answer multiple times, and then take the most frequent answer from the distribution.
"""
import os
from dotenv import load_dotenv
from langchain_google_vertex import ChatVertexAI

#load environments and key
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"

#Initialize model
llm_small = ChatVertexAI(model_name="gemini-1.5-flash-001")

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

math_cot_prompt = hub.pull("arietem/math_cot")
cot_chain = math_cot_prompt |llm_small | StrOutputParser()
print(cot_chain.invoke("Solve equation 2*x+5=15"))

from operator import itemgetter
from langchain_core.prompts import PromptTemplate

parse_prompt_template =(
  "Given the initial question and a full answer, "
    "extract the concise answer. Do not assume anything and "
    "only use a provided full answer.\n\nQUESTION:\n{question}\n"
    "FULL ANSWER:\n{full_answer}\n\nCONCISE ANSWER:\n"
)

parse_prompt = PromptTemplate.from_template(parse_prompt_template)
final_chain = (
  {"full_answer": itemgetter("question") | cot_chain,
  "question": itemgetter("question")
  } 
  | parse_prompt 
  | llm_small
  | StrOutputParser
)
print(final_chain.invoke({"question": "Solve equation 2*x**2-96*x+1152"}}))

#Let's run generation multiple times and sample the most frequest one from the distribution:
generations = []
for _ in range(20):
  generations.append(final_chain.invoke({"question": "Solve equation 2*x**2-96*x+1152"}, temperature=2.0).strip())

from collections import Counter
print(Counter(generations).most_common(1)[0][0])
