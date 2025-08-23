from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

llm = ChatVertexAI(model="gemini-2.0-flash-001")

#Define the data structure that describes a plan to solve a complex task.

class Step(BaseModel):
  """A step that is a part of the plan to solve the task."""
  step: str = Field(description="Decsription of the step")

class Plan(BaseModel):
  """A plan to solve the task."""
  steps: list[Step]

#Create a workflow
prompt = PromptTemplate.from_template(
  "Prepare a step-by-step plan to solve the given task.\n"
  "TASK:\n{task}\n"
)

"""Test workflow:
An LLM has the with_structured_output method that takes a schema as a Pydantic model, converts it to a tool, invokes the LLM with a given prompt by forcing it to call this tool, and parses the output by compiling to a corresponding Pydantic model instance.
"""
result = (prompt|llm.with_structured_output(Plan)).invoke(
  "How to write a bestseller on Amazon about generative AI?")

#Inspect the output
assert isinstance(result,Plan)
print(f"Amount of steps:{len(result.steps)}")
for step in result.steps:
  print(step.step)
  break


#Example 2: We can also ask Gemini to return an enum—in other words, only one value from a set of values:

from langchain_core.output_parsers import StrOutputParser

response_schema = {"type":"STRING","enum":["positive","negative","neutral"]}
prompt = PromptTemplate.from_template(
   "Classify the tone of the following customer's review:"
   "\n{review}\n")
llm_enum = ChatVertexAI(model_name="gemini-1.5-pro-002", response_mime_type="text/x.enum", response_schema=response_schema)
review = "I like this movie!"
result = (prompt | llm_enum | StrOutputParser()).invoke(review)
print(result)

  



