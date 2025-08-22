from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent

llm = ChatVertexAI(model="gemini-2.0-flash-001")

"""numexpr
 A fast numerical expression evaluator library based on NumPy to evaluate math expressions.
 https://github.com/pydata/numexpr
"""

import math
from langchain_core.tools import tool
import numexpr as numerical_expression

# Define calculator as a Python function, and wrap it with a built-in @tool decorator to create a tool from it

@tool
def calculator(expression:str) -> str:
  """Calculates a single mathematical expression, incl. complex numbers.

    Always add * to operations, examples:
      73i -> 73*i
      7pi**2 -> 7*pi**2
    """
  math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
  result = numerical_expression.evaluate(expression.strip ,local_dict=math_constants)
  return str(result)


#Test our calculator tool
  query = "How much is 2+3i squared?"
  agent= create_react_agent(llm,[calculator])
  for event in agent.stream({"messages":[("user", query)]},stream_mode="values"):
    event["messages"][-1].pretty_print()


""" Creating a tool from a Runnable.
 1. Define calculator function 
 2. Convert calculator function to Runnable using RunnableLambda
 3. Convert runnable to tool
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import convert_runnable_to_tool

def calculator(expression:str) -> str:
  math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
  result = numerical_expression.evaluate(expression.strip ,local_dict=math_constants)
  return str(result)

calculator_with_retry = RunnableLambda(calculator).with_retry(
  wait_exponential_jitter=True,
   stop_after_attempt=3,
)

calculator_tool = convert_runnable_to_tool(
  calculator_with_retry,
  name="calculator",
  description=(
     "Calculates a single mathematical expression, incl. complex numbers."
     "'\nAlways add * to operations, examples:\n73i -> 73*i\n"
     "7pi**2 -> 7*pi**2"),
     arg_types={"expression": "str"}
)

#Test
print(llm.invoke("How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0])