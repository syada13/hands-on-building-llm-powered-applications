
"""User case : Let's see how we can defined state fields that accumulate values. 
Option 1: Use a default reducer that replaces the values in the state
Option 2: use add method as a reducer:
"""


from typing import Annotated,Optional
from operator import add
class JobApplicationState:
  job_description: str,
  is_suitable: bool,
  application: str,
  actions:list[str]
  #actions:Annotated[list[str],add] #use add method as a reducer


def analyze_job_description(state):
  print("...Analyzing a provided job description ...")
  result = {
    "is_suitable":len(state["job_description"])< 100,
    "actions": ["action1"]}
  return result

def generate_application(state):
  print("...generating application...")
  return {"application": "some_fake_application", "actions": ["action2"]}


builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

#Display generated graph
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))


async for chunk in graph.astream(
  input={"job_description":"fake_id"},
  stream_mode="values"
):
print(chunk)
print("\n\n")







