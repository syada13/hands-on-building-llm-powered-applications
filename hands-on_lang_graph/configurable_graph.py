from typing import Annotated, Optional, Union
from operator import add

def my_reducer(left: list[str], right: Optional[Union[str, list[str]]]) -> list[str]:
  if right:
    return left + [right] if isinstance(right, str) else left + right
  return left


class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: Annotated[list[str], my_reducer]

def analyze_job_description(state):
    print("...Analyzing a provided job description ...")
    result = {
        "is_suitable": len(state["job_description"]) < 100,
        "actions": "action1"}
    return result

#add configuration to our graph:
from langchain_core.runnables.config import RunnableConfig
def generate_configurable_application(state:JobApplicationState,config:RunnableConfig):
  model_provider = config["configurable"].get("model_provider","Google")
  model_name = config["configurable"].get("model_name", "gemini-1.5-flash-002")
  print(f"...generating application with {model_provider} and {model_name} ...")
  return {"application": "some_fake_application", "actions": ["action2", "action3"]}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_configurable_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

response = graph.invoke({"job_description":"fake_jd"})
print(response)

"""Output
...Analyzing a provided job description ...
...generating application with Google and gemini-2.0-flash ...
{'job_description': 'fake_jd', 'is_suitable': True, 'application': 'some_fake_application', 'actions': ['action1', 'action2', 'action3']}
"""
response = graph.invoke({"job_description":"fake_jd"}, config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}})
print(response)

"""Output
...Analyzing a provided job description ...
...generating application with OpenAI and gpt-4o ...
{'job_description': 'fake_jd', 'is_suitable': True, 'application': 'some_fake_application', 'actions': ['action1', 'action2', 'action3']}
"""




