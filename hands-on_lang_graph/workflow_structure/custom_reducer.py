from typing import Annotated, Optional, Union
from operator import add
from langgraph.graph import StateGraph,START,END,Graph

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

def generate_application(state):
    print("...generating application...")
    return {"application": "some_fake_application", "actions": ["action2", "action3"]}



builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

async for chunk in graph.astream(
    input={"job_description":"fake_jd"},
    stream_mode="values"
):
    print(chunk)
    print("\n\n")


