from typing_extensions import TypedDict

# Define state schema
# State - a shared data structure that holds the current context and information as an AI system processes data within a graph
class JobApplicationState(TypedDict):
  job_description: str
  is_suitable: str
  application: str

#Define workflow
from langgraph.graph import StateGraph,START,END,Graph

def analyze_job_description(state):
  print("...Analyzing a provided jo description....")
  return {"is_suitable":len(state["job_description"]) > 100}

def generate_application(state):
  print("......generating application...")
  return {"application":"some_fake_application"}


#The StateGraph class is used to define the state graph of a workflow, which consists of nodes and edges between them. This class allows for the addition of nodes and edges, which can be used to represent different actions or decision points in the workflow
builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description",analyze_job_description)
builder.add_node("generate_application",generate_application)
builder.add_edge(START,"analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application",END)
graph = builder.compile()

#Look the created graph
from IPython.display import Image,display
display(Image(graph.get_graph().draw_mermaid_png()))

#Execute workflow
response = graph.invoke({"job_description":"fake_job_id"})
print(response)

#Example 2 - Graph with conditional logic
from typing import Literal
conditional_builder = StateGraph(JobApplicationState)
conditional_builder.add_node("analyze_job_description", analyze_job_description)
conditional_builder.add_node("generate_application", generate_application)

def is_suitable_condition(state:StateGraph)->Literal("generate_application",END):
  if state.is_suitable:
    return "generate_application"
  return END


conditional_builder.add_edge(START, "analyze_job_description")
conditional_builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
conditional_builder.add_edge("generate_application", END)
conditional_graph = conditional_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

#Execute conditional worflow
response = conditional_graph.invoke({"job_description":"fake_job_id"})
print(response)





