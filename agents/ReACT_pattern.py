from langchain_google_vertex import ChatVertexAI
# I am skipping environment variables load on purpose
llm = ChatVertexAI(model="gemini-1.5")

search_tool = {
   "title": "google_search",
    "description": "Returns about fresh events and news from Google Search engine based on a query",
   "type": "object",
   "properties": {
       "query": {
           "description": "Search query to be sent to the search engine",
           "title": "search_query",
           "type": "string"},
   },
   "required": ["query"]
}


from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
calculator_tool = {
  "title": "calculator",
  "description": "Compute mathematical expressions",
  "type": "object",
  "properties": {
    "expression":{
      "description": "A mathematical expression to be evaluated by a calculator",
      "title": "expression",
      "type": "string"
      },
    },
  "required": ["expression"]
  }

  prompt = ChatPromptTemplate.from_messages([
    ("system", "Always use a calculator for mathematical computations, and use Google Search for information about fresh events and news."), 
    MessagesPlaceholder(variable_name="messages"),]
    )
  
  #Bind tools with llm for all future calls to avoid it to pass in each call
  llm_with_tools = llm.bind(tools=[search_tool, calculator_tool]).bind(prompt=prompt)


  # Mocked tools
def mocked_google_search(query: str) -> str:
  print(f"Called GOOGLE_SEARCH with query={query}")
  return "Draupadi is a president of INDIA and she's a 67 years old"

import math
def calculator(expression: str) -> float:
  print(f"Called CALCULATOR with expression={expression}")
  if "sqrt" in expression:
    return math.sqrt(4*4)
  return 4*4

  """
  Now that we have an LLM that can call tools, let’s create the nodes we need. 
   1. Function that calls an LLM,
   2. Function that invokes tools and returns tool-calling results (by appending ToolMessages to the list of messages in the state), 
   3. And a function that will determine whether the orchestrator should continue calling tools or whether it can return the result to the user:
  """

  from typing import TypedDict
  from langgraph.graph import MessageState, StateGraph,STRT,END

  def invoke_llm(state: MessageState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


  def call_tools(state: MessageState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    new_messages = []
    for tool_call in tool_calls:
      if tool_call["name"] == "google_search":
        tool_result = mocked_google_search(**tool_call["args"])
        new_messages.append(
          ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
          )
      elif tool_call["name"] == "calculator":
       tool_result = mocked_calculator(**tool_call["args"])
       new_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
      else:
       raise ValueError(f"Tool {tool_call['name']} is not defined!")
    return {"messages": new_messages} 


def should_run_tools(state: MessagesState):
   last_message = state["messages"][-1]
   if last_message.tool_calls:
     return "call_tools"
   return END

#Lets define LangGraph workflow
builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("call_tools", call_tools)
builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", should_run_tools)
builder.add_edge("call_tools", "invoke_llm")
graph = builder.compile()

#Test tools calling
question = "What is a square root of the current INDIA president's age multiplied by 132?"
result = graph.invoke({"messages":[HumanMessage(content=question)]})
print(result["messages"][-1].content)

""" DO NOT implement ReACT pattern

LangGraph offers a pre-built implementation of a ReACT pattern, so you don’t need to implement it yourself: Below is an example
"""

from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
  llm=llm,
  tools=[search_tool, calculator_tool],
  prompt=system_prompt)





