from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-2.0-flash-001")

from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun(api_wrapper_kwargs={"backend": "api"})
print(f"Tool's name = {search.name}")
print(f"Tool's description = {search.description}")
print(f"Tool's arg schema = {search.args_schema}")

#Input for the DuckDuckGo search tool
from langchain_community.tools.ddg_search.tool import DDGInput
query = "What is the weather in Noida like tomorrow?"
search_input = DDGInput(query=query)

#Run the tool:
result = search.invoke(search.invoke(search_input.model_dump()))
print(result)

#Invoke the LLM with tools
result = llm.invoke(query, tools=[search])
print(result)

#Use LangGraph in-built ReACT architecture based agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
  model=llm,
  tools=[search],
  state_modifier="Always use a duckduckgo_search tool for queries that require a fresh information"
)

"""
 The agent.stream method is used to stream events from the agent, 
 and the event.get method is used to retrieve the agent or tools from the event. 
"""
for event in agent.stream({"messages":[("user",query)]}):
  messages = event.get("agent", event.get("tools", {})).get("messages", [])
  for m in messages:
     m.pretty_print()



"""External system with an API can be wrapped as a tool.
 The RequestsToolkit allows to easily wrap any HTTP API:
 API: Use a free open-source currency API (https://frankfurter.dev/)
"""
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper

toolkit=RequestsToolkit(
  requests_wrapper = TextRequestsWrapper(headers={}),
  allow_dangerous_requets=False,
)

#Retrieve tools name
for tool in toolkit.get_tools():
  print(tool.name)


#Define payload schema
import api_spec

system_message = (
 "You're given the API spec:\n{api_spec}\n"
 "Use the API to answer users' queries if possible. "
)

agent = create_react_agent(
  model=llm,
  tools=toolkit.get_tools(),
  state_modifier=system_message.format(api_spec=api_spec)
)

query = "What is the swiss franc to US dollar exchange rate?"

events = agent.stream(
  {"messages":[("user", query)]},
  stream_mode="values"
)

for event in events:
  event["messages"][-1].pretty_print

"""OUTPUT
>> ============================== Human Message =================================
What is the swiss franc to US dollar exchange rate?
================================== Ai Message ==================================
Tool Calls:
  requests_get (541a9197-888d-4ffe-a354-c726804ad7ff)
 Call ID: 541a9197-888d-4ffe-a354-c726804ad7ff
  Args:
    url: https://api.frankfurter.dev/v1/latest?symbols=CHF&base=USD
================================= Tool Message =================================
Name: requests_get
{"amount":1.0,"base":"USD","date":"2025-01-31","rates":{"CHF":0.80706}}
================================== Ai Message ==================================
The Swiss franc to US dollar exchange rate is 0.0.80706.
"""