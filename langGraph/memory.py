"""
Store chat message history in-memory using InMemoryChatMessageHistory class.
This class extends BaseListChatMessageHistory class and provides methods to get, add and clear messages.

RunnableWithMessageHistory: It is used to incorporate previous messages into a runnable chain, enabling the development of applications that can quickly retrieve and utilize conversation history. This concept is particularly useful in chatbot applications where maintaining context is crucial. By using RunnableWithMessageHistory, developers can focus on building the core functionality of their applications without worrying about the complexities of managing chat memory.

 a. We create a fake chat model with a callback that prints out the amount of input messages each time it’s called. 

 b. Then we initialize the dictionary that keeps histories, and we create a separate function that returns a history given the session_id:

 FakeListChatModel is a part of LangChain's FakeChatModel classes, which are designed to help test error handling in chains by simulating LLM failures. This class specifically takes a list of messages and returns them one by one on each invocation, allowing for controlled testing of how a chain handles different responses.

"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import trim_messages,HumanMessage
from langchain.callbacks.base import BaseCallbackHandler

class PrintOutputCallback(BaseCallbackHandler):
  def on_chat_model_start(self,serialized, message, **kwargs):
    print(f"Amount of input messages: {message}")


sessions = {}
handler=PrintOutputCallback()
llm = FakeListChatModel(responses=["ai1","ai2","ai3"])


def get_session_history(session_id: str):
  if session_id not in sessions:
    sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id]


trimmer = trim_messages(
  max_tokens=1,
  strategy="last",
  token_counter=len,
  include_system=True,
  start_on="human"
)

raw_chain = trimmer | llm
chain = RunnableWithMessageHistory(raw_chain,get_session_history)

"""
The config dictionary is used to specify the configuration for the chain, including callbacks and configurable settings. In below example:
   The "callbacks" key can be used to specify a list of callback functions, such as PrintOutputCallback(). 
   The "configurable" key can be used to specify a session_id, which ensures that the agent can track the conversation within the same session
"""
config ={"callbacks": [PrintOutputCallback()],"configurable":{"session_id": "1"}}

#Testing : Make sure that our history keeps all the interactions with the user but a trimmed history is passed to the LLM:
_ = chain.invoke([HumanMessage("Hi Suresh!")],config = config)

print(f"History length: {len(sessions['1'].messages)}")

_ = chain.invoke(
    [HumanMessage("How are you?")],
    config=config,
)
print(f"History length: {len(sessions['1'].messages)}")

"""Production Usage Note :

When designing a real application, you should be cautious about managing access to somebodys sessions. For example, if you use a sequential session_id, users might easily access sessions that dont belong to them. Practically, it might be enough to use a uuid (a uniquely generated long identifier) instead of a sequential session_id, or, depending on your security requirements, add other permissions validations during runtime.
"""

"""Checkpoint

Checkpointing in LangGraph is used to record the state of an application at a particular point in time, allowing it to be restored later. This is achieved by retrieving the checkpoint_id from the last checkpoint in the list of checkpoints. The graph is then invoked with a HumanMessage and a config that includes the thread_id and checkpoint_id.
"""

from langgraph.graph import MessageGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from langgraph.graph import START, END


def test_node(state):
   # ignore the last message since it's an input one
   print(f"History length = {len(state[:-1])}")
   return [AIMessage(content="Hello!")]

builder = MessageGraph()
builder.add_node("test_node", test_node)
builder.add_edge(START, "test_node")
builder.add_edge("test_node", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

#Invoke graph
_ = graph.invoke([HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-a"}})
_ = graph.invoke([HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-b"}})
_ = graph.invoke([HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-a"}})

#Inspect checkpoints for a given thread:
checkpoints = list(memory.list(config={"configurable": {"thread_id": "thread-a"}}))
for check_point in checkpoints:
  print(check_point.config["configurable"]["checkpoint_id"])

#Restore from the initial checkpoint for thread-a. 
checkpoint_id = checkpoints[-1].config["configurable"]["checkpoint_id"]
_ = graph.invoke(
    [HumanMessage(content="test")],
    config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}})

#Start from an intermediate checkpoint
heckpoint_id = checkpoints[-3].config["configurable"]["checkpoint_id"]
_ = graph.invoke(
    [HumanMessage(content="test")],
    config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}})


