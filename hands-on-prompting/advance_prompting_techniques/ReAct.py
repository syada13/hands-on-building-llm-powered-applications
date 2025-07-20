import os
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.schema import HumanMessage

model = ChatOpenAI(
    model_name="gpt-35-turbo"
)

load_dotenv()
key = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper()
tools= [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when need to answer questions about current event"
    )
]

agent_executor=initialize_agent(tools,model,
                                agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True)

print(agent_executor.agent.llm_chain.prompt.template)
agent_executor('who won Gold Medal for India at the Paris 2024 Olympics?')