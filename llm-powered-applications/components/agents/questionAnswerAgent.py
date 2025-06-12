import os
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import AgentType, initialize_agent


# Read .env file and load environment variables
load_dotenv()

#Read OPEN_API_KEY and configure to be sent via API call to model
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key= os.getenv('SERPAPI_API_KEY')

# Create an instance of SerpAPIWrapper and OpenAI model
search = SerpAPIWrapper(serpapi_api_key)

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-4o'
)

# Defining a single tool
tools =[
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )]

#Creating the Conversational Agent with Custom Tools
agent = initialize_agent(tools,llm=turbo_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, Verbose=True)
agent.run('When did India win first Cricket World Cup?')










