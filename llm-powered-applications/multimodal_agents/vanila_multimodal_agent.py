import os
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.agents import initialize_agent,AgentType
from langchain import ChatOpenAI,OpenAI

# Load Azure AI services environment variables
load_dotenv()
azure_cogs_key = os.environ["AZURE_COGS_KEY"]
azure_cogs_endpoint = os.environ["AZURE_COGS_ENDPOINT"]
azure_cogs_region = os.environ["AZURE_COGS_REGION"]
openai_api_key = os.environ["OPENAI_API_KEY"]

#Configure our toolkit and also see which tools we have
toolkit = AzureCognitiveServicesToolkit()
[ (tool.name,tool.decsription) for tool in toolkit.get_tools()]

# Initialized agent 'STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION ' as it allows for multi-tools input.

llm = OpenAI()
model = ChatOpenAI()
agent = initialize_agent(
  tools = toolkit.get_tools(),
  llm = llm,
  agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose = True
)

#Testing: Leveraging a single tool
#Scenario 1: let’s simply ask the agent to describe the slingshot picture
image_url = "https://www.stylo24.it/wp-content/uploads/2020/03/fionda.jpg"
description = agent.run(f"what shows the following image?: {image_url}" )
print(decsription)

#Scenario 2- Let’s ask our model something more challenging. Let’s ask it to reason about the consequences of letting the slingshot go:
agent.run(f"what happens if the person lets the slingshot go?: {image_url}")

#Leveraging multiple tools
# Scenario 3 - we want the model to read a story aloud to us based on a picture.
story_picture = "https://i.redd.it/diawvlriobq11.jpg"
agent.run(f"Tell me a story related to the following picture and read the story aloud to me: {story_picture}")

'''
Oservation: The agent was able to invoke two tools to accomplish the request:
1 : It first started with the 'image_analysis' tool to generate the image caption used to produce the story.
2: Then, it invoked the 'text2speech' tool to read it aloud to the user.

'''





