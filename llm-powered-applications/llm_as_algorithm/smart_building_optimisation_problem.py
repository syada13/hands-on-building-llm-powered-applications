import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools import PythonREPLTool

#Problem : Optimize the Heating, Ventilation and Air Conditioning (HVAC) setpoints in the building to minimize energy costs while ensuring occupant comfort. 

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
model= ChatOpenAI(temperature=0,model='gpt-3.5-turbo-0613')
agent_executor= create_python_agent(
  llm=model,
  tool=PythonREPLTool(),
  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)

#Goal :Find the minimum total energy cost, given some constraints

query = """
**Problem**:
You are tasked with optimizing the HVAC setpoints in a smart building to minimize energy costs while ensuring occupant comfort. The building has three zones, and you can adjust the temperature setpoints for each zone. The cost function for energy consumption is defined as:
- Zone 1: Energy cost = $0.05 per degree per hour
- Zone 2: Energy cost = $0.07 per degree per hour
- Zone 3: Energy cost = $0.06 per degree per hour
You need to find the optimal set of temperature setpoints for the three zones to minimize the energy cost while maintaining a comfortable temperature. The initial temperatures in each zone are as follows:
- Zone 1: 72°F
- Zone 2: 75°F
- Zone 3: 70°F
The comfort range for each zone is as follows:
- Zone 1: 70°F to 74°F
- Zone 2: 73°F to 77°F
- Zone 3: 68°F to 72°F
**Question**:
What is the minimum total energy cost (in dollars per hour) you can achieve by adjusting the temperature setpoints for the three zones within their respective comfort ranges?
"""
agent_executor.run(query)