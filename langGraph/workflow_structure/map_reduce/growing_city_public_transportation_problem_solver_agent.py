
"""
Use case : Automate the process of generating, evaluating, and analyzing solutions for improving public transportation in a growing city.

Overall flow:
— Solution generation
— Evaluation of each solution
— In-depth analysis of each evaluation
— Final ranking
Use of StateGraph: The code uses StateGraph to define the workflow. This manages the flow of data between different steps.
Use of Send: Send plays a crucial role in implementing the Map-Reduce pattern. a. continue_to_evaluation function:
"""
from typing import Annotated,TypedDict
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import START,END,StateGraph
from langgraph.types import Send
from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import operator
import os

#Load environment variables
load_dotenv()

# initalize AI model
llm = AzureChatOpenAI(
  azure_deployment="gpt-4o-mini",
  api_version="2024-08-01-preview",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2
)

# Define prompts for each step of the process
step1_prompt = """Step 1: I have a problem related to {input}. Could you brainstorm three distinct solutions? Please consider a variety of factors such as {perfect_factors}"""

step2_prompt = """Step 2: For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors.

Solutions:
{solutions}"""

step3_prompt = """Step 3: For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled.

Evaluation:
{review}"""

step4_prompt = """Step 4: Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution.

Detailed analysis:
{deepen_thought_process}"""


# Define data structures for AI outputs
class Solutions(BaseModel):
    solutions: list[str]

class Review(BaseModel):
    review: str

class DeepThought(BaseModel):
    deep_thought: str

class RankedSolutions(BaseModel):
    ranked_solutions: str

# Define the overall state of the process
class OverallState(TypedDict):
    input: str
    perfect_factors: str
    solutions: Annotated[list[str], operator.add]
    reviews: Annotated[list[str], operator.add]
    deep_thoughts: Annotated[list[str], operator.add]
    ranked_solutions: str

# Define the state for individual solution processing
class SolutionState(TypedDict):
    solution: str


# Graph component functions
def generate_solutions(state: OverallState):
    # Generate initial solutions based on the input problem and factors
    prompt = step1_prompt.format(input=state["input"], perfect_factors=state["perfect_factors"])
    response = model.with_structured_output(Solutions).invoke(prompt)
    return {"solutions": response.solutions}

def evaluate_solution(state: SolutionState):
    # Evaluate each solution individually
    prompt = step2_prompt.format(solutions=state["solution"])
    response = model.with_structured_output(Review).invoke(prompt)
    return {"reviews": [response.review]}

def deepen_thought(state: SolutionState):
    # Perform deeper analysis on each solution
    prompt = step3_prompt.format(review=state["solution"])
    response = model.with_structured_output(DeepThought).invoke(prompt)
    return {"deep_thoughts": [response.deep_thought]}

def rank_solutions(state: OverallState):
    # Rank all solutions based on the deep analysis
    deep_thoughts = "\n\n".join(state["deep_thoughts"])
    prompt = step4_prompt.format(deepen_thought_process=deep_thoughts)
    response = model.with_structured_output(RankedSolutions).invoke(prompt)
    return {"ranked_solutions": response.ranked_solutions}

# Define the mapping logic for parallel processing

def continue_to_evaluation(state: OverallState):
    # Create parallel branches for evaluating each solution
    return [Send("evaluate_solution", {"solution": s}) for s in state["solutions"]]

def continue_to_deep_thought(state: OverallState):
    # Create parallel branches for deep thinking on each evaluation
    return [Send("deepen_thought", {"solution": r}) for r in state["reviews"]]

# Construct the graph
graph = StateGraph(OverallState)

# Add nodes to the graph
graph.add_node("generate_solutions", generate_solutions)
graph.add_node("evaluate_solution", evaluate_solution)
graph.add_node("deepen_thought", deepen_thought)
graph.add_node("rank_solutions", rank_solutions)

# Add edges to connect the nodes
graph.add_edge(START, "generate_solutions")
graph.add_conditional_edges("generate_solutions", continue_to_evaluation, ["evaluate_solution"])
graph.add_conditional_edges("evaluate_solution", continue_to_deep_thought, ["deepen_thought"])
graph.add_edge("deepen_thought", "rank_solutions")
graph.add_edge("rank_solutions", END)

# Compile the graph
app = graph.compile()

# Execute the graph
for s in app.stream({
    "input": "improving public transportation in a growing city",
    "perfect_factors": "cost, efficiency, environmental impact, and user experience"
}):
    print(s)


