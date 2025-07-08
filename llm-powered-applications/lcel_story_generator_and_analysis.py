"""Build a multi-stage workflow that demonstrates how to:
  Generate content with one LLM call
  Feed that content into a second LLM call
  Preserve and transform data throughout the chain"""

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_prasers import StrOutputParser
from config import setEnvironments


setEnvironments()
llm = GoogleGenerativeAI(model="gemini-1.5-pro")
story_prompt = PromptTemplate.from_template("Write a short story about topic:{topic}")
analysis_prompt = PromptTemplate.from_template( "Analyze the following story's mood:\{story}")

story_chain =story_prompt |llm |StrOutputParser()
analysis_chain = analysis_prompt|llm|StrOutputParser()

story_with_analysis_chain = story_chain | analysis_chain
story_analysis = story_with_analysis_chain.invoke({"topic":"a rainy day"})
print("\nAnalysis:", story_analysis)



#Observation: We’ve lost the original story in our result – we only get the analysis! In production applications, we typically want to preserve context throughout the chain:

from langchain_core.runnables import RunnablePassthrough
# Using RunnablePassthrough.assign to preserve data
enhanced_chain = RunnablePassthrough.assign(story=story_chain)
.assign(analysis=analysis_chain)

chain_result = enhanced_chain.invoke({"topic": "a rainy day"})
print(chain_result.keys())

# Simplified dictionary construction
dict_chain = story_chain | {"analysis": analysis_chain}
result = dict_chain.invoke({{"topic": "a rainy day"}})
print(result.keys())