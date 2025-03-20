from langchain.chains import LLMChain,SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Read .env file and load environment variables
load_dotenv()

# Read OPEN_API_KEY and configure to be sent via API call to model
openai_api_key = os.getenv('OPENAI_API_KEY')

#Initialise a LLM model
llm=OpenAI(temperature=0.7,openai_api_key=openai_api_key)

template =""" You are a comedian. Generate a joke on the following{topic}
Joke:"""
prompt_template_comedian = PromptTemplate(template=template, input_variables=["topic"])
joke_chain = LLMChain(llm=llm,prompt=prompt_template_comedian)

template="""You are a translator. Give a text input, translate it into {language}
Translation:
"""
prompt_template_translator= PromptTemplate(template=template,input_variables=["language"])
translator_chain = LLMChain(llm=llm,prompt=prompt_template_translator)

# This is the overall chain where we run these two chains in sequence.
chains_toBeExecuted = SimpleSequentialChain(chains=[joke_chain,translator_chain], verbose=True)
translated_joke = chains_toBeExecuted.run("Cats and Dogs")








