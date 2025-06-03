from langchain.chains import SequentialChain, llm_chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.environ['OPENAI_API_KEY']

llm=OpenAI()

'''
1. Generates a story based on a topic given by the user.
2. Generates a social media post to promote the story.
3. Generates an image to go along with the social media post.
'''

#Story generator
story_template = """
You are a storyteller. Given a topic, genre a target audience, you generate a story.
Topic: {topic}
Genre: {genre}
Audience: {audience}
Story: This is a story about the above topic, with the above genre and for the above audience:
"""
story_prompt_template = PromptTemplate(template=story_template, input_variables=["topic","genre","audience"])
story_chain = llm_chain(llm=llm, prompt=story_prompt_template,output_key="story")
result = story_chain({'topic': 'friendship story','genre': 'adventure','audience':'young adults'})
print(result['story'])


