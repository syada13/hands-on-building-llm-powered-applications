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
story_template = """You are a storyteller. Given a topic, genre a target audience, you generate a story.
Topic: {topic}
Genre: {genre}
Audience: {audience}
Story: This is a story about the above topic, with the above genre and for the above audience:
"""
story_prompt_template = PromptTemplate(template=story_template, input_variables=["topic","genre","audience"])
story_chain = llm_chain(llm=llm, prompt=story_prompt_template,output_key="story")
result = story_chain({'topic': 'friendship story','genre': 'adventure','audience':'young adults'})
print(result['story'])

#Social post generator
post_generator_template = """You are an influencer that, a given story generate a social media post to promote the story.
The style should reflect the type of social media used.
Story: {story}
Social media: {social}
Review from a New York Times play critic of the above play. 
"""
post_generator_prompt_template = PromptTemplate(input_variables=["story", "social"], template=post_generator_template)
social_chain = llm_chain(llm=llm, prompt=post_generator_prompt_template, output_key='post')
post = social_chain({'story': result['story'], 'social': 'Instagram'})
print(post['post'])

#Image generator chain
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
image_generator_template= """Generate a detailed prompt to generate an image based on the following social media post:
Social media post:{post}
"""
image_generator_prompt = PromptTemplate(input_variables=["post"],template=image_generator_template)
image_chain = llm_chain(llm=llm,prompt=image_generator_prompt,output_key='image')
image_url = DallEAPIWrapper().run(image_chain().run("a cartoon-style cat playing piano"))

import cv2
from skimage import io
image = io.imread(image_url)
cv2.imshow('image',image)
cv2.waitKey(0)   #wait for a keyboard input
cv2.destroyAllWindows()

#Putting all together using Sequential chain
overall_chain = SequentialChain(input_variables = ['topic', 'genre', 'audience', 'social'],
 chains=[story_chain,social_chain,image_chain],
 output_variables = ['post','image'],
 verbose=True
)

result = overall_chain({'topic': 'friendship story','genre':'adventure', 'audience': 'young adults', 'social': 'Instagram'}, return_only_outputs=True)
print(result)

