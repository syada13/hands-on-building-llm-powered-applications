import os
from codeinterpreterapi import CodeInterpreterSession
from dotenv import load_dotenv

#load environments and key
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']
os.environ['VERBOSE'] = "True"

# create a session
async with CodeInterpreterSession() as session:
  # generate a response based on user input
  response = await session.generate_response(
    "Generate a plot of the price of S&P500 index in the last 5 days."
  )

  # output the response
  print("AI: ",response.content)
  for file in response.files:
    file.show_image()
