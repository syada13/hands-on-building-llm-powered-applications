#Import required library
import os
import openai
from dotenv import load_dotenv

# Read .env file and load environment variables
load_dotenv()

# Read OPEN_API_KEY and configure to be sent via API call to model
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create messages as required by the API
system_message = """
"""
instructions = """
"""

#The chat model comes with two variables placeholders:
# system message: how we want our model to behave(in this case an AI Assistant)
# and instructions (or query), where the user will ask the model its questions.
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": instructions}
]


def get_chat_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model= model,
        messages= messages,
        temperature= 0
    )
    return response.choices[0].message['content']