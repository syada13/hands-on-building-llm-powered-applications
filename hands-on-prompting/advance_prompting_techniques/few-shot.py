from prompting.general_prompting_principles import chat_completion

system_message ="""
You are an AI Marketing assistant. You help users to create taglines for new product names.
Given a product name, produce a tagline similar to the following examples:
Peak Pursuit - Conquer Heights with Comfort
Summit Steps - Your Partner for Every Ascent
Crag Conquerors - Step Up, Stand Tall
Product name:
"""

product_name='Elevation Embrace'

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": product_name}
]

response = chat_completion.get_chat_completion(messages)
print(response)

