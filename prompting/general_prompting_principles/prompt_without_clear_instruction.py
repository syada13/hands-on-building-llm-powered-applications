from prompting.general_prompting_principles.chat_completion import get_chat_completion

system_message = """
You are an AI assistant that helps humans by generating tutorials given a text.
You will be provided with text.If the text contains any kind of instructions on how to proceed with something, generate a tutorial in a bullet list.
Otherwise, inform the user that the text does not contain any instructions.
Text:
"""
vague_instruction ="""
The sun is shining and dogs are running on the beach.
"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": vague_instruction}
]

# Call get_chat_completion method with prompt to get response from model
model_response = get_chat_completion(messages)
print(model_response)

