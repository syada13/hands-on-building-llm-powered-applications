from  chat_completion import get_chat_completion

system_message ="""
You are an AI assistant specialized in solving riddles.
Given a riddle, solve it the best you can.
Provide a clear justification of your answer and the reasoning behind behind it.
Riddle:
"""

riddle="""
What has a face and two hands, but no arms or legs?
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": riddle}
]


model_response = get_chat_completion(messages)
print(model_response)
