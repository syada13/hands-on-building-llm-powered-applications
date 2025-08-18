from prompting.general_prompting_principles.chat_completion import get_chat_completion

system_message ="""
You are an AI assistant that summarizes articles.
To complete this task, do the following subtasks:
Read the provided article context comprehensively and identify the main topic and key points
Generate a paragraph summary of current article context that captures the essential information and conveys the main idea
Print each step of the process
Article:
"""

article="""
Recurrent neural networks, long short-term memory, and gated recurrent neural networks
in particular, […]
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": article}
]


model_response = get_chat_completion(messages)
print(model_response)
