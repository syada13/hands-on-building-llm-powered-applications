from prompting.general_prompting_principles.chat_completion import get_chat_completion

unordered_system_message ="""
You are a sentiment analyzer. You classify conversations into three categories: positive, negative and neutral
Return only the sentiment in lowercase and without punctuation.
Conversation:
"""

conversation= """
Customer: Hi, I need some help with my order.
AI agent: Hello, welcome to our online store. I'm an AI agent and I'm here to assist you. 
Customer: I ordered a pair of shoes yesterday, but I haven't received a confirmation email yet. Can you check the status of my order?
AI agent: Sure, I can help you with that. Can you please provide me with your order number and email address?
Customer: Yes, my order number is 123456789 and my email is john.doe@example.com.
AI agent: Thank you. I have found your order in our system. It looks like your order is still being processed and it will be shipped soon. You should receive a confirmation email within 24 hours.
Customer: OK, thank you for the information. How long will it take for the shoes to arrive?
AI agent: You're welcome. According to our shipping policy, it will take about 3 to 5 business days for the shoes to arrive at your address. You can track your order online using the tracking number that will be sent to your email once your order is shipped.
Customer: Alright, sounds good. Thank you for your help.
AI agent: It's my pleasure. Is there anything else I can do for you today?
Customer: No, that's all. Have a nice day.
AI agent: Thank you for choosing our online store. Have a nice day too. Goodbye.
"""

messages_1 = [
    {"role": "system", "content": unordered_system_message},
    {"role": "user", "content": conversation}
]


model_response_1 = get_chat_completion(messages_1)
print(model_response_1)

ordered_system_message= f'''
You are a sentiment analyzer. You classify conversations into three categories: positive, negative and neutral
Return only the sentiment in lowercase and without punctuation.
Conversation:
{conversation}
Remember to return only the sentiment, in lowercase and without punctuation.
'''

messages_2 = [
    {"role": "system", "content": ordered_system_message},
    {"role": "user", "content": conversation}
]

model_response_2 = get_chat_completion(messages_2)
print(model_response_2)



