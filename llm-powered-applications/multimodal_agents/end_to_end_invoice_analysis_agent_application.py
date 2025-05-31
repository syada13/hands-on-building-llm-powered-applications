import os
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from IPython import display

# Load Azure AI services environment variables
load_dotenv()
azure_cogs_key = os.environ["AZURE_COGS_KEY"]
azure_cogs_endpoint = os.environ["AZURE_COGS_ENDPOINT"]
azure_cogs_region = os.environ["AZURE_COGS_REGION"]
openai_api_key = os.environ["OPENAI_API_KEY"]

#Configure our toolkit
toolkit = AzureCognitiveServicesToolkit()

#Task - Tell us all the men's Stock Keeping Units(SKUs) on the invoice.
invoice_image_url = "https://www.whiteelysee.fr/design/wp-content/uploads/2022/01/custom-t-shirt-order-form-template-free.jpg"
mens_skus = agent.run(f"what are all men's skus? {invoice_image_url}")
print(mens_skus)

# Task - Ask for multiple information (women’s SKUs, shipping address, and delivery dates) as follows.

#Negative Scenario - The delivery date is not specified, as we want our agent not to hallucinate:

agent.run(f"Give me the following information about the invoice:women's SKUs, shipping address and delivery date. {invoice_image_url}")

# Task - Leverage the text2speech tool to produce the audio of the response and read it aloud:
womens_skus_audio = agent.run(f"extract women's SKUs in the following invoice, then read it aloud to me:{invoice_image_url}")
display.display(womens_skus_audio)

#Task - Customize the default agent prompt.
   #Step 1 - Inspect the default template

print(agent.agent.llm_chain.prompt.messages[0].prompt.template)

  #step 2 - Customize template giving specific instructions.
  # In particular, we want the agent to produce the audio output without the user explicitly asking for it:

PREFIX = """
You are an AI assistant that help users to interact with invoices.
You extract information from invoices and read it aloud to users.
You can use multiple tools to answer the question.
Always divide your response in 2 steps:
1. Extracting the information from the invoice upon user's request
2. Converting the transcript of the previous point into an audio file
ALWAYS use the tools.
ALWAYS return an audio file using the proper tool.
You have access to the following tools:
"""

agent = initialize_agent(tools, model,
  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose = True,
  agent_kwargs={'prefix':PREFIX}
  )

# Run the agent
agent.run(f"what are women's SKUs in the following invoice?:{invoice_image_url}")



'''
Output:
> Entering new AgentExecutor chain...
I will need to use the azure_cognitive_services_form_recognizer tool to extract the information from the invoice.
Action:
```
{
  "action": "azure_cognitive_services_form_recognizer",
  "action_input": {
    "query": "https://www.whiteelysee.fr/design/wp-content/uploads/2022/01/custom-t-shirt-order-form-template-free.jpg"
  }
}
```
Observation: Content: PURCHASE ORDER TEMPLATE […]
Observation: C:\Users\suresh\AppData\Local\Temp\tmpx1n4obf3.wav
Thought:Now that I have provided the answer, I will wait for further inquiries.
'''