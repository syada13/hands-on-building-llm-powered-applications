
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
temperature = 0

llm = ChatOpenAI(temperature = 0)
messages = [
    SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
    HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")
]

output = llm(messages)
print(output.content)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm =llm,
    memory=memory,
    Verbose=True
)

# Conversation between Chat bot and user with ability to keep conversation history in memory
conversation.run("Hi there !")
conversation.run("What is the most iconic place in Italy?")
conversation.run("What kind of other events?")

#Make conversation interactive
while True:
    query = input('you:')
    if query == 'q':
        break
    output = conversation({'input':query})
    print('User', query)
    print('AI System', output['response'])


# Add a non-parametric knowledge
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load Italy_travel.pdf
raw_documents = PyPDFLoader("../frontend/italy_travel.pdf").load()

#Initialize TextSplitter to split loaded document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

# Split raw_documents using Text Splitter
documents = text_splitter.split_documents(raw_documents)

#Embed text chunks using the by default embedding model = text-embedding-ada-002 and save into FAISS vector store.
db = FAISS.from_documents(documents,OpenAIEmbeddings())

#Use ConversationBufferMemory to keep chat history in memory
memory = ConversationBufferMemory(
    memory_key ='chat_history',
    return_messages=True
)

# Let's interact with Conversation now.
# The ConversationalRetrievalChain uses a prompt template called CONDENSE_QUESTION_PROMPT,
# which merges the last user’s query with the chat history, so that it results as just one query to the retriever.
# If you want to pass a custom prompt, you can do so using the condense_question_prompt parameter in the ConversationalRetrievalChain.from_llm module.

query_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory,
    verbose=True
)

query_chain.run({'question':'Give me some review about Pantheon'})


#Make our chat bot agentic
from langchain.agents.agent_toolkits import create_retriever_tool,create_conversational_retrieval_agent
from langchain.tools import Tool
from langchain import SerpAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["SERPAPI_API_KEY"]
search = SerpAPIWrapper()


#create_retriever_tool method creates a custom tool that acts as a retriever for an agent. It will need a database to retrieve from, a name,
# and a short description, so that the model can understand when to use it.

tools =[
      Tool.from_function(
          func=search.run,
          name="Google Search",
          description="useful for when you need to answer questions about current events"
      ),
      create_retriever_tool(
      db.as_retriever(),
     "italy_travler",
    "Searches and returns documents regarding Italy."
)]


agent_executor = create_conversational_retrieval_agent(llm,tools,memory_key='chat_history', verbose=True)

#Question -1
#The model doesn’t need external knowledge to answer the question, hence it is responding without invoking any tool.
agent_executor({"input": "what can I visit in India in 3 days?"})

#Question -2.
#The agent will invoke the Google Search tool; this is due to the reasoning capability of the underlying
# gpt-3.5-turbo model, which captures the user’s intent and dynamically understands which tool to use to accomplish the request.

agent_executor({"input": "What is the weather currently in Delhi?"})

#Question -3 The agent will invoke the document retriever to provide the  output pertaining to Italy information saved in FAISS vector store
agent_executor({"input": "I’m traveling to Italy. Can you give me some suggestions for the main attractions to visit?"})
