#1. Setting the configuration of the webpage:
from typing import Optional, Union, Any
from uuid import UUID

import streamlit  as st
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from numpy.f2py.cfuncs import callbacks
from streamlit.external.langchain import StreamlitCallbackHandler

st.set_page_config(page_title="Travlet")
st.header(('Welcome to Travlet, your travel assistant with Internet access. What are you planning for your next trip?'))

#2 Initializing the LangChain backbone components we need.
from langchain.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents.agent_toolkits import create_retriever_tool,create_conversational_retrieval_agent
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["SERPAPI_API_KEY"]
openai_api_key =os.getenv("OPENAI_API_KEY")

search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

import os

# Base directory and filename
base_dir = '/conversational_ai_chat_bot/'
filename = 'italy_travel.pdf'

# Construct the full path
full_path = os.path.join(base_dir, filename)
print(full_path)



import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

uploaded_file = st.file_uploader("italy_travel.pdf")
raw_documents = []
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    i = 1
    for page in reader.pages:
        raw_documents.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        i += 1

#raw_documents = PyPDFLoader(full_path).load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

llm = ChatOpenAI(openai_api_key)
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(),
        "italy_travel",
        "Searches and returns documents regarding Italy."
    )]

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)


#3 -Setting the input box for the user with a placeholder question:
user_query = st.text_input("**Where are you planning your next vacation?**", placeholder="Ask me anything!")

#4 - Setting Streamlit’s session states. Session state is a way to share variables between reruns, for each user session.
# We can use the session state API to initialize, read, update, and delete variables in the session state

if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "memory" not in st.session_state:
    st.session_state["memory"] = memory

#5 - Display the whole conversation.
# For each message, it creates a Streamlit element called st.chat_message that displays a chat message in a nice format:
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

#6 -Configuring the AI assistant to respond when given a user’s query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response= agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


#7 - Add a button to clear the history of the conversation and start from scratch:
if st.sidebar.button('"Reset chat history"'):
    st.session_state.messages = []


# Give streaming experience to user while chatting with bot

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text= ''):
        self.container = container
        self.text = initial_text

    def non_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


#The StreamHandler is designed to capture and display streaming data, such as text or other content, in a designated container.
with st.chat_message("assistant"):
    stream_handler = StreamHandler(st.empty())
    llm = ChatOpenAI(streaming=True,callbacks=[stream_handler])
    response=llm.invoke(st.session_state.messages)
    st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))