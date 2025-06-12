from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables fro, .ENV file
load_dotenv()

# Read OPENAI_API_KEY
os.environ("OPENAI_API_KEY")

# Load the document
raw_documents = TextLoader('dialogue.txt').load()

#Select a document spliter to split documents into small chunks
text_splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap=0,
    separator="\n"
)

#Split raw documents using above stated text splitter in a chunk of 50
documents = text_splitter.split_documents(raw_documents)

#Embedd chunked data using OpenAI embedding model 'text-embedding-ada-002' , and load into Vector DB/Store
db = FAISS.from_documents(documents,OpenAIEmbeddings(model='text-embedding-ada-002'))


#Embed a user’s query so that it can be used to search the most similar text chunk
# using cosine similarity as a measure
query= 'what is the reason for calling?'
docs = db.similarity_search(query)
print(docs[0])


#OUTPUT - I want to report an accident

# Retriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=retriever)
query = "What was the reason of the call?"
qa.run(query)

#OUTPUT - The reason for the call was to report an accident.








