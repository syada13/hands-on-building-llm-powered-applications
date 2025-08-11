import sys
import os
from config import set_environment

sys.path.insert(0, os.path.abspath('..'))
set_environment()

#Basic RAG implementation
# For query transformation
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

#For basic RAG implementation
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#1. Load documents
loader = JSONLoader(
  file_path="knowledge_base.json",
  jq_schema=".[].content", #This extracts the content field from each array item
  text_content=True
)

loaded_documents = loader.load()

#2. Convert loaded documents to vectors 
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = embedder.embed_documents(
  [doc.page_content for doc in loaded_documents])

#3. Store embeddings in Vector store - FAISS
vector_db = FAISS.from_documents(loaded_documents,embedder)

#Retrieve similar docs
query= "What are the effects of climate change?"
results = vector_db.similarity_search(query)
print(results)

#KNN Retriever
from langchain_community.retrievers import KNNRetriever
retriever = KNNRetriever.from_documents(loaded_documents,embedder)
results = retriever.invoke("What are the effects of climate change?")

#External Search API Retriever
from langchain_community.retrievers.pubmed import PubMedRetriever
retriever = PubMedRetriever()
results = retriever.invoke("COVID research")



