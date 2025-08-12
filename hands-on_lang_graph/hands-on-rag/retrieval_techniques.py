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


#Hybrid retrieval: Combining semantic and keyword search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Setup semantic retriever
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Setup lexical retriever to improve precision
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Combine retrievers using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
  retrievers=[vector_retriever,bm25_retriever],
  weights=[.75,.25] # Weight semantic search higher than keyword search
)
results = hybrid_retriever.get_relevant_documents("climate change impacts")
print(results)

"""Re-ranking: 

It is a post-processing step that can follow any retrieval method, including hybrid retrieval:

1. First, retrieve a larger set of candidate documents
2. Apply a more sophisticated model to re-score documents
3. Reorder based on these more precise relevance scores

Cohere rerank: This compressor uses a ranking-based approach to detect and remove unimportant and irrelevant tokens from the retrieved documents.
"""

#1. Compress document
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# Initialize the compressor
compressor = CohereRerank(top_n=3)

# Create a compression retriever
compression_retriever = ContextualCompressionRetriever(
  base_compressor=compressor,
  base_retriever=base_retriever
)

# Original documents
print("Original documents:")

original_docs = base_retriever.get_relevant_documents("How do transformers work?")
for i, doc in enumerate(original_docs):
  print(f"Doc{i}:  {doc.page_content[:100]}----")

#Compressed documents
print("\nCompressed documents:")
compressed_docs = compression_retriever.get_relevant_documents("How do transformers work?")
for i, doc in enumerate(compressed_docs):
    print(f"Doc {i}: {doc.page_content[:100]}...")

""" Hybrid retrieval vs re-ranking: 
Hybrid retrievalfocuses on how documents are retrieved, 
re-ranking focuses on how they’re ordered after retrieval. 

Use case : These approaches can, and often should, be used together in a pipeline. When evaluating re-rankers, use position-aware metrics like Recall@k, which measures how effectively the re-ranker surfaces all relevant documents in the top positions.