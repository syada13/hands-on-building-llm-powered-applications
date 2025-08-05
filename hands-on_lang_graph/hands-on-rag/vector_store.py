
import sys
import os
from config import set_environment
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Setting the environment variables, the keys
sys.path.insert(0, os.path.abspath('..'))
set_environment()


# Create embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create some sample documents with explicit IDs
docs=[
  Document(page_content="Language models content",metadata={"id":"doc_1"}),
  Document(page_content="Information about Vector databases",metadata={"id":"doc_2"})
  Document(page_content="Retrieval systems detail",metadata={"id":"doc_3"})
]

#Create vector store db instance
vector_store = Chroma(embedding_function=embeddings_model)


# Add documents to vector store with explicit IDs
vector_store.add_documents(docs)

# Similarity Search with appropriate k value
found_documents= vector_store.similarity_search(
  "How do language models work?", k=2
)

"""
Maximum marginal relevance(MMR): 
max_marginal_relevance_search function is used to retrieve documents from a vector store, and takes in a query, the number of documents to return (k), the number of documents to consider (fetch_k), and a lambda value (lambda_mult). The lambda value controls the trade-off between relevance and diversity of the retrieved documents.(0=max diversity, 1=max relevance)
"""

query = "How does LangChain work?"
k = 3 
fetch_k = 10 
lambda_mult = 0.5  

# Perform the search to find relevant BUT diverse documents (reduce redundancy)
results = vector_store.max_marginal_relevance_search(
    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
)

# Print the results
for result in results:
    print(result)