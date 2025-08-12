
import sys
import os
from config import set_environment

sys.path.insert(0, os.path.abspath('..'))
# setting the environment variables, the keys
set_environment()

""" Query transformation/expansion: Improving retrieval through better queries
 
 Use case : Query transformation techniques are particularly useful when dealing with ambiguous queries, questions formulated by non-experts, or situations where terminology mismatches between queries and documents are common. 
 
 They do add computational overhead but can dramatically improve retrieval quality, especially for complex or poorly formulated questions.
 """

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

expansion_template = """Given the user question: {question}
Generate three alternative versions of the question that express the same information need but with different wording:
1.
"""
expansion_prompt = PromptTemplate(
  input_variables=["question"],
  template=expansion_template
)

llm = ChatOpenAI(temperature=0.7)
expansion_chain = expansion_prompt|llm|StrOutputParser()

# Generate expanded queries
original_query = "What are the effects of climate change?"
expanded_queries= expansion_chain.invoke(original_query)
print(expanded_queries)


"""Hypothetical Document Embeddings (HyDE)
HyDE uses an LLM to generate a hypothetical answer document based on the query, and then uses that document’s embedding for retrieval. This technique is especially powerful for complex queries where the semantic gap between query and document language is significant:
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="knowledge_base.json",
    jq_schema=".[].content", # This extracts the content field from each array item
    text_content=True
)
documents = loader.load()
embedder = OpenAIEmbeddings()
embeddings = embedder.embed_documents([doc.page_content for doc in documents])
vector_db = FAISS.from_documents(documents, embedder)


# Create prompt for generating hypothetical document
hyde_template = """Based on the question: {question}
Write a passage that could contain the answer to this question:"""

hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template=hyde_template
)

llm = ChatOpenAI(temperature=0.2) #responses with a moderate level of randomness
hyde_chain = hyde_prompt | llm | StrOutputParser()

# Generate hypothetical document
query = "What dietary changes can reduce carbon footprint?"
hypothetical_doc = hyde_chain.invoke(query)

# Use the hypothetical document for retrieval
embeddings = OpenAIEmbeddings()
embedded_query= embeddings.embedded_query(hypothetical_doc)
result = vector_db.similarity_search_by_vector(embedded_query,k =3)
print(results)




