
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

"""Context processing: maximizing retrieved information value
Once documents are retrieved, context processing techniques help distill and organize the information to maximize its value in the generation phase.

Contextual compression
Contextual compression extracts only the most relevant parts of retrieved documents, removing irrelevant content that might distract the generator.

LLMChainExtractor, which will iterate over the initially returned documents and extract from each only the content that is relevant to the query.

The Contextual Compression Retriever passes queries to the base retriever, takes the initial documents and passes them through the Document Compressor. The Document Compressor takes a list of documents and shortens it by reducing the contents of documents or dropping documents altogether.
"""

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# Create a basic retriever from the vector store
base_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

compressed_docs = compression_retriever.invoke("How do transformers work?")
print(compressed_docs)

"""Output:
 [Document(metadata={'source': 'Neural Network Review 2021', 'page': 42}, page_content="The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017."),
 Document(metadata={'source': 'Large Language Models Survey', 'page': 89}, page_content='GPT models are autoregressive transformers that predict the next token based on previous tokens.')]
"""



