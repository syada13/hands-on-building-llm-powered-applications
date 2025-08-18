
import sys
import os
from config import set_environment

sys.path.insert(0, os.path.abspath('..'))
# setting the environment variables, the keys
set_environment()

""" 
Query transformation/expansion: Improving retrieval through better queries
 
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


"""
Hypothetical Document Embeddings (HyDE)

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

"""
Context processing: maximizing retrieved information value.

Use cases : Context processing techniques are especially valuable when dealing with lengthy documents where only portions are relevant, or when providing comprehensive coverage of a topic requires diverse viewpoints. They help reduce noise in the generator’s input and ensure that the most valuable information is prioritized.


Once documents are retrieved, context processing techniques help distill and organize the information to maximize its value in the generation phase.

Contextual compression: Contextual compression extracts only the most relevant parts of retrieved documents, removing irrelevant content that might distract the generator.

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


"""Maximum Marginal Relevance (MMR):
The max_marginal_relevance_search method is used to find relevant but diverse documents, reducing redundancy. This is achieved by controlling diversity using a lambda multiplier, where 0 represents max diversity and 1 represents max relevance. 

Use cases : The method is useful in applications like recommendation systems, similarity searches, and clustering, where understanding the relationship between data points based on their embeddings is essential.
"""

mmr_result = vector_db.max_marginal_relevance_search(
  query="what are transformers models?",
  k=5, # Number of documents to return
  fetch_k=20 # Number of documents to initially fetch
  lambda_mult=0.5 # Diversity parameter (0 = max diversity, 1 = max relevance) 
)
print(mmr_result)


"""
Response enhancement: Improving generator output

Source attribution:Source attribution explicitly connects generated information to the retrieved sources, helping users verify facts and understand where information comes from.

Use cases : Educational, medical and legal advice applications where accuracy and transparency are important.

Eaxample:
1. Retrieving relevant documents for a query
2. Formatting each document with a citation number
3. Using a prompt that explicitly requests citations for each fact
4. Generating a response that includes inline citations ([1], [2], etc.)
5. Adding a references section that links each citation to its source
"""

from langchain_core.documents import Document

# Documents 
documents =[
    Document(
        page_content="The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017.",
        metadata={"source": "Neural Network Review 2021", "page": 42}
    ),
    Document(
        page_content="BERT uses bidirectional training of the Transformer, masked language modeling, and next sentence prediction tasks.",
        metadata={"source": "Introduction to NLP", "page": 137}
    ),
    Document(
        page_content="GPT models are autoregressive transformers that predict the next token based on previous tokens.",
        metadata={"source": "Large Language Models Survey", "page": 89}
    )
]


# Source attribution prompt template
from langchain_core.prompts import ChatPromptTemplate

# Create a vector store and retriever
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

attribution_prompt = ChatPromptTemplate.from_template("""
You are a precise AI assistant that provides well-sourced information.
Answer the following question based ONLY on the provided sources. For each fact or claim in your answer,include a citation using [1], [2], etc. that refers to the source. Include a numbered reference list at the end.
Question: {question}
Sources:
{sources}
Your answer:
""")

# Create a function/method to format the sources with citation numbers
def format_sources_with_citations(docs):
    formatted_sources: []
    for i, doc in enumerate(docs):
        source_info = f"[{i}] {doc.metadata.get('source','Unknown source')}"
        if doc.metadata.get('page'):
            source_info += f",page {doc.metadata['page']}"
        formatted_sources.append(f"{source_info}\n{doc.page_content}")
    return "\n\n".join(formatted_sources)


# Build the RAG chain with source attribution
def generate_attributed_response(question):
    retrieved_docs = retriever.invoke(question)
    sources_formatted = format_sources_with_citations(retrieved_docs)
    attribution_chain =(attribution_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser())

    response = attribution_chain.invoke({
        "question": question,
        "sources": sources_formatted
    })
    return response


#Testing :
question = "How do transformer models work and what are some examples?"
answer = generate_attributed_response(question)
print(answer)

"""Output:

Transformer models are a type of neural network architecture that relies on self-attention mechanisms to process input data. They were first introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017 [1]. One example of a transformer model is BERT, which utilizes bidirectional training of the Transformer, masked language modeling, and next sentence prediction tasks [2]. Another example is the GPT series of models, which are autoregressive transformers that predict the next token based on previous tokens [3].

Reference List:
[1] Neural Network Review 2021, page 42
[2] Introduction to NLP, page 137
[3] Large Language Models Survey, page 89
"""


"""Self-consistency Checking:

Self-consistency checking verifies that generated responses accurately reflect the information in retrieved documents, providing a crucial layer of protection against hallucinations.

Verify if a generated answer is fully supported by the retrieved documents.
    Args:
        retrieved_docs: List of documents used to generate the answer
        generated_answer: The answer produced by the RAG system
        llm: Language model to use for verification
    Returns:
        Dictionary containing verification results and any identified issues

"""

from typing import List, Dict

def verify_response_accuracy(
    retrieved_docs: List[Document],
    generated_answer: str,
    llm: ChatOpenAI = None
) -> Dict:
    if llm is None:
        llm = ChatOpenAI(model="gpt-turbo-3.5",temperature= 0)

    # Create context from retrieved document
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Define fact-checking template
    verification_template = """As a fact-checking assistant, verify whether the following answer is fully supported
    by the provided context. Identify any statements that are not supported or contradict the context.
    
    Context:
    {context}
    
    Answer to verify:
    {answer}
    
    Perform a detailed analysis with the following structure:
    1. List any factual claims in the answer
    2. For each claim, indicate whether it is:
       - Fully supported (provide the supporting text from context)
       - Partially supported (explain what parts lack support)
       - Contradicted (identify the contradiction)
       - Not mentioned in context
    3. Overall assessment: Is the answer fully grounded in the context?
    
    Return your analysis in JSON format with the following structure:
    {{
      "claims": [
        {{
          "claim": "The factual claim",
          "status": "fully_supported|partially_supported|contradicted|not_mentioned",
          "evidence": "Supporting or contradicting text from context",
          "explanation": "Your explanation"
        }}
      ],
      "fully_grounded": true|false,
      "issues_identified": ["List any specific issues"]
    }}
    """

    #Define the verification prompt that instructs the LLM to perform a detailed fact-checking analysis.
    verification_prompt = ChatPromptTemplate.from_template(verification_template)

    # Create verification chain using LCEL
    verification_chain = (
        verification_prompt 
        | llm 
        | StrOutputParser()
    )

    # Run verification
    verification_chain.invoke({
        "context": context,
        "answer": generated_answer
    })
    return result


#Testing
retrieved_docs = [
    Document(page_content="The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It relies on self-attention mechanisms instead of recurrent or convolutional neural networks."),
    Document(page_content="BERT is a transformer-based model developed by Google that uses masked language modeling and next sentence prediction as pre-training objectives.")
]

generated_answer = "The transformer architecture was introduced by OpenAI in 2018 and uses recurrent neural networks. BERT is a transformer model developed by Google."

verification_result = verify_response_accuracy(retrieved_docs, generated_answer)
print(verification_result)

"""OUTPUT
{
    "claims": [
        {
            "claim": "The transformer architecture was introduced by OpenAI in 2018",
            "status": "contradicted",
            "evidence": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017.",
            "explanation": "The claim is contradicted by the fact that the transformer architecture was actually introduced in 2017 by Vaswani et al., not by OpenAI in 2018."
        },
        {
            "claim": "The transformer architecture uses recurrent neural networks",
            "status": "contradicted",
            "evidence": "It relies on self-attention mechanisms instead of recurrent or convolutional neural networks.",
            "explanation": "The claim is contradicted by the fact that the transformer architecture does not use recurrent neural networks, but rather self-attention mechanisms."
        },
        {
            "claim": "BERT is a transformer model developed by Google",
            "status": "fully_supported",
            "evidence": "BERT is a transformer-based model developed by Google that uses masked language modeling and next sentence prediction as pre-training objectives.",
            "explanation": "This claim is fully supported by the provided context."
        }
    ],
    "fully_grounded": false,
    "issues_identified": ["The answer contains incorrect information about the introduction of the transformer architecture and its use of recurrent neural networks."]
}
"""



















