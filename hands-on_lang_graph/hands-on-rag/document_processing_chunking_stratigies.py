
import sys
import os
from config import set_environment

# This line of code adds the parent directory of the current script or working directory to the very beginning of the Python interpreter's module search path. This ensures that any modules located in that parent directory will be found and can be imported successfully, even if they are not in the same directory as the script being run or in a standard Python installation location.

sys.path.insert(0, os.path.abspath('..'))

# Setting the environment variables, the keys .Since keys are private, they are not included in the repository.
set_environment()

# Document loader
from langchain_community.document_loaders import JSONLoader

#Load a json file
loader = JSONLoader(
  file_path="knowledge_base.json",
  jq_schema=".[].content", #This extracts the content field from each array item
  text_content=True
)

documents = loader.load()
print(documents)

"""Fixed size chunking.

Fixed-size chunking is good for quick prototyping or when document structure is relatively uniform, however, it often splits text at awkward positions, breaking sentences, paragraphs, or logical units.
"""

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
  separator=" ",# Split on spaces to avoid breaking words
  chunk_size=200,
  chunk_overlap=30
)

text_chunks = text_splitter.split_documents(documents)
print(f"Generated {len(text_chunks)} chunks from document")

"""Recursive Character Chunking
This chunking method respects natural text boundaries by recursively applying
different separators.

Recursive character chunking is the recommended default strategy for most applications. It works well for a wide range of document types and provides a good balance between preserving context and maintaining manageable chunk sizes.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_text_splitter = RecursiveCharacterTextSplitter(
  separators=["\n\n","\n","."," ",""],
  chunk_size=150,
  chunk_overlap=30
)
document = """# Introduction to RAG
Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI models.

It helps address hallucinations by grounding responses in retrieved information.

## Key Components
RAG consists of several components:
1. Document processing
2. Vector embedding
3. Retrieval
4. Augmentation
5. Generation

### Document Processing
This step involves loading and chunking documents appropriately.
"""
recursive_text_chunks = recursive_text_splitter.split_text(document)
print(recursive_text_chunks)

"""Semantic chunking
Unlike previous approaches that rely on textual separators, semantic chunking analyzes the meaning of content to determine chunk boundaries.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
semantic_text_splitter = SemanticChunker(
  embedding=embeddings,
  add_start_index=True # Include position metadata
)

semantic_chunks = semantic_text_splitter.split_text(document)
print(semantic_chunks)









