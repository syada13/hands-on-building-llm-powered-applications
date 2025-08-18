
# Skipping API configuration
# Initialize the embeddings model
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create embeddings for the original example sentences
text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet"
text3 = "Python is a programming language"

#Get embeddings using Langchain
embeddings = embeddings_model.embed_documents([text1,text2,text3])

# These similar sentences will have similar embeddings
embedding1 = embeddings[0] # Embedding for "The cat sat on the mat"
embedding2 = embeddings[1] # Embedding for "A feline rested on the
carpet"
embedding3 = embeddings[2] # Embedding for "Python is a programming
language"

"""
 Embedding dimensions: The embedding dimension is like the number of coordinates/numebrical representation needed to pinpoint each word's location in that space. For example, a 300-dimensional embedding means each word is represented by a list of 300 numbers. 
 """

print(f"Numbers of documents: {len(embeddings)}")
print(f"Dimensions per embedding: {len(embeddings[0])}")










