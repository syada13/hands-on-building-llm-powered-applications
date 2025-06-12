import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]

embeddings_model = OpenAIEmbeddings( model= 'text-embedding-ada-002')
embeddings = embeddings_model.embed_documents(
    [
        "Good morning!",
        "oh, hello!",
        "I want to report an accident",
        "Sorry to hear that. May I ask your name?",
        "Sure, Suresh Yadav"
    ]
)


print("Embed documents:")
print(f"Number of vectors: {len(embeddings)} ; Dimension of each vector: {len(embeddings[0])}")
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print("Embed query:")
print(f"Dimension of the vector: {len(embedded_query)}")
print(f"Sample of the first 5 elements of the vector: {embedded_query[:5]}")



""" OUTPUT -------
Embed documents:
Number of vector: 5; Dimension of each vector: 1536
Embed query:
Dimension of the vector: 1536
Sample of the first 5 elements of the vector: 
[0.00538721214979887, -0.0005941778072156012, 0.03892524912953377, -0.002979141427204013, -0.008912666700780392]
"""



