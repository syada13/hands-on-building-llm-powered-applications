from langchain_google_vertexai import ChatVertexAI
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

#Cache

# Set up an in-memory cache and invoke an LLM:
cache = InMemoryCache()
set_llm_cache(cache)
llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0.5)
print(llm.invoke("What is the capital of UK?"))

# Check if the request-response pair has been cached:
import langchain
print(langchain.llm_cache._cache)

# String representation of the LLM instance
print(llm._get_llm_string())


# Store
from langgraph.store.memory import InMemoryStore
from langchain_google_vertexai import VertexAIEmbeddings

# Initialize a store 
in_memory_store = InMemoryStore()

# Add values to store with a specific namespace and key.Value is a dictionary
in_memory_store.put(namespace=("users", "user1"), key="fact1", value={"message1": "My name is John."})
in_memory_store.put(namespace=("users", "user1", "conv1"), key="address", value={"message": "I live in Berlin."}

# Query the value using get() method. We need a full matching namespace
print(in_memory_store.get(namespace=("users", "user1", "conv1"), key="address"))

# When using search, we can use a partial namespace path:
print(in_memory_store.search(("users", "user1", "conv1"), query="name"))



