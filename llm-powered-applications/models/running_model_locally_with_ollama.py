from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_prasers import StrOutputParser

#initialize Ollama with choosen model
local_llm = ChatOllama(
  model="mistral:q4_K_M",# 4-bit quantized model (smaller memory footprint,~~2GB RAM required)
  num_gpu=1,# Number of GPUs to utilize (adjust based on hardware)
  num_threads=4,# Number of CPU threads for parallel processing
  temperature=0
)

# Create an Lanchchain Expression Language chain using the local model
prompt = PromptTemplate.from_template("Explain{concept} in simple terms")
local_chain= prompt |local_llm | StrOutputParser

#Use the chain with local model
result = local_chain.invoke({"concept":"quantum computing"})
print(result)

"""Error handling: 
   Safely call a local model with retry logic and graceful
   failure"""

def safe_model_call(llm,prompt,max_retries=2):
  retries = 0
  while retries <=max_retries:
    try:
      return llm.invoke(prompt)

    except RuntimeError as e:
        # Common error with local models when running out of VRAM
        if "CUDA out of memory" in str(e):
          print(f"GPU memory error, waiting and retrying({retries+1}/{max_retries+1})")
          time.sleep(2) # Give system time to free resources
          retries += 1
        else:
          print(f"Runtime error: {e}")
          return "An error occurred while processing your request."
          except Exception as e:
            print(f"Unexpected error calling model: {e}")
            return "Model is currently experiencing high load. Please try again later."

# Use the safety wrapper in your LCEL chain
from langchain_core.runnables import RunnableLambda
safe_llm = RunnableLambda(lambda x:safe_model_call(local_llm,x))
safe_chain = prompt | safe_llm
response = safe_chain.invoke({{"concept":"quantum computing"}})
print(response)


  