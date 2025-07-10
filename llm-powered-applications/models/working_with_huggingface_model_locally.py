from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

#Create pipeline with small model
llm = HuggingFacePipeline.from_model_id(
  model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task="text-generation",
  pipeline_kwargs=dict(
    max_new_token=512,
    do_sample=False,
    repetition_penalty=1.03
  )
)
chat_model =ChatHuggingFace(llm=llm)

# Use it like any other LangChain LLM
messages =[
  SystemMessage(content="You're a helpful assistant"),
  HumanMessage(content="Explain the concept of machine learning in simple terms"]

ai_generated_message = chat_model.invoke(messages)
print(ai_generated_message)