from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI

llm =OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm)
memory.save_context({"input": "hi, I'm looking for some ideas to write an essay in AI"},
                    {"output": "hello, what about writing on LLMs"})
memory.load_memory_variables({})


#OUTPUT - {'history': '\nThe human asked for ideas to write an essay in AI and the AI suggested writing on LLMs.'}

