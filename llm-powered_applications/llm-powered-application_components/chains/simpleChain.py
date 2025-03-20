from langchain import OpenAI, LLMChain,PromptTemplate

# Template
template ="""Sentence:{sentence}
Translation in {language}:
"""

prompt= PromptTemplate(template=template, input_variables=["sentence","language"])

# Put Prompt Template into an LLMChain.
llm = OpenAI(temperature=0)
llm_chain = LLMChain(prompt=prompt,llm=llm)
llm_chain.predict(sentence = "the cat is on the table", language= "Spanish")



