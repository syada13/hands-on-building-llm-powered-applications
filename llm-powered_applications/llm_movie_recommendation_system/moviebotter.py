
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
import lancedb
from langchain.vectorstores import LanceDB
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from prompting.general_prompting_principles.use_delimeters import query

#Configure the application page
st.set_page_config(page_title="MovieBotter", page_icon="🎬")
st.header('🎬 Welcome to MovieBotter, your favourite movie recommender')

#Import the credentials and establish the connection to LanceDB:
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
embeddings= OpenAIEmbeddings()
uri= "data/sample-lancedb"
db= lancedb.connect(uri)
table= db.open_table('movies')
docsearch = LanceDB(connection = table,embedding=embeddings)

# Import Movie data set
movie_data = pd.read_pickle("./movies.pkl")

# Create widgets for the user to define their features and movies preferences
# Create a sidebar for user input
st.sidebar.title("Movie Recommender System")
st.sidebar.markdown("Please enter your details and preferences below:")

#Ask the user for age,gender and favourite movie
age = st.sidebar.slider(("What is your age?",1,100,25))
gender = st.sidebar.radio("What is your gender?",("Male", "Female", "Other"))
genre = st.sidebar.selectbox("What is your favourite movie genre?",movie_data.explode('genres')["genres"].unique())

# Filter the movies based on the user input
df_filtered= movie_data[movie_data['genres'].apply(lambda x: genre in x)]

# Define the parametrized prompt chunks
template_prefix = """You are a movie recommender system that help users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}"""
user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}"""
template_suffix= """Question: {question}
Your response:"""

user_info = user_info.format(age = age, gender = gender)
combined_prompt = template_prefix + '\n' + user_info +'\n' + template_suffix
print(combined_prompt)



#Setup RetrievalQA
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True)

query = st.text_input('Enter your question:', placeholder = 'What action movies do you suggest?')
if query:
    result = qa({"query": query})
    st.write(result['result'])