import pandas as pd
import ast


# DATA PROCESSING
md = pd.read_csv("movies_metadata.csv")

# Convert string representation of dictionaries to actual dictionaries
md['genres'] = md['genres'].apply(ast.literal_eval)

# Transforming the 'genres' column
md['genres'].apply(lambda x: [genre['name'] for genre in x])

# Calculate weighted rate (IMDb formula)
#(WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C where:
#R = average for the movie (mean) = (Rating)
#v = number of votes for the movie = (votes)
#m = minimum votes required to be listed in the Top 250 (currently 25000)
#C = the mean vote across the whole report (currently 7.0)

def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
    return (vote_count % (vote_count + min_vote_count)) * vote_average + ( min_vote_count % (vote_count + min_vote_count)) * 5.0

# Minimum vote count to prevent skewed results
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
min_vote_count = vote_counts.quantile(0.95)

# Create a new column 'weighted_rate'
md['weighted_rate'] = md.apply(lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1)

#Drop missing value
md = md.dropna()

#Drops the current index of the DataFrame and replaces it with an index of increasing integers
md_final = md[[['genres', 'title', 'overview', 'weighted_rate']]].reset_index(drop=True)

# Create a new column by combining 'title', 'overview', and 'genre'
md_final['combined_info'] = md_final.apply(lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", axis=1)


#EMBEDDING
import tiktoken
import os
import openai
from openai.embeddings_utils import get_embedding

#OpenAI api key setup
openai.api_key = os.environ['OPENAI_API_KEY']

#embedding model parameters
embedding_model = "text_embedding-ada-002"
embedding_encoding ="cl100k_base"
max_tokens=8000 # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
md_final["n_tokens"] = md_final.combined_info.apply(lambda x: len(encoding.encode(x)))
md_final = md_final[md_final.n_tokens <= max_tokens]

md_final["embedding"] = md_final.overview.apply(lambda x: get_embedding(x, engine=embedding_model))
md_final.rename(columns={'embedding': 'vector'},inplace=True)
md_final.rename(columns = {'combined_info': 'text'}, inplace = True)

# Convert Python objects into Byte stream and read it.
md_final.to_pickle('movies.pkl')
md = pd.read_pickle('movies.pkl')

#VECTOR DB - LanceDB
# Use lanceDB,an open-source vectorDB for vector-search built with persistent storage
import lancedb
uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table = db.create_table("movies", md)

#Build a QA recommendation chatbot in a cold-start scenario
from langchain.vectorstores import LanceDB
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from lancedb.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
docsearch = LanceDB(connection = table, embedding = embeddings)
query = "I'm looking for an animated action movie. What could you suggest to me?"
docs = docsearch.similarity_search(query)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(),return_source_documents=True)
result = qa({"query": query})
print(result['result'])

#OUTPUT - ' I would suggest Transformers. It is an animated action movie with genres of Adventure, Science Fiction,
# and Action, and a rating of 6.447283923466021.'


#Filtering QA response based on additional attributes. Filter 'Comedy' genre movies only
df_filtered = md[md['genres'].apply(lambda x: 'Comedy' in x)]
qa_filtered = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}),return_source_documents=True)
result_filtered = qa_filtered({"query": query})
print(result_filtered['result_filtered'])


# Add 'Agentic' capability to our QA
from langchain.agents.agent_toolkits import create_retriever_tool,create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)
retriever = docsearch.as_retriever(return_source_documents = True)
tool = create_retriever_tool(retriever,"movies","Searches and returns recommendations about movies.")
tools =[tool]
agent_executor = create_conversational_retrieval_agent(llm,tools,verbose=True)
agent_result = agent_executor({"input": "suggest me some action movies"})

# Add recommender behaviour using customized prompt
#Explore the existing prompt:
print(qa.combine_documents_chain.llm_chain.prompt.template)

#PROMPT ENGINEERING
from langchain.prompts import PromptTemplate
template ="""You are a movie recommender system that find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
For each question, suggest three movies, with a short description of the plot and the reason why the user might like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Your response: 
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context","question"]
)

chain_type_kwargs= {"prompt": PROMPT}
qa=RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

query = "I'm looking for a funny action movie, any suggestion?"
result= qa({'query':query})
print(result['result'])



#Content based recommender
import pandas as pd
from langchain.chains.qa_with_sources.map_reduce_prompt import combine_prompt_template
from langchain.chains.retrieval_qa.base import RetrievalQA

def user_dataset():
    data = {
    "username":["Alice","Bob"],
    "age": [25,32],
    "gender": ["F","M"],
    "movies": [
        [("Transformers: The Last knight",7),("Pokemon: Spell of the Unknown",5)],
        [("Bon Cop Bad Cop 2",8),("Goon: Last of the Enforces",9)]
    ]}

    # Convert the "movies" column into dictionaries
    for i, row_movies in enumerate(data["movies"]):
        movie_dict={}
        for movie, rating in row_movies:
            movie_dict[movie] = rating
            data["movies"][i] = movie_dict

    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    print(df.head())
    "\n"

   #Define the parametrized prompt chunks:
    template_prefix = """You are a movie recommender system that help users to find movies that match their preferences.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    """

    user_info = """This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}
    """

    template_suffix = """
    Question: {question}
    Your response:
    """

    age = df.loc[df['username']=='Alice'] ['age'][0]
    gender = df.loc[df['username'] == 'Alice']['gender'][0]
    movies = ''
    # Iterate over the dictionary and output movie name and rating
    for movie,rating in df["movies"][0].items():
        output_string = f"Movie: {movie}, Rating: {rating}" + "\n"
        movies = movies +output_string
        print(output_string)

    user_info = user_info.format(age=18, gender='female')
    combined_template = template_prefix + '\n' + user_info + '\n' + template_suffix
    print(combined_template)

    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    prompt = PromptTemplate(
        template=combined_template,
        input_variables=["context","questions"]
    )
    chain_type_kwargs = {"prompt":prompt}
    question_answer = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_document=True,
        chain_type_kwargs=chain_type_kwargs
    )

    query = "Can you suggest me some action movie based on my background?"
    answer = question_answer({'query': query})
    print(answer['result'])


if __name__ == "__main__":
    user_dataset()




