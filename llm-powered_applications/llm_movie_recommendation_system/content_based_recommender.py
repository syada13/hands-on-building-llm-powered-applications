import pandas as pd

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




if __name__ == "__main__":
    user_dataset()

















