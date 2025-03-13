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


if __name__ == "__main__":
    user_dataset()














