
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message ="""
You are a binary classifier for sentiment analysis.
Given a text, based on its sentiment, you classify it into one of two categories: positive or negative.
You can use the following texts as examples:
Text: "I love this product! It's fantastic and works perfectly."
Positive
Text: "I'm really disappointed with the quality of the food."
Negative
Text: "This is the best day of my life!"
Positive
Text: "I can't stand the noise in this restaurant."
Negative
ONLY return the sentiment as output (without punctuation).
Text:
"""

import numpy as np
import pandas as pd
df = pd.read_csv('movie.csv', encoding='utf-8')
df['label'] = df['label'].replace({0:'Negative', 1:'Positive'})
df.head()

# Test the performance of our model over a sample of 10 observations of this dataset:
df = df.sample(n=10, random_state=42)


def process_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']
df['predicted'] = df['text'].apply(process_text)
print(df)







