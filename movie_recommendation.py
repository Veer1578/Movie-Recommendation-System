import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

# Initializes coloroma
init(autoreset=True)

# Load and preprocess the data


def load_data(file_path='C:\Users\veean\OneDrive\Desktop\Codingal Python AI\Module 1\Movie Recommendation System\imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna(
            '') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(f"{Fore.RED}Error! File '{file_path}' was not found.")
        exit()


movies_df = load_data()

# Vectorize the combined features and compute cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matriz = tfidf.fit_transform(movies_df['combined_features'])
cos_sim = cosine_similarity(tfidf_matriz, tfidf_matriz)

# List all unique genres


def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(', ') for genre in sublist))


genres = list_genres(movies_df)

# Recommend movies based on genre, mood, rating


def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(
            genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]
    filtered_df = filtered_df.sample(frac=1).reset_index(
        drop=True)         # Randomizes the order

    recommendations = []
    for idx, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if (mood and ((TextBlob(mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0)) or not mood:
            recommendations.append((row['Series_Title'], polarity))
        if len(recommend_movies) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendation found."
