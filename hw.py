import random
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


def load_data(file_path='imdb_top_1000.csv'):
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
        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendation found."


def display_random(df=movies_df, top_n=5):
    df = df.dropna(subset=['Series_Title'])
    sampled = df.sample(n=top_n)
    recommendations = []

    for _, row in sampled.iterrows():
        polarity = TextBlob(str(row['Overview'])).sentiment.polarity
        recommendations.append((row['Series_Title'], polarity))
    return recommendations
# Display recommendations


def display_recommendations(recs, name, genre):
    print(f"{Fore.YELLOW}🎥🍿AI Analyzed movie recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive 😄" if polarity > 0 else "Negative 😔" if polarity < 0 else "neutral 😐"
        print(
            f"{Fore.CYAN}{idx}. 🎥 {title} (Polarity: {polarity:.2f}, {sentiment}, {genre})")

# Processing animation


def animation():
    for i in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)


def handle_AI(name):
    print(f"{Fore.CYAN}🔍Let's find the last movie for you")
    # Show genres in a single line
    for idx, genre in enumerate(genres, 1):
        print(f"{Fore.CYAN}{idx}. {genre}")
    print()  # To move the name to the next line

    while True:
        genre_input = input(
            f"{Fore.YELLOW}Enter genre name or number: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print(f"{Fore.RED}Invalid input. Please enter again.")

    mood = input(f"{Fore.YELLOW}How do you feel today?(Descirbe your mood): ")

    print(f"{Fore.BLUE}Analyzing mood", end="", flush=True)
    animation()  # Small animation during mood analysis
    polarity = TextBlob(mood).sentiment.polarity
    mood_desc = "Positive 😄" if polarity > 0 else "Negative 😔" if polarity < 0 else "neutral 😐"
    print(f"{Fore.GREEN}Your mood is {mood_desc} (Polarity: {polarity:.2f})")

    while True:
        rating_input = input(
            f"{Fore.YELLOW}Enter minimum IMDB rating (7.6 - 9.3) or 'skip': ").strip()
        if rating_input.lower() == 'skip':
            rating = None
            break
        try:
            rating = float(rating_input)
            if 7.6 <= rating <= 9.3:
                break
            print(f"{Fore.RED}Value out of range. Try Again.")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Try Again")

    print(f"{Fore.BLUE}Finding movies for {name}: ", end="", flush=True)
    animation()  # Animation while finding movies

    recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
    if isinstance(recs, str):
        print(f"{Fore.RED}{recs}\n")
    else:
        display_recommendations(recs, name, genre)

    while True:
        action = input(
            f"{Fore.YELLOW}Would you like some more recommendations🎥(yes/random/no): ").strip().lower()
        if action == "no":
            print(f"{Fore.GREEN}Enjoy your movie picks, {name}!")
            break
        elif action == "yes":
            recs = recommend_movies(
                genre=genre, mood=mood, rating=rating, top_n=5)
            if isinstance(recs, str):
                print(f"{Fore.RED}{recs}\n")
            else:
                display_recommendations(recs, name, genre)
        elif action == 'random':
            print(f"{Fore.BLUE}🎲 Picking random recommendations", end="", flush=True)
            animation()
            recs = display_random(top_n=5)
            display_recommendations(recs, name, "Random Genre")
        else:
            print(f"{Fore.RED}Invalid choice. Try Again \n")

# Main program


def main():
    print(f"{Fore.BLUE}🍿Welcome to your movie assistant! 🎥")
    name = input(f"{Fore.YELLOW}What is your name: ").strip()
    print(f"\n{Fore.GREEN}Greetings {name}")
    handle_AI(name)


if __name__ == "__main__":
    main()
