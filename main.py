# Author: Polly Zheng
# Date: 2025-11-23
# Purpose: Week 12 Assignment

import requests
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from dotenv import load_dotenv
import os

# Load environment variable
load_dotenv()
api_key = os.getenv("RAPIDAPI_KEY")

if not api_key:
    raise ValueError("❌ RAPIDAPI_KEY not found. Did you set it in .env?")

# Get apt requests

url = "https://imdb-top-100-movies.p.rapidapi.com/"
headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "imdb-top-100-movies.p.rapidapi.com"
}

response = requests.get(url, headers=headers)
movies = response.json()
df = pd.DataFrame(movies)

# Save CSV
csv_path = "imdb_top100.csv"
df.to_csv(csv_path, index=False)

# Load and explore csv file for eaiser analysis
df = pd.read_csv(csv_path)
print("Dataset preview:")
print(df.head(), "\n")


# NLP analysis to extract plot keywords
nlp = spacy.load("en_core_web_sm")


def extract_keywords(text):
    doc = nlp(text)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "VERB"] and not token.is_stop
    ]
    return [word for word, count in Counter(keywords).most_common(5)]


df["plot_keywords"] = df["description"].apply(extract_keywords)

# Test case
print("Example extracted keywords:")
print(df[["title", "plot_keywords"]].head(), "\n")


# Movie genre analysis (not with NLP, but for final project practice)

all_genres = []
for g in df["genre"]:
    if isinstance(g, str) and g.startswith("["):
        g = eval(g)
    all_genres.extend(g)

genre_counts = Counter(all_genres)
genre_df = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"]).sort_values(
    by="Count", ascending=False
)

print("Top genres:")
for genre, count in genre_counts.most_common(10):
    print(f"- {genre}: {count}")


# Plotting genre distribution
plt.figure(figsize=(12, 6))
plt.bar(genre_df["Genre"], genre_df["Count"])
plt.xticks(rotation=45, ha="right")
plt.title("Most Popular Genres in IMDB Top 100 Movies")
plt.xlabel("Genre")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Final summary
print("\n================ FINAL SUMMARY ================")

top_genre, top_num = genre_df.iloc[0]
print(f"Most dominant genre: {top_genre} ({top_num} movies)")

print("\n Core repeated plot themes (from NLP):")
keywords_flat = [kw for sub in df["plot_keywords"].tolist() for kw in sub]
top_keywords = Counter(keywords_flat).most_common(10)
for word, count in top_keywords:
    print(f"- {word} ({count})")

print("\n Interpretation:")
print("Top-rated movies consistently emphasize emotional depth, tension, moral conflict,\n"
      "and personal transformation — which appear strongly in both genre distribution\n"
      "and textual plot themes.")
