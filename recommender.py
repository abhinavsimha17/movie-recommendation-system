"""
recommender.py
----------------
A simple content-based movie recommendation system based on GENRE and CAST.

How it works (the ML part):
  1. For every movie we build a small "tag soup" string combining its
     genres and its top-billed actors.
  2. CountVectorizer turns every movie's tag soup into a numeric vector
     (bag-of-words over genres+actors).
  3. We fit a NearestNeighbors model (cosine distance) on those vectors.
     This is the actual ML model: given a movie's vector, it finds the
     k closest movies in that vector space.
  4. To recommend, we look up the query movie's vector and ask the model
     for its nearest neighbours -> "movies most like this one, by genre
     and cast."

Data:
  Uses data/movies.csv if present (produced by prepare_kaggle_data.py from
  the Kaggle "TMDB 5000 Movie Dataset"), otherwise falls back to the small
  bundled data/sample_movies.csv demo dataset so this always runs out of
  the box.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_PATH_REAL = "data/movies.csv"
DATA_PATH_SAMPLE = "data/sample_movies.csv"

# How much extra weight to give genres vs. cast when building the tag soup.
# Repeating genre words makes the vectorizer treat genre matches as more
# important than a single shared actor.
GENRE_WEIGHT = 2


class MovieRecommender:
    def __init__(self, data_path=None):
        self.data_path = data_path or self._pick_data_path()
        self.df = self._load_data(self.data_path)
        self.vectorizer = None
        self.vectors = None
        self.model = None
        self._build()

    @staticmethod
    def _pick_data_path():
        if os.path.exists(DATA_PATH_REAL):
            return DATA_PATH_REAL
        return DATA_PATH_SAMPLE

    @staticmethod
    def _load_data(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=["title", "genres", "cast"]).reset_index(drop=True)
        # Normalize "Action|Crime" style lists into a single lowercase string
        df["genres_list"] = df["genres"].apply(lambda s: [g.strip() for g in s.split("|")])
        df["cast_list"] = df["cast"].apply(lambda s: [c.strip() for c in s.split("|")])
        return df

    def _build_tag_soup(self, row):
        genre_tokens = ["_".join(g.lower().split()) for g in row["genres_list"]] * GENRE_WEIGHT
        cast_tokens = ["_".join(c.lower().split()) for c in row["cast_list"]]
        return " ".join(genre_tokens + cast_tokens)

    def _build(self):
        self.df["tags"] = self.df.apply(self._build_tag_soup, axis=1)

        self.vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
        self.vectors = self.vectorizer.fit_transform(self.df["tags"])

        n_neighbors = min(11, len(self.df))  # self + up to 10 recommendations
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.vectors)
        self._n_neighbors = n_neighbors

    def _find_index(self, title):
        matches = self.df.index[self.df["title"].str.lower() == title.lower()]
        if len(matches) == 0:
            # fall back to a loose "contains" match
            matches = self.df.index[self.df["title"].str.lower().str.contains(title.lower())]
        if len(matches) == 0:
            return None
        return matches[0]

    def recommend(self, title, top_n=5):
        """Return up to top_n movies most similar to `title` by genre+cast."""
        idx = self._find_index(title)
        if idx is None:
            return None, f"No movie found matching '{title}'."

        k = min(top_n + 1, len(self.df))  # +1 because the movie itself is its own nearest neighbor
        distances, indices = self.model.kneighbors(self.vectors[idx], n_neighbors=k)

        results = []
        for dist, i in zip(distances[0], indices[0]):
            if i == idx:
                continue
            row = self.df.iloc[i]
            similarity = round(1 - dist, 3)  # cosine distance -> similarity
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "cast": row["cast"],
                "similarity": similarity,
            })
        return results[:top_n], None

    def list_titles(self):
        return self.df["title"].tolist()


def _cli():
    rec = MovieRecommender()
    print(f"Loaded {len(rec.df)} movies from {rec.data_path}")
    print("Type a movie title to get recommendations (or 'quit' to exit).\n")
    while True:
        title = input("Movie: ").strip()
        if title.lower() in ("quit", "exit"):
            break
        results, error = rec.recommend(title, top_n=5)
        if error:
            print(f"  {error}\n")
            continue
        print(f"\nBecause you liked '{title}', you might also like:")
        for r in results:
            print(f"  - {r['title']}  ({r['genres']})  [similarity: {r['similarity']}]")
        print()


if __name__ == "__main__":
    _cli()
