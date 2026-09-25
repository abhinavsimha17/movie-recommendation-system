"""
prepare_kaggle_data.py
-----------------------
Converts the raw "TMDB 5000 Movie Dataset" from Kaggle into a clean,
simple CSV (title, genres, cast, director, year) that recommender.py
can load directly.

Dataset (download this first): "TMDB 5000 Movie Dataset"
https://www.kaggle.com/tmdb/tmdb-movie-metadata
Files needed: tmdb_5000_movies.csv, tmdb_5000_credits.csv

Steps to get the data:
  1) pip install kaggle
  2) Get an API token from kaggle.com -> Account -> Create New API Token
     (this downloads kaggle.json). Place it at ~/.kaggle/kaggle.json
  3) Run:  kaggle datasets download -d tmdb/tmdb-movie-metadata -p data --unzip
  4) Run this script:  python prepare_kaggle_data.py

If you don't want to bother with the Kaggle API, you can also just
click "Download" on the Kaggle dataset page in your browser and drop
the two CSVs into the data/ folder.

Output: data/movies.csv
"""

import ast
import os
import pandas as pd

RAW_MOVIES_PATH = "data/tmdb_5000_movies.csv"
RAW_CREDITS_PATH = "data/tmdb_5000_credits.csv"
OUTPUT_PATH = "data/movies.csv"

TOP_N_CAST = 3  # how many top-billed actors to keep per movie


def parse_names(json_like_string, limit=None):
    """TMDB stores genres/cast as a stringified list of dicts, e.g.
    '[{"id": 28, "name": "Action"}, ...]'. Extract just the 'name' fields."""
    try:
        items = ast.literal_eval(json_like_string)
    except (ValueError, SyntaxError):
        return []
    names = [item.get("name") for item in items if item.get("name")]
    return names[:limit] if limit else names


def parse_director(crew_json_string):
    try:
        crew = ast.literal_eval(crew_json_string)
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def main():
    if not (os.path.exists(RAW_MOVIES_PATH) and os.path.exists(RAW_CREDITS_PATH)):
        raise FileNotFoundError(
            "Couldn't find the raw Kaggle files. Expected:\n"
            f"  {RAW_MOVIES_PATH}\n  {RAW_CREDITS_PATH}\n"
            "See the instructions at the top of this script to download them."
        )

    movies = pd.read_csv(RAW_MOVIES_PATH)
    credits = pd.read_csv(RAW_CREDITS_PATH)

    # The two files share the movie title; credits.csv also has a matching id.
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", suffixes=("", "_credits"))

    df["genres"] = df["genres"].apply(lambda x: "|".join(parse_names(x)))
    df["cast"] = df["cast"].apply(lambda x: "|".join(parse_names(x, limit=TOP_N_CAST)))
    df["director"] = df["crew"].apply(parse_director)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    clean = df[["title", "genres", "cast", "director", "year"]].dropna(
        subset=["title", "genres", "cast"]
    )
    clean = clean[(clean["genres"] != "") & (clean["cast"] != "")]

    clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(clean)} cleaned movies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
