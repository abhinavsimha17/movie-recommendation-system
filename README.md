# 🎬 Movie Recommender — Genre & Cast Based (ML Content-Based Filtering)

A simple, portfolio-ready movie recommendation system that suggests movies
similar to one you like, based on **genre** and **cast (actors)**, built
with scikit-learn and (optionally) the Kaggle **TMDB 5000 Movie Dataset**.

## How it works

This is a **content-based recommender**, not a rating-based one — it
doesn't need any user history, just one movie you like.

1. **Feature engineering** — every movie is turned into a short "tag soup"
   string of its genres and its top-billed cast (e.g. `action crime drama
   christian_bale heath_ledger`). Genres are repeated to weight them more
   heavily than a single shared actor.
2. **Vectorization** — `CountVectorizer` (scikit-learn) turns each movie's
   tag soup into a numeric vector — a bag-of-words over the whole
   genre/actor vocabulary.
3. **The ML model** — a `NearestNeighbors` model is fit on all the movie
   vectors using **cosine distance**. Given a movie, it finds the *k*
   closest movies in that vector space.
4. **Recommendation** — look up the chosen movie's vector, ask the model
   for its nearest neighbours, and those are the recommendations.

This is the same core idea used in most "TMDB movie recommender" tutorials,
kept intentionally small and readable.

## Project structure

```
movie-recommender/
├── data/
│   └── sample_movies.csv     # small bundled demo dataset (40 movies) — no download needed
├── prepare_kaggle_data.py    # cleans the raw Kaggle TMDB CSVs into data/movies.csv
├── recommender.py            # core ML logic + CLI demo
├── app.py                    # Streamlit web UI
├── requirements.txt
└── README.md
```

## Quick start (no dataset download needed)

The project ships with a small sample dataset so you can try it immediately:

```bash
pip install -r requirements.txt
python recommender.py        # command-line demo
# or
streamlit run app.py         # web UI
```

## Using the full Kaggle dataset (recommended for a real portfolio demo)

The sample dataset is only 40 movies. For a more impressive demo, use the
**TMDB 5000 Movie Dataset**:
https://www.kaggle.com/tmdb/tmdb-movie-metadata

1. Install the Kaggle CLI and set up your API token:
   ```bash
   pip install kaggle
   # Get kaggle.json from kaggle.com -> Account -> Create New API Token
   # Place it at ~/.kaggle/kaggle.json
   ```
2. Download the dataset:
   ```bash
   kaggle datasets download -d tmdb/tmdb-movie-metadata -p data --unzip
   ```
   (Alternatively, just download the two CSVs manually from the Kaggle
   page and drop them into `data/`.)
3. Clean it into the format the recommender expects:
   ```bash
   python prepare_kaggle_data.py
   ```
   This creates `data/movies.csv`. Both `recommender.py` and `app.py`
   automatically prefer `data/movies.csv` over the sample data if it exists.
4. Run as before:
   ```bash
   streamlit run app.py
   ```

## Using your own dataset

Any CSV with these columns works:

| column   | format                              |
|----------|--------------------------------------|
| title    | movie title                          |
| genres   | pipe-separated, e.g. `Action\|Drama` |
| cast     | pipe-separated, e.g. `Actor One\|Actor Two` |
| director | (optional, not used in similarity)   |
| year     | (optional, not used in similarity)   |

Just point `MovieRecommender(data_path="path/to/your.csv")` at it.

## Possible extensions (nice talking points for an interview / write-up)

- Add **director** and **keywords/overview** into the tag soup for richer similarity.
- Weight cast by billing order (lead actor counts more than actor #5).
- Swap `CountVectorizer` for `TfidfVectorizer` to down-weight very common actors/genres.
- Add a hybrid approach: blend this content-based score with a
  collaborative-filtering score if you also have user ratings data
  (e.g. MovieLens dataset).
- Deploy the Streamlit app on Streamlit Community Cloud for a live demo link.

## Tech used

`pandas` · `scikit-learn` (`CountVectorizer`, `NearestNeighbors`, cosine similarity) · `streamlit`
