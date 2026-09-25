**Movie Recommender**

A small project that recommends movies based on genre and cast. You give it a movie you like, and it finds other movies with similar genres and actors.

## Why content-based?

This isn't the "Netflix recommends based on what everyone else watched" kind of system. It doesn't need any user ratings or history — just one movie you already like, and it finds similar ones based on genre and cast.

## How it works

1. Combine each movie's genres and top 3 actors into one string.
2. Turn that string into a vector using `CountVectorizer`.
3. Use `NearestNeighbors` (cosine similarity) to find the closest movies in that vector space.
4. Return the closest ones as recommendations.

That's really it. No deep learning, no training data with labels — just comparing movies based on shared genres/actors.

## Files

- `data/sample_movies.csv` — 40 movies to test with, so you don't need to download anything to try it
- `prepare_kaggle_data.py` — turns the raw Kaggle TMDB dataset into a clean CSV
- `recommender.py` — the actual recommendation logic
- `app.py` — a Streamlit app to try it in the browser

## Running it

```bash
pip install -r requirements.txt
python recommender.py
```

or for the web version:

```bash
streamlit run app.py
```

## Using the full dataset

The sample data only has 40 movies, which is fine for testing but not very impressive. For a bigger dataset:

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle and put the two CSVs in `data/`.
2. Run `python prepare_kaggle_data.py` to clean it up.
3. Run the app again — it automatically uses the bigger dataset if it finds it.

## Using your own data

Any CSV with `title`, `genres`, and `cast` columns works, as long as genres/cast are separated by `|` (e.g. `Action|Drama`).

## Ideas for later

- Use actor billing order as a weight (main actor should matter more than actor #5)
- Add the director as another feature
- Try TF-IDF instead of plain counts
- Combine with a ratings-based approach if I get user data

## Built with

Python, pandas, scikit-learn, Streamlit
