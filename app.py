"""
app.py
-------
Streamlit UI for the genre + cast movie recommender.

Run with:  streamlit run app.py
"""

import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommender")
st.caption("Content-based recommendations using genre + cast (CountVectorizer + KNN/cosine similarity)")


@st.cache_resource
def load_recommender():
    return MovieRecommender()


rec = load_recommender()

st.write(f"Loaded **{len(rec.df)}** movies from `{rec.data_path}`")

titles = sorted(rec.list_titles())
selected = st.selectbox("Pick a movie you like:", titles)
top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Recommend", type="primary"):
    results, error = rec.recommend(selected, top_n=top_n)
    if error:
        st.error(error)
    else:
        st.subheader(f"Because you liked *{selected}*:")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r['title']}**")
                st.caption(f"Genres: {r['genres']}")
                st.caption(f"Cast: {r['cast']}")
                st.progress(min(max(r["similarity"], 0.0), 1.0), text=f"Similarity: {r['similarity']}")

st.divider()
st.caption(
    "Swap in the full Kaggle dataset by running `prepare_kaggle_data.py` "
    "(see README) — the app will pick up data/movies.csv automatically."
)
