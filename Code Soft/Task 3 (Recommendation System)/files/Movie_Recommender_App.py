# ============================================================
#  🎬 Movie Recommendation System — Streamlit Web App
#  Fine-tuned Content-Based Filtering using Cosine Similarity
# ============================================================
#
#  HOW TO RUN:
#  1. Make sure movies.pkl and similarity.pkl are in the same folder
#  2. Install dependencies:
#       pip install streamlit pandas numpy scikit-learn
#  3. Run in terminal:
#       streamlit run Movie_Recommender_App.py
#  4. Browser opens at: http://localhost:8501
#
# ============================================================


# ============================================================
#  SECTION 1 — IMPORTS
# ============================================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import MinMaxScaler


# ============================================================
#  SECTION 2 — PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================

st.set_page_config(
    page_title = 'Movie Recommender',
    page_icon  = '🎬',
    layout     = 'wide',
    initial_sidebar_state = 'expanded'
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Page background */
    .main { background-color: #0e1117; }

    /* Movie card */
    .movie-card {
        background: linear-gradient(135deg, #1e2130, #252b3b);
        border: 1px solid #2e3555;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 8px 0;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        border-color: #e50914;
    }

    /* Movie title inside card */
    .movie-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    /* Movie meta info */
    .movie-meta {
        font-size: 13px;
        color: #9ca3af;
    }

    /* Genre badge */
    .genre-badge {
        display: inline-block;
        background: #e50914;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        margin: 2px 2px;
        font-weight: 600;
    }

    /* Rank number */
    .rank-num {
        font-size: 28px;
        font-weight: 900;
        color: #e50914;
        opacity: 0.9;
    }

    /* Star rating */
    .star-rating {
        color: #f59e0b;
        font-size: 14px;
    }

    /* Section header */
    .section-header {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        border-left: 4px solid #e50914;
        padding-left: 12px;
        margin: 20px 0 14px 0;
    }

    /* Input movie info box */
    .input-movie-box {
        background: linear-gradient(135deg, #1a1f2e, #e50914);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 20px;
        color: white;
    }

    /* Stat box */
    .stat-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        border: 1px solid #2e3555;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #141722;
    }

    /* Streamlit buttons */
    .stButton > button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 15px;
        padding: 10px 28px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #b00710;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1e2130;
        border: 1px solid #2e3555;
        border-radius: 8px;
        color: white;
    }

    /* Slider */
    .stSlider > div { color: #9ca3af; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SECTION 3 — LOAD MODEL FILES
#  FIX: Cached loading so files are only read once per session
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the pre-computed movies dataframe and similarity matrix.
    Uses st.cache_resource so they are only loaded once.
    Returns (movies_df, similarity_matrix) or (None, None) on error.
    """
    try:
        movies     = pickle.load(open('movies.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except FileNotFoundError:
        return None, None


movies, similarity = load_model()


# ============================================================
#  SECTION 4 — RECOMMENDATION ENGINE
#  (Self-contained so app doesn't depend on the notebook)
# ============================================================

def find_movie_index(title):
    """Find movie index by exact or partial title match."""
    if movies is None:
        return -1
    exact = movies[movies['title'] == title]
    if not exact.empty:
        return exact.index[0]
    partial = movies[movies['title'].str.lower().str.contains(
        title.lower(), na=False, regex=False
    )]
    return partial.index[0] if not partial.empty else -1


def recommend(movie_title, top_n=5, popularity_weight=0.15, genre_filter=None):
    """
    Core recommendation function.
    Args:
        movie_title      : input movie title string
        top_n            : number of results to return
        popularity_weight: 0.0 = pure similarity, 0.15 = fine-tuned default
        genre_filter     : optional genre string to filter results
    Returns:
        list of result dicts with title, genres, year, avg_rating, score
    """
    if movies is None or similarity is None:
        return []

    idx = find_movie_index(movie_title)
    if idx == -1:
        return []

    # Normalize popularity score
    scaler = MinMaxScaler()
    pop_scores = scaler.fit_transform(
        movies[['rating_count']].fillna(0)
    ).flatten()

    # Compute boosted scores
    sim_scores    = similarity[idx].copy()
    boosted       = (1 - popularity_weight) * sim_scores + popularity_weight * pop_scores

    # Rank and skip the input movie itself
    ranked = sorted(enumerate(boosted), key=lambda x: x[1], reverse=True)[1:]

    results = []
    for i, score in ranked:
        if len(results) >= top_n * 3:   # Collect extra for genre filtering
            break
        row = movies.iloc[i]
        results.append({
            'title'        : row['title'],
            'clean_title'  : row.get('clean_title', row['title']),
            'genres'       : row['genres'],
            'year'         : int(row.get('year', 0)),
            'avg_rating'   : float(row.get('avg_rating', 0.0)),
            'rating_count' : int(row.get('rating_count', 0)),
            'score'        : round(float(score), 4)
        })

    # Apply genre filter
    if genre_filter and genre_filter != 'All':
        results = [r for r in results if genre_filter.lower() in r['genres'].lower()]

    return results[:top_n]


def get_movie_info(title):
    """Return metadata dict for a given movie title."""
    idx = find_movie_index(title)
    if idx == -1:
        return None
    row = movies.iloc[idx]
    return {
        'title'        : row['title'],
        'genres'       : row['genres'],
        'year'         : int(row.get('year', 0)),
        'avg_rating'   : float(row.get('avg_rating', 0.0)),
        'rating_count' : int(row.get('rating_count', 0))
    }


# ============================================================
#  SECTION 5 — HELPER RENDER FUNCTIONS
# ============================================================

def render_genre_badges(genres_str):
    """Convert 'Action|Comedy' into styled HTML badge spans."""
    genres = str(genres_str).split('|')
    badges = ''.join(
        "<span class='genre-badge'>{}</span>".format(g.strip())
        for g in genres if g.strip() and g.strip() != '(no genres listed)'
    )
    return badges


def render_stars(rating, max_rating=5.0):
    """Convert numeric rating to star string."""
    filled  = int(round(rating / max_rating * 5))
    empty   = 5 - filled
    return '★' * filled + '☆' * empty


def render_movie_card(rank, movie_dict):
    """Render a single recommendation card as HTML."""
    title   = movie_dict['title']
    year    = movie_dict['year'] if movie_dict['year'] > 0 else 'N/A'
    rating  = movie_dict['avg_rating']
    count   = movie_dict['rating_count']
    genres  = render_genre_badges(movie_dict['genres'])
    stars   = render_stars(rating)
    score   = movie_dict['score']

    st.markdown("""
    <div class='movie-card'>
        <div style='display:flex; align-items:flex-start; gap:16px;'>
            <div class='rank-num'>{rank}</div>
            <div style='flex:1'>
                <div class='movie-title'>{title}</div>
                <div class='movie-meta'>
                    📅 {year} &nbsp;|&nbsp;
                    <span class='star-rating'>{stars}</span>
                    &nbsp;{rating} ({count} ratings) &nbsp;|&nbsp;
                    Match: {score}
                </div>
                <div style='margin-top:8px'>{genres}</div>
            </div>
        </div>
    </div>
    """.format(
        rank=rank, title=title, year=year,
        stars=stars, rating=rating, count=count,
        score=score, genres=genres
    ), unsafe_allow_html=True)


# ============================================================
#  SECTION 6 — SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("*Content-Based Filtering — Cosine Similarity*")
    st.divider()

    if movies is not None:
        st.markdown("### ⚙️ Settings")

        top_n = st.slider(
            "Number of Recommendations",
            min_value = 3,
            max_value = 15,
            value     = 5,
            step      = 1,
            help      = "How many movies to recommend"
        )

        pop_weight = st.slider(
            "Popularity Boost",
            min_value = 0.0,
            max_value = 0.5,
            value     = 0.15,
            step      = 0.05,
            help      = "0 = pure genre match, 0.5 = half popularity"
        )

        # Genre filter
        all_genres = sorted(set(
            g.strip()
            for row in movies['genres'].dropna()
            for g in str(row).split('|')
            if g.strip() and g.strip() != '(no genres listed)'
        ))
        genre_filter = st.selectbox(
            "Filter by Genre",
            options = ['All'] + all_genres,
            index   = 0,
            help    = "Restrict recommendations to this genre"
        )

        st.divider()
        st.markdown("### 📊 Dataset Stats")
        c1, c2 = st.columns(2)
        c1.metric("Movies",  f"{len(movies):,}")
        c2.metric("Genres",  len(all_genres))

        avg_r = movies['avg_rating'].mean()
        c1.metric("Avg Rating", f"{avg_r:.2f} ⭐")
        c2.metric("Intents",   "Content-Based")

        st.divider()
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. You pick a movie
        2. System converts genres → vectors
        3. Cosine similarity finds closest matches
        4. Top results shown ranked by similarity
        """)
        st.caption("Built with Python · scikit-learn · Streamlit")

    else:
        st.error("Model files not found!")
        st.markdown("""
        **Missing files:**
        - `movies.pkl`
        - `similarity.pkl`

        **Fix:** Run `Movie_Recommendation_Complete.ipynb` first (Step 23 saves these files).
        """)
        top_n        = 5
        pop_weight   = 0.15
        genre_filter = 'All'


# ============================================================
#  SECTION 7 — MAIN PAGE HEADER
# ============================================================

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <div style='font-size:42px;'>🎬</div>
    <div style='font-size:32px; font-weight:900; color:#ffffff;'>Movie Recommendation System</div>
    <div style='font-size:15px; color:#9ca3af; margin-top:6px;'>
        Content-Based Filtering · TF-IDF Vectorization · Cosine Similarity
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
#  SECTION 8 — MOVIE SELECTION & INPUT
# ============================================================

if movies is None:
    st.error("""
    **Model files not found.**

    Please run all cells in `Movie_Recommendation_Complete.ipynb` first.
    Step 23 will generate `movies.pkl` and `similarity.pkl` in the same folder.
    Then restart this app.
    """)
    st.stop()


col_select, col_btn = st.columns([4, 1])

with col_select:
    movie_list = sorted(movies['title'].dropna().unique().tolist())
    selected_movie = st.selectbox(
        "Select a Movie",
        options     = movie_list,
        index       = 0,
        placeholder = "Search for a movie...",
        label_visibility = 'collapsed'
    )

with col_btn:
    get_recs = st.button("🎯 Recommend", use_container_width=True)


# ============================================================
#  SECTION 9 — DISPLAY INPUT MOVIE INFO
# ============================================================

if selected_movie:
    info = get_movie_info(selected_movie)
    if info:
        genres_html = render_genre_badges(info['genres'])
        year_str    = str(info['year']) if info['year'] > 0 else 'N/A'
        stars_str   = render_stars(info['avg_rating'])

        st.markdown("""
        <div class='input-movie-box'>
            <div style='font-size:12px; opacity:0.8; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:4px;'>Selected Movie</div>
            <div style='font-size:24px; font-weight:800;'>{title}</div>
            <div style='font-size:14px; margin:6px 0;'>
                📅 {year} &nbsp;|&nbsp;
                <span style='color:#f59e0b'>{stars}</span>
                &nbsp;{rating} ⭐ ({count} ratings)
            </div>
            <div style='margin-top:8px'>{genres}</div>
        </div>
        """.format(
            title  = info['title'],
            year   = year_str,
            stars  = stars_str,
            rating = info['avg_rating'],
            count  = info['rating_count'],
            genres = genres_html
        ), unsafe_allow_html=True)


# ============================================================
#  SECTION 10 — RECOMMENDATIONS OUTPUT
# ============================================================

if get_recs and selected_movie:
    results = recommend(
        movie_title      = selected_movie,
        top_n            = top_n,
        popularity_weight = pop_weight,
        genre_filter     = genre_filter if genre_filter != 'All' else None
    )

    if not results:
        st.warning("""
        No recommendations found for the current filters.
        Try changing the genre filter to 'All' or pick a different movie.
        """)
    else:
        st.markdown(
            "<div class='section-header'>🍿 Top {} Recommendations</div>".format(len(results)),
            unsafe_allow_html=True
        )

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Recommendations",  len(results))
        mc2.metric("Avg Rating",       f"{np.mean([r['avg_rating'] for r in results]):.2f} ⭐")
        mc3.metric("Popularity Boost", f"{int(pop_weight * 100)}%")
        mc4.metric("Genre Filter",     genre_filter)

        st.markdown("")

        # Render movie cards
        for rank, movie_dict in enumerate(results, 1):
            render_movie_card(rank, movie_dict)

        st.divider()

        # Download results as CSV
        df_results = pd.DataFrame(results)[['title', 'genres', 'year', 'avg_rating', 'score']]
        df_results.columns = ['Title', 'Genres', 'Year', 'Avg Rating', 'Similarity Score']
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            label     = "📥 Download Recommendations as CSV",
            data      = csv_data,
            file_name = "recommendations_{}.csv".format(
                selected_movie.replace(' ', '_').replace('/', '_')[:40]
            ),
            mime      = "text/csv"
        )


# ============================================================
#  SECTION 11 — GENRE BROWSER (Expandable)
# ============================================================

with st.expander("🗂️ Browse Movies by Genre"):
    if movies is not None:
        browse_genre = st.selectbox(
            "Select Genre to Browse",
            options = all_genres,
            key     = "browse_genre"
        )

        filtered_movies = movies[
            movies['genres'].str.contains(browse_genre, na=False, case=False)
        ].sort_values('avg_rating', ascending=False).head(20)

        st.markdown(f"**Top 20 {browse_genre} movies by rating:**")

        browse_df = filtered_movies[['title', 'year', 'avg_rating', 'rating_count', 'genres']].copy()
        browse_df.columns = ['Title', 'Year', 'Avg Rating', 'Ratings Count', 'Genres']
        browse_df['Year'] = browse_df['Year'].astype(str).replace('0', 'N/A')

        st.dataframe(browse_df, use_container_width=True, hide_index=True)


# ============================================================
#  SECTION 12 — SIMILARITY EXPLORER (Expandable)
# ============================================================

with st.expander("🔍 Explore Similarity Scores"):
    if movies is not None and selected_movie:
        idx = find_movie_index(selected_movie)
        if idx != -1:
            sim_scores = similarity[idx]
            top_sim    = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)[1:11]

            sim_data = []
            for i, score in top_sim:
                row = movies.iloc[i]
                sim_data.append({
                    'Movie'      : row['title'],
                    'Year'       : int(row.get('year', 0)),
                    'Similarity' : round(float(score), 4),
                    'Genres'     : row['genres']
                })

            sim_df = pd.DataFrame(sim_data)
            st.markdown(f"**Raw similarity scores for:** {selected_movie}")
            st.dataframe(sim_df, use_container_width=True, hide_index=True)

            # Bar chart
            st.bar_chart(
                pd.DataFrame({
                    'Movie'      : [d['Movie'][:30] for d in sim_data],
                    'Similarity' : [d['Similarity'] for d in sim_data]
                }).set_index('Movie')
            )


# ============================================================
#  SECTION 13 — QUICK PICKS (Pre-set popular movies)
# ============================================================

st.divider()
st.markdown("<div class='section-header'>⚡ Quick Picks</div>", unsafe_allow_html=True)
st.markdown("Click any movie below to instantly load recommendations.")

quick_movies = [
    'Toy Story (1995)',
    'The Silence of the Lambs (1991)',
    'Pulp Fiction (1994)',
    'Forrest Gump (1994)',
    'The Matrix (1999)',
    'Schindler\'s List (1993)'
]

# Filter to only movies that exist in the dataset
quick_movies = [m for m in quick_movies if find_movie_index(m) != -1]

if quick_movies:
    qcols = st.columns(len(quick_movies))
    for i, qm in enumerate(quick_movies):
        if qcols[i].button(
            qm.split(' (')[0],
            key             = f'quick_{i}',
            use_container_width = True
        ):
            recs = recommend(qm, top_n=top_n, popularity_weight=pop_weight)
            if recs:
                st.markdown(
                    "<div class='section-header'>🎬 Quick Recommendations for: {}</div>".format(qm),
                    unsafe_allow_html=True
                )
                for rank, r in enumerate(recs, 1):
                    render_movie_card(rank, r)
