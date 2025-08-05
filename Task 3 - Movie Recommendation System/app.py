import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import streamlit as st
import time

# Set page config
st.set_page_config(
    page_title="🎬 FilmFinder AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme with good contrast
st.markdown("""
    <style>
        :root {
            --primary: #4a6fa5;
            --secondary: #ff7e5f;
            --accent: #6b8c42;
            --background: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333333;
            --text-light: #666666;
            --border: #e0e0e0;
        }
        
        .main {
            background-color: var(--background);
        }
        
        .stSelectbox, .stSlider, .stButton {
            margin-bottom: 1rem;
        }
        
        .recommendation-card {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }
        
        .movie-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        
        .predicted-rating {
            font-size: 0.95rem;
            color: var(--secondary);
            margin-bottom: 0.75rem;
        }
        
        .section-header {
            color: var(--primary);
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }
        
        .sidebar .sidebar-content {
            background-color: var(--card-bg);
        }
        
        .stProgress > div > div > div > div {
            background-color: var(--secondary);
        }
        
        .stMetric {
            background-color: var(--card-bg) !important;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border);
        }
        
        .stMetric label {
            color: var(--text-light) !important;
            font-size: 0.9rem !important;
        }
        
        .stMetric div {
            color: var(--primary) !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }
        
        .stButton button {
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        
        .stButton button:hover {
            background-color: #3a5a8c !important;
        }
    </style>
""", unsafe_allow_html=True)

# Data loading and processing functions
@st.cache_data
def load_data():
    try:
        ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                               names=['user_id', 'movie_id', 'rating', 'timestamp'], 
                               encoding='latin-1')
        movies_df = pd.read_csv('ml-100k/u.item', sep='|', 
                              names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 
                                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                                     'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                                     'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                              encoding='latin-1')
        df = pd.merge(ratings_df, movies_df[['movie_id', 'movie_title']], on='movie_id')
        return df, movies_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_data
def create_user_movie_matrix(df):
    user_movie_matrix = df.pivot_table(index='user_id', columns='movie_title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    return user_movie_matrix, user_movie_matrix_filled

@st.cache_data
def calculate_user_similarity(user_movie_matrix_filled):
    user_similarity = cosine_similarity(user_movie_matrix_filled)
    return pd.DataFrame(user_similarity, 
                      index=user_movie_matrix_filled.index, 
                      columns=user_movie_matrix_filled.index)

@st.cache_resource
def train_svd_model(user_movie_matrix, n_components=20):
    user_movie_matrix_svd = user_movie_matrix.fillna(user_movie_matrix.mean())
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_matrix_svd)
    predicted_ratings = np.dot(latent_matrix, svd.components_)
    return pd.DataFrame(predicted_ratings, 
                      index=user_movie_matrix.index, 
                      columns=user_movie_matrix.columns)

# Recommendation functions
def recommend_movies_user_based(user_id, user_movie_matrix, user_similarity_df, num_recommendations=10, num_similar_users=5):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found in the dataset."

    similar_users = user_similarity_df[user_id].drop(user_id, errors='ignore').sort_values(ascending=False)
    top_similar_users = similar_users.head(num_similar_users).index

    user_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index

    predicted_ratings = {}
    for similar_user in top_similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        unseen_movies = similar_user_ratings[~similar_user_ratings.index.isin(user_rated_movies)]
        unseen_movies = unseen_movies[unseen_movies > 0]

        for movie_title, rating in unseen_movies.items():
            if movie_title not in predicted_ratings:
                predicted_ratings[movie_title] = 0
            predicted_ratings[movie_title] += rating * user_similarity_df.loc[user_id, similar_user]

    normalized_predicted_ratings = {}
    for movie_title, sum_weighted_rating in predicted_ratings.items():
        users_who_rated_this_movie = user_movie_matrix[movie_title][user_movie_matrix[movie_title] > 0].index
        relevant_similar_users = [u for u in top_similar_users if u in users_who_rated_this_movie]

        if relevant_similar_users:
            sum_of_similarities = user_similarity_df.loc[user_id, relevant_similar_users].sum()
            normalized_predicted_ratings[movie_title] = sum_weighted_rating / sum_of_similarities if sum_of_similarities > 0 else 0
        else:
            normalized_predicted_ratings[movie_title] = 0

    recommendations = pd.Series(normalized_predicted_ratings).sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

def recommend_movies_item_based(user_id, user_movie_matrix, num_recommendations=10, num_similar_items=5):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found in the dataset."

    item_user_matrix = user_movie_matrix.T.fillna(0)
    if item_user_matrix.shape[0] < 2:
        return pd.Series(), "Not enough items to compute item similarity."

    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

    user_ratings = user_movie_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    predicted_ratings = {}
    for movie_title, rating in rated_movies.items():
        if movie_title not in item_similarity_df.index:
            continue
        
        similar_items = item_similarity_df[movie_title].drop(movie_title, errors='ignore').sort_values(ascending=False)
        top_similar_items = similar_items.head(num_similar_items).index

        for similar_movie in top_similar_items:
            if similar_movie not in rated_movies.index:
                if similar_movie not in predicted_ratings:
                    predicted_ratings[similar_movie] = 0
                predicted_ratings[similar_movie] += item_similarity_df.loc[movie_title, similar_movie] * rating

    normalized_predicted_ratings = {}
    for movie_title, sum_weighted_rating in predicted_ratings.items():
        contributing_movies = [m for m in rated_movies.index if m in item_similarity_df.columns and movie_title in item_similarity_df.index]
        sum_of_similarities_for_movie = sum(item_similarity_df.loc[m, movie_title] for m in contributing_movies 
                                          if m in item_similarity_df.index and movie_title in item_similarity_df.columns)
        
        normalized_predicted_ratings[movie_title] = (sum_weighted_rating / sum_of_similarities_for_movie 
                                                    if sum_of_similarities_for_movie > 0 else 0)

    recommendations = pd.Series(normalized_predicted_ratings).sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

def recommend_movies_svd(user_id, user_movie_matrix, predicted_ratings_df_svd, num_recommendations=10):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found in the dataset."

    user_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    user_predictions = predicted_ratings_df_svd.loc[user_id]
    unseen_movies_predictions = user_predictions.drop(user_rated_movies, errors='ignore')

    recommendations = unseen_movies_predictions.sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    num_relevant_and_recommended = len(set(recommended_at_k) & set(relevant_items))
    return num_relevant_and_recommended / k if k > 0 else 0

def display_recommendations(recommendations, title, method_color="primary"):
    with st.container():
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        
        if recommendations.empty:
            st.info("No recommendations available for this method.")
            return
        
        for movie, rating in recommendations.items():
            with st.container():
                st.markdown(f"<div class='movie-title'>{movie}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='predicted-rating'>Predicted rating: {rating:.2f}/5</div>", unsafe_allow_html=True)
                st.progress(min(1.0, rating/5.0))
                st.markdown("---")

# Main app function
def main():
    # App header
    st.title("🎬 FilmFinder AI")
    st.markdown("""
        <div style='color: var(--text); margin-bottom: 2rem;'>
        Discover your next favorite movie with our intelligent recommendation system.
        </div>
    """, unsafe_allow_html=True)
    
    # Load data with progress animation
    with st.spinner("Loading movie database..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        df_full, movies_df_full = load_data()
        user_movie_matrix_full, user_movie_matrix_filled_full = create_user_movie_matrix(df_full)
        user_similarity_df_full = calculate_user_similarity(user_movie_matrix_filled_full)
        predicted_ratings_df_svd_full = train_svd_model(user_movie_matrix_full)
        progress_bar.empty()
    
    st.success("✨ Movie database loaded successfully!")
    
    # Sidebar controls
    with st.sidebar:
        st.title("Settings")
        all_user_ids = sorted(df_full['user_id'].unique().tolist())
        target_user_id = st.selectbox("Select User ID", all_user_ids, index=0)
        num_recs = st.slider("Number of Recommendations", 1, 20, 10)
        
        st.markdown("---")
        st.markdown("**Evaluation Settings**")
        k_value_eval = st.slider("Select K for Precision@K", 1, 20, 10)
        
        if st.button("Run Evaluation", use_container_width=True):
            st.session_state.run_evaluation = True
        else:
            st.session_state.run_evaluation = False
    
    # Recommendation section
    st.header(f"Recommendations for User {target_user_id}")
    
    # Get recommendations in parallel columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.spinner("Finding similar users..."):
            user_based_recs, _ = recommend_movies_user_based(
                target_user_id, user_movie_matrix_full, user_similarity_df_full, num_recs
            )
            display_recommendations(user_based_recs, "User-Based Recommendations")
    
    with col2:
        with st.spinner("Analyzing movie similarities..."):
            item_based_recs, _ = recommend_movies_item_based(
                target_user_id, user_movie_matrix_full, num_recs
            )
            display_recommendations(item_based_recs, "Item-Based Recommendations")
    
    with col3:
        with st.spinner("Computing latent factors..."):
            svd_recs, _ = recommend_movies_svd(
                target_user_id, user_movie_matrix_full, predicted_ratings_df_svd_full, num_recs
            )
            display_recommendations(svd_recs, "SVD Recommendations")
    
    # Evaluation section
    if st.session_state.get('run_evaluation', False):
        st.header("Model Evaluation")
        
        with st.spinner("Evaluating models... This may take a few moments"):
            train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)
            train_user_movie_matrix = train_df.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)
            train_user_similarity_df = calculate_user_similarity(train_user_movie_matrix.fillna(0))
            predicted_ratings_df_svd_eval = train_svd_model(train_user_movie_matrix)
            
            # Evaluation metrics
            metrics = {
                "User-Based": {"precision": 0, "count": 0},
                "SVD": {"precision": 0, "count": 0}
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, user_id in enumerate(test_df['user_id'].unique()):
                relevant_movies = test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4)]['movie_title'].tolist()
                if not relevant_movies or user_id not in train_user_movie_matrix.index:
                    continue
                
                # User-based evaluation
                user_recs, _ = recommend_movies_user_based(user_id, train_user_movie_matrix, train_user_similarity_df, k_value_eval)
                metrics["User-Based"]["precision"] += precision_at_k(user_recs.index.tolist(), relevant_movies, k_value_eval)
                metrics["User-Based"]["count"] += 1
                
                # SVD evaluation
                svd_recs, _ = recommend_movies_svd(user_id, train_user_movie_matrix, predicted_ratings_df_svd_eval, k_value_eval)
                metrics["SVD"]["precision"] += precision_at_k(svd_recs.index.tolist(), relevant_movies, k_value_eval)
                metrics["SVD"]["count"] += 1
                
                progress_bar.progress((i + 1) / len(test_df['user_id'].unique()))
                status_text.text(f"Evaluating user {i+1}/{len(test_df['user_id'].unique())}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Display metrics
            if metrics["User-Based"]["count"] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_precision = metrics["User-Based"]["precision"] / metrics["User-Based"]["count"]
                    st.metric(label="User-Based CF", 
                             value=f"{avg_precision:.1%}", 
                             help=f"Precision@{k_value_eval} based on {metrics['User-Based']['count']} users")
                
                with col2:
                    avg_precision = metrics["SVD"]["precision"] / metrics["SVD"]["count"]
                    st.metric(label="SVD-Based CF", 
                             value=f"{avg_precision:.1%}", 
                             help=f"Precision@{k_value_eval} based on {metrics['SVD']['count']} users")
            else:
                st.info("No users could be evaluated with relevant ratings in test set.")

if __name__ == '__main__':
    main()