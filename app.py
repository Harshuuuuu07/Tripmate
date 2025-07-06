import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# 🔧 Load and Preprocess
# -----------------------
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Location', 'Type', 'Price_Range'], inplace=True)
    df.fillna('', inplace=True)
    df['combined_features'] = df['Location'].str.lower() + ' ' + df['Type'].str.lower() + ' ' + df['Price_Range'].str.lower()
    df['Name_lower'] = df['Name'].str.lower()
    return df

@st.cache_resource
def compute_similarity(df):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(feature_vectors)

# -----------------------
# 🤖 Recommender Logic
# -----------------------
def recommend_places(df, similarity_matrix, place_name, top_n=5):
    place_name = place_name.strip().lower()

    if place_name not in df['Name_lower'].values:
        return None

    idx = df[df['Name_lower'] == place_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return df.iloc[[i[0] for i in sorted_scores]]


# -----------------------
# 🚀 Streamlit App
# -----------------------
st.set_page_config(page_title="TripMate", page_icon="🌍")
st.title("🌍 TripMate: Place Recommender")
st.markdown("Find similar places based on location, type, and budget!")

# Load data
df = load_data("data.csv")
similarity_matrix = compute_similarity(df)

# Input from user
place_input = st.selectbox("Select a place from the list:", sorted(df['Name'].unique()))
num_recommendations = st.slider("Number of recommendations", 1, 10, 5)

if st.button("🔍 Recommend"):
    recommendations = recommend_places(df, similarity_matrix, place_input, top_n=num_recommendations)

    if recommendations is not None:
        st.subheader(f"Places similar to **{place_input}**:")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{row['Name']}** - {row['Location']} - {row['Type']} - {row['Price_Range']}")
    else:
        st.error("Place not found. Please try another.")

