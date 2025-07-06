import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 🔁 ₹ Conversion from $
# ---------------------------
def convert_price_symbols(price):
    symbol_map = {
        '$': '₹',
        '$$': '₹₹',
        '$$$': '₹₹₹',
        '$$$$': '₹₹₹₹'
    }
    return symbol_map.get(price.strip(), price.strip())

# ---------------------------
# 🧹 Load and preprocess data
# ---------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Clean and fill missing values
    df.dropna(subset=['Name', 'Location', 'Type', 'Price_Range'], inplace=True)
    df.fillna('', inplace=True)

    # Convert Price Range
    df['Price_Range'] = df['Price_Range'].apply(convert_price_symbols)

    # Combined features for content filtering
    df['combined_features'] = (
        df['Location'].str.lower() + ' ' +
        df['Type'].str.lower() + ' ' +
        df['Price_Range'].str.lower()
    )

    df['Name_lower'] = df['Name'].str.lower()
    return df

# ---------------------------
# 🤖 Compute similarity
# ---------------------------
@st.cache_resource
def compute_similarity(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

# ---------------------------
# 🔍 Recommendation Function
# ---------------------------
def recommend_places(df, similarity_matrix, selected_place, top_n=5):
    selected_place = selected_place.lower().strip()

    if selected_place not in df['Name_lower'].values:
        return None

    index = df[df['Name_lower'] == selected_place].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    recommended_df = df.iloc[[i[0] for i in scores]]
    return recommended_df

# ---------------------------
# 🚀 Streamlit App UI
# ---------------------------
st.set_page_config(page_title="TripMate", page_icon="🌍")
st.title("🌍 TripMate: Smart Place Recommender")
st.markdown("Discover similar places based on location, type, and budget.")

# Load data
df = load_data("data.csv")
similarity_matrix = compute_similarity(df)

# UI inputs
place_input = st.selectbox("🔎 Choose a place to get recommendations:", sorted(df['Name'].unique()))
num_recommendations = st.slider("📌 Number of suggestions", 1, 10, 5)

# Show recommendations
if st.button("🎯 Recommend"):
    recommendations = recommend_places(df, similarity_matrix, place_input, top_n=num_recommendations)

    if recommendations is not None and not recommendations.empty:
        st.subheader(f"✅ Similar places to: **{place_input}**")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{row['Name']}**  \n📍 *{row['Location']}*  \n🍽️ *{row['Type']}*  \n💰 *{row['Price_Range']}*")
            st.markdown("---")
    else:
        st.error("❌ Place not found or no similar places found.")
