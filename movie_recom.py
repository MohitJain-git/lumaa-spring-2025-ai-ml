import pandas as pd
import ast
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

# Load and preprocess movie data from a CSV file and return a DataFrame
def load_movie_data(csv_file, preprocessed_file="preprocessed_df.csv"):
    if os.path.exists(preprocessed_file):
        return pd.read_csv(preprocessed_file)

    columns_to_keep = [
        "keywords", "overview", "tagline", "genres", "title",
        "vote_average", "original_language"
    ]

    try:
        df = pd.read_csv(csv_file, usecols=columns_to_keep)
        df["genres"] = df["genres"].apply(lambda x: ", ".join([genre["name"] for genre in ast.literal_eval(x)]) if pd.notna(x) else "")
        df["keywords"] = df["keywords"].apply(lambda x: ", ".join([keyword["name"] for keyword in ast.literal_eval(x)]) if pd.notna(x) else "")

        # Sample and preprocess data
        df = sample_data(df)
        df = preprocess_text_data(df)

        # Save preprocessed data
        df.to_csv(preprocessed_file, index=False)
        return df

    except FileNotFoundError:
        st.error("Error: File not found.")
        return None

    except ValueError:
        st.error("Error: Some specified columns are missing from the CSV file.")
        return None

# Sample English-language movies from the dataset as there are not a lot of non-English movies, so we will focus on English movies
def sample_data(df, sample_size=500):
    df_english = df[df["original_language"] == "en"]
    return df_english.sample(n=min(sample_size, len(df_english)), random_state=42)

# Preprocess the text data to create a single column with all the text combined
def preprocess_text_data(df):
    df["combined_text"] = df[["title", "overview", "tagline", "genres", "keywords"]].fillna('').agg(' '.join, axis=1)
    return df

# Fit the TF-IDF model to the movie descriptions
def fit_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    pickle.dump((vectorizer, tfidf_matrix), open("tfidf_model.pkl", "wb"))
    return vectorizer, tfidf_matrix

# Fit the SentenceTransformer model to the movie descriptions
def fit_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_list = df["combined_text"].tolist()
    embeddings = model.encode(text_list, convert_to_tensor=True)
    pickle.dump((model, embeddings), open("embeddings_model.pkl", "wb"))
    return model, embeddings

# Load the trained models from disk
def load_models():
    vectorizer, tfidf_matrix = pickle.load(open("tfidf_model.pkl", "rb"))
    model, embeddings = pickle.load(open("embeddings_model.pkl", "rb"))
    return vectorizer, tfidf_matrix, model, embeddings

# Code to check the similarity between the user input and the movie descriptions to recommend the top N most similar movies
# The recommendation is based on a weighted score of the cosine similarity between the user input and movie descriptions, and the movie ratings
# The user input is enhanced by adding the genres of the movies that match the user input
def recommend_movies(df, user_input, method, vectorizer=None, tfidf_matrix=None, model=None, embeddings=None, top_n=5):
    all_genres = " ".join(df["genres"].dropna().str.lower())  # Fix: Properly join all genres into a single string
    user_genres = ' '.join([word for word in user_input.split() if word.lower() in all_genres])
    enhanced_input = user_input + " " + user_genres * 3  
    
    if method == "TF-IDF":
        user_tfidf = vectorizer.transform([enhanced_input])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    elif method == "Word Embeddings":
        user_embedding = model.encode([enhanced_input], convert_to_tensor=True)
        similarities = cosine_similarity(user_embedding.cpu().numpy(), embeddings.cpu().numpy()).flatten()
    
    df["normalized_vote"] = df["vote_average"] / df["vote_average"].max()
    weighted_score = (
        similarities * 0.75 + df["normalized_vote"].values * 0.25
    )
    
    top_indices = weighted_score.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][["title", "vote_average", "overview", "genres"]]
    
    return recommendations


def main():
    st.title("Movie Recommendation System")
    
    csv_file_path = "movies.csv"  
    df_sampled = load_movie_data(csv_file_path)

    if not os.path.exists("tfidf_model.pkl") or not os.path.exists("embeddings_model.pkl"):
        st.write("Training models, please wait...")
        fit_tfidf(df_sampled)
        fit_embeddings(df_sampled)

    vectorizer, tfidf_matrix, model, embeddings = load_models()
    
    method = st.radio("Choose Recommendation Method:", ["TF-IDF", "Word Embeddings"])
    user_query = st.text_input("Describe the type of movie you want to watch:")
    
    if st.button("Get Recommendations") and user_query:
        recommendations = recommend_movies(df_sampled, user_query, method, vectorizer, tfidf_matrix, model, embeddings)
        st.write("### Top Recommended Movies:")
        st.write(recommendations)

if __name__ == "__main__":
    main()

