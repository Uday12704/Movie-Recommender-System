import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load Dataset
movies = pd.read_csv("dataset/movies.csv")

# Select Relevant Columns and Drop Missing Values
movies = movies[['Series_Title', 'Overview', 'Genre']].dropna()

# Clean Text Data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

movies['Overview'] = movies['Overview'].apply(clean_text)
movies['Genre'] = movies['Genre'].apply(lambda x: x.replace(',', ' '))

# Combine Text Features
movies['content'] = movies['Overview'] + ' ' + movies['Genre']

# Convert Text to Vectors
vectorizer = TfidfVectorizer(stop_words='english')
content_matrix = vectorizer.fit_transform(movies['content'])

# Compute Similarity
cosine_sim = cosine_similarity(content_matrix, content_matrix)

# Recommendation Function
def recommend_movies(title, top_n=5):
    if title not in movies['Series_Title'].values:
        return "Movie not found!"
    idx = movies[movies['Series_Title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return movies.iloc[movie_indices][['Series_Title', 'Genre']]

# Example Usage
print(recommend_movies("The Godfather"))
