from django.http import JsonResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from django.shortcuts import render

def home(request):
    return render(request, "index.html")

# Load Dataset
movies = pd.read_csv("dataset/movies.csv")

movies = movies[['Series_Title', 'Overview', 'Genre', 'IMDB_Rating', 'Director', 'Poster_Link']].dropna()

# Clean Text Data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

movies['Overview'] = movies['Overview'].apply(clean_text)
movies['Genre'] = movies['Genre'].apply(lambda x: x.replace(',', ' '))

# Combine Text Features
movies['content'] = movies['Overview'] + ' ' + movies['Genre'] + ' ' + movies['Director']

# Convert Text to Vectors
vectorizer = TfidfVectorizer(stop_words='english')
content_matrix = vectorizer.fit_transform(movies['content'])

# Compute Similarity
cosine_sim = cosine_similarity(content_matrix, content_matrix)

# Recommendation Function
def recommend_movies(title, offset=0, top_n=5):
    # Convert the input title to lowercase
    title = title.lower()
    
    # Create a lowercase version of the Series_Title for comparison
    movies['Series_Title_Lower'] = movies['Series_Title'].str.lower()
    
    if title not in movies['Series_Title_Lower'].values:
        return []
    
    idx = movies[movies['Series_Title_Lower'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude the first movie (itself)
    
    # Apply offset to get the next set of movies
    movie_indices = [i[0] for i in scores[offset:offset + top_n]]
    
    # Include Overview, Director, and Poster_Link in the returned data
    recommendations = movies.iloc[movie_indices][['Series_Title', 'Genre', 'IMDB_Rating', 'Overview', 'Director', 'Poster_Link']].to_dict(orient='records')
    
    # Clean up the DataFrame by dropping the temporary column
    movies.drop(columns=['Series_Title_Lower'], inplace=True)
    
    return recommendations

# Django View for API
def recommend_movie(request):
    title = request.GET.get('title', '')
    offset = int(request.GET.get('offset', 0))  # Get the offset from the request
    recommendations = recommend_movies(title, offset)
    return JsonResponse({'movies': recommendations})
