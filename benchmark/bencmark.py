import os
import sys

import numpy as np
import torch
import pandas as pd

path_to_models = os.path.abspath(os.path.dirname(__file__))
root_path = path_to_models.replace('/benchmark', '')
sys.path.append(f'{root_path}')  # necessary to add in order to properly see dependencies

from models.training import Loader, MatrixFactorization


def recommend_movies(user_id, model, rated_movies, n=5):
    # Get all movie IDs
    all_movie_ids = np.array(list(train_set.movieid2idx.values()))

    # Filter out movies that the user has already rated
    unrated_movies = np.setdiff1d(all_movie_ids, rated_movies)

    # Prepare input for the selected user and unrated movies
    user_input = torch.tensor([[user_id, movie_id] for movie_id in unrated_movies])

    predictions = model.predict(user_input)
    # Get the indices of the top N movies with the highest predicted ratings
    top_indices = torch.argsort(predictions, descending=True)[:n]

    # Get the movie IDs corresponding to the top indices
    top_movie_ids = unrated_movies[top_indices]
    # Map movie IDs to movie names
    top_movie_names = [movie_names[movie_id] for movie_id in top_movie_ids]

    return top_movie_names


model = torch.load(f"{root_path}/models/my_model.pth")

ratings_df = pd.read_csv(f'{root_path}/data/u.data', sep='\t', header=None,
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])
item_df = pd.read_csv(f'{root_path}/data/u.item', sep='|', header=None, encoding='latin-1',
                      names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                             'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
train_set = Loader(ratings_df)
# Establish a mapping of Movie IDs to Movie Titles
movie_names = item_df.set_index('movie_id')['title'].to_dict()

# Example: Recommend top 5 movies for user with ID 1
user_id_to_recommend = 196
user_ratings = ratings_df[ratings_df['user_id'] == user_id_to_recommend]['movie_id'].values
recommended_movies = recommend_movies(user_id_to_recommend, model, user_ratings, n=5)

print(f"Movies that have been rated:")
for i, movie in enumerate(user_ratings[:10]):
    print(f"{i + 1}. {movie_names[movie]}")

print(f"\nRecommended Movies for User {user_id_to_recommend}:")
for i, movie in enumerate(recommended_movies):
    print(f"{i + 1}. {movie}")
