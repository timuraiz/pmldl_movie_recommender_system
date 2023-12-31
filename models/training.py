# -*- coding: utf-8 -*-
"""1.0-data-exporation-training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MNyCjTfCXOOVR8nVWsF3kHcRyBbSM0tx

## Data preparation & Data exploration
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings


class MatrixFactorization(torch.nn.Module):
    '''
      The purpose of this class is to represent a Matrix Factorization model.
      It is designed to capture the latent factors (embeddings) of users and items
      in a way that allows the model to predict user-item interactions.

      The user_factors and item_factors embeddings are created as lookup tables.
      These embeddings represent users and items in a lower-dimensional space (n_factors),
      allowing the model to learn patterns and relationships in the data.

      The forward method is where the actual matrix multiplication happens.
      It takes user and item indices and computes the dot product of their embeddings,
      resulting in a prediction for the user-item interaction.

      The predict method is a convenient way to use the model for making predictions.
      It calls the forward method with user and item indices to generate predictions.
    '''

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()

        # User embeddings: Representing users in a lower-dimensional space
        self.user_factors = torch.nn.Embedding(n_users, n_factors)

        # Item embeddings: Representing items (e.g., movies) in a lower-dimensional space
        self.item_factors = torch.nn.Embedding(n_items, n_factors)

        # Initialize embeddings with small random values to start training
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        # Extract user and item indices from the input data
        users, items = data[:, 0], data[:, 1]

        # Matrix multiplication to estimate user-item interactions
        # This operation generates predictions for user-item ratings
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

    def predict(self, data):
        # Convenience method for making predictions
        return self.forward(data)


class Loader(Dataset):
    '''

      The purpose of this class is to create a PyTorch DataLoader for training a model.
      It inherits from the Dataset class, allowing it to be used with PyTorch's DataLoader.

      The initialization method sets up the necessary data structures for mapping
      between original user and movie IDs and continuous indices.
      It also transforms the data into PyTorch tensors.

      The __getitem__ method returns a tuple of features (x) and target rating (y)
      for a given index, making the dataset compatible with PyTorch's DataLoader.

      The __len__ method returns the total number of ratings in the dataset,
      which is essential for iterating through the DataLoader during training.

    '''

    def __init__(self, ratings_df):
        # Store the ratings DataFrame
        self.ratings = ratings_df

        # Extract all unique user IDs and movie IDs
        users = ratings_df.user_id.unique()
        movies = ratings_df.movie_id.unique()

        # Producing new continuous IDs for users and movies

        # Map unique values to indices
        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}

        # Obtain continuous ID for users and movies
        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        # Replace original movie and user IDs with continuous indices
        self.ratings.movie_id = ratings_df.movie_id.apply(lambda x: self.movieid2idx[x])
        self.ratings.user_id = ratings_df.user_id.apply(lambda x: self.userid2idx[x])

        # Extract features (x) and target ratings (y) for training
        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values

        # Transform the data to PyTorch tensors (ready for torch models)
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)

    def __getitem__(self, index):
        # Return a tuple of features (x) and target rating (y) for a given index
        return self.x[index], self.y[index]

    def __len__(self):
        # Return the total number of ratings in the dataset
        return len(self.ratings)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    path_to_models = os.path.abspath(os.path.dirname(__file__))
    root_path = path_to_models.replace('/models', '')

    ratings_df = pd.read_csv(f'{root_path}/data/u.data', sep='\t', header=None,
                             names=['user_id', 'movie_id', 'rating', 'timestamp'])

    item_df = pd.read_csv(f'{root_path}/data/u.item', sep='|', header=None, encoding='latin-1',
                          names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                                 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

    item_df.head()

    # Establish a mapping of Movie IDs to Movie Titles
    movie_names = item_df.set_index('movie_id')['title'].to_dict()

    # Determine the count of distinct users and movies in the dataset
    unique_user_count = len(ratings_df['user_id'].unique())
    unique_movie_count = len(item_df['movie_id'].unique())

    # Provide insights into the dataset
    print("Number of distinct users:", unique_user_count)
    print("Number of distinct movies:", unique_movie_count)
    print("The complete rating matrix consists of:", unique_user_count * unique_movie_count, 'elements.')
    print('----------')
    print("Total number of ratings recorded:", len(ratings_df))
    # Compute the percentage of the matrix that has been populated
    matrix_fill_percentage = (len(ratings_df) / (unique_user_count * unique_movie_count)) * 100
    print(f"As a result, approximately {matrix_fill_percentage:.2f}% of the matrix is populated.")

    """## Train & Test & Evaluate"""

    train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # The purpose of this section is to set up and configure the training process for the MatrixFactorization model.

    # The model is instantiated, and information about its parameters is displayed.
    # GPU is enabled if available to accelerate training.

    # The Mean Squared Error (MSE) loss function and the ADAM optimizer are defined.

    # DataLoader instances for both the training and testing sets are created.
    # These DataLoader instances will be used to efficiently load batches of data during training and evaluation.

    num_epochs = 128
    cuda = torch.cuda.is_available()

    print("Is running on GPU:", cuda)

    # Create an instance of the MatrixFactorization model
    model = MatrixFactorization(unique_user_count, unique_movie_count, n_factors=8)
    print(model)

    # Display the model's trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # Enable GPU if available
    # if cuda:
    #     model = model.cuda()

    # Define the Mean Squared Error (MSE) loss function
    loss_fn = torch.nn.MSELoss()

    # Set up the ADAM optimizer with a specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create DataLoader instances for training and testing sets
    # Note: The DataLoader is essential for efficiently iterating through the data during training
    train_set, test_set = Loader(train_data), Loader(test_data)
    train_loader = DataLoader(train_set, 128, shuffle=True)
    test_loader = DataLoader(test_set, 128, shuffle=True)

    for it in (range(num_epochs)):
        losses = []
        for x, y in train_loader:
            # if cuda:
            #   x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Step: {}".format(it), "Loss:", sum(losses) / len(losses))

    model.eval()  # Set the model to evaluation mode

    test_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            # if cuda:
            #     x, y = x.cuda(), y.cuda()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
            test_losses.append(loss.item())

    average_test_loss = sum(test_losses) / len(test_losses)
    print("Average Test Loss:", average_test_loss)

    c = 0
    uw = 0
    iw = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            if c == 0:
                uw = param.data
                c += 1
            else:
                iw = param.data

    trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()

    # Fit the clusters based on the movie weights
    kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)

    len(kmeans.labels_)

    # The purpose of this section is to analyze movie clusters obtained from KMeans clustering.

    # It iterates through each cluster and prints the top 10 movies in each cluster based on rating count.

    # Analyzing movie clusters and their genres based on KMeans clustering results
    # Iterate through each cluster (assuming there are 10 clusters)
    for cluster in range(10):
        print("Cluster #{}".format(cluster))
        movs = []

        # Iterate through movie indices in the current cluster
        for movidx in np.where(kmeans.labels_ == cluster)[0]:
            try:
                # Convert the index back to the original movie ID
                movid = train_set.idx2movieid[movidx]

                # Count the number of ratings for the current movie
                rat_count = ratings_df.loc[ratings_df['movie_id'] == movid].count()[0]

                # Append movie name and rating count to the list
                movs.append((movie_names[movid], rat_count))
            except:
                pass

        # Display the top 10 movies in the cluster based on rating count
        for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
            print("\t", mov[0])

    model_path = f'{root_path}/benchmark/my_model.pth'

    # Save the entire model including architecture and parameters
    torch.save(model, model_path)


if __name__ == '__main__':
    main()
