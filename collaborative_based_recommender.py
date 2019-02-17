import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# Reading all dataset files
ratings = pd.read_csv('ratings_given.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv('all_users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movies = pd.read_csv('movies_dataset.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

# counting total no. of users and movies
num_users = ratings.user_id.unique().shape[0]
num_movies = ratings.movie_id.unique().shape[0]

# formatting my ratings matrix to be one row per user and one column per movie using pivot
Ratings = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Now, normalizing the data by taking the difference of ratings and total mean of ratings of user
R = Ratings.as_matrix()  # formatting
user_ratings_mean = np.mean(R, axis=1)  # taking mean
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)  # taking the difference and getting normalized data

# Applying svds algo. or matrix factorization which will give us three individual matrix
U, sigma, Vt = svds(Ratings_demeaned, k=50)
sigma = np.diag(sigma)  # converting into diagonal matrix form for getting predictions

# calculating dot products and denormalizing the data by adding the mean again to get all predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns=Ratings.columns)
# preds.head()


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):

    # Getting and sorting the user's predictions
    user_row_number = userID - 1  # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == userID]
    user_full = (user_data.merge(movies, how='left', left_on='movie_id', right_on='movie_id').
                 sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending {0} predictions for movies that the user has not yet rated/watched.'.format(num_recommendations))

    # Recommending the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='movie_id',
                                 right_on='movie_id').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

    return user_full, recommendations


already_rated, predictions = recommend_movies(preds, 1310, movies, ratings, 10)
# print(already_rated.head(10))
print(predictions)
