import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def load_ratings(file_path):
    """
    Loads and preprocesses the ratings data.
    
    Args:
        file_path (str): Path to the ratings CSV file.
    
    Returns:
        DataFrame: Preprocessed ratings data.
    """
    # Load the ratings data
    ratings = pd.read_csv(file_path)
    print("\nRatings Data:")
    print(ratings.head())
    return ratings

def build_recommendation_system(ratings_file):
    """
    Builds a personalized movie recommendation system using collaborative filtering.
    
    Args:
        ratings_file (str): Path to the ratings CSV file.
    
    Returns:
        None
    """
    # Load ratings data
    ratings = load_ratings(ratings_file)
    
    # Prepare the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Use SVD for collaborative filtering
    model = SVD()
    model.fit(trainset)
    
    # Evaluate the model
    predictions = model.test(testset)
    print("\nModel Performance:")
    accuracy.rmse(predictions)
    
    # Recommend movies for a specific user
    user_id = 1  # Example user ID
    print(f"\nRecommendations for User {user_id}:")
    movie_ids = ratings['movieId'].unique()
    user_ratings = ratings[ratings['userId'] == user_id]['movieId'].values
    recommendations = []
    
    for movie_id in movie_ids:
        if movie_id not in user_ratings:
            est_rating = model.predict(user_id, movie_id).est
            recommendations.append((movie_id, est_rating))
    
    # Sort recommendations by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    for movie_id, rating in recommendations:
        print(f"Movie ID: {movie_id}, Predicted Rating: {rating:.2f}")

# Example usage
if __name__ == "__main__":
    ratings_file = "ml-32m/ml-32m/ratings.csv"
    try:
        build_recommendation_system(ratings_file)
    except FileNotFoundError:
        print(f"Error: The file '{ratings_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")