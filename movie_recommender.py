from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def extract_bert_embeddings(texts, batch_size=32):
    """
    Extracts BERT embeddings for a list of texts in batches.
    
    Args:
        texts (list): List of text strings.
        batch_size (int): Number of texts to process in each batch.
    
    Returns:
        torch.Tensor: BERT embeddings.
    """
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory
    
    # Concatenate all batch embeddings
    return torch.cat(embeddings, dim=0)

def save_metadata_and_embeddings(movies, embeddings, metadata_file, embeddings_file):
    """
    Saves movie metadata and embeddings to files.
    
    Args:
        movies (DataFrame): DataFrame containing movie metadata.
        embeddings (torch.Tensor): BERT embeddings for movie titles.
        metadata_file (str): Path to save the metadata CSV file.
        embeddings_file (str): Path to save the embeddings file.
    """
    # Save metadata
    movies.to_csv(metadata_file, index=False)
    print(f"Metadata saved to {metadata_file}")
    
    # Save embeddings
    torch.save(embeddings, embeddings_file)
    print(f"Embeddings saved to {embeddings_file}")

def recommend_movies(movie_id, movies, embeddings, top_n=10):
    """
    Recommends similar movies based on BERT embeddings.
    
    Args:
        movie_id (int): ID of the movie to base recommendations on.
        movies (DataFrame): DataFrame containing movie metadata.
        embeddings (torch.Tensor): BERT embeddings for movie titles.
        top_n (int): Number of recommendations to return.
    
    Returns:
        DataFrame: Top N recommended movies.
    """
    # Get the embedding for the given movie
    movie_idx = movies[movies["movieId"] == movie_id].index[0]
    movie_embedding = embeddings[movie_idx].unsqueeze(0)
    
    # Compute cosine similarity with all other embeddings
    similarities = cosine_similarity(movie_embedding, embeddings)[0]
    
    # Get top N similar movies
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclude the movie itself
    recommended_movies = movies.iloc[similar_indices]
    recommended_movies["similarity"] = similarities[similar_indices]
    
    return recommended_movies

def load_ratings(file_path, sample_size=None):
    """
    Loads and preprocesses the ratings data efficiently.
    
    Args:
        file_path (str): Path to the ratings CSV file.
        sample_size (int, optional): Number of ratings to load. If None, loads all.
    
    Returns:
        DataFrame: Preprocessed ratings data.
    """
    # Definer datatyper for å spare minne
    dtype_dict = {
        'userId': 'int32',  # Bruker int32 i stedet for int64
        'movieId': 'int32',
        'rating': 'float32',  # Bruker float32 i stedet for float64
        'timestamp': 'int32'
    }
    
    # Les data i chunks hvis vi ikke har sample_size
    if sample_size is None:
        # Les først noen rader for å se strukturen
        ratings = pd.read_csv(file_path, dtype=dtype_dict, nrows=1000)
        print("\nEksempel på ratings data:")
        print(ratings.head())
        
        # Les hele filen i chunks
        chunk_size = 1000000  # 1 million rader om gangen
        chunks = []
        for chunk in pd.read_csv(file_path, dtype=dtype_dict, chunksize=chunk_size):
            chunks.append(chunk)
        ratings = pd.concat(chunks, ignore_index=True)
    else:
        # Les bare et utvalg av dataene
        ratings = pd.read_csv(file_path, dtype=dtype_dict, nrows=sample_size)
    
    print(f"\nLastet inn {len(ratings)} vurderinger")
    return ratings

def train_collaborative_filtering_model(ratings, sample_size=1000000):
    """
    Trains a collaborative filtering model using SVD.
    
    Args:
        ratings (DataFrame): User-item interaction data.
        sample_size (int): Number of ratings to use for training.
    
    Returns:
        SVD: Trained collaborative filtering model.
    """
    # Ta et utvalg av dataene for trening
    if len(ratings) > sample_size:
        ratings_sample = ratings.sample(n=sample_size, random_state=42)
    else:
        ratings_sample = ratings
    
    # Prepare the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
    
    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train the SVD model
    model = SVD()
    model.fit(trainset)
    
    # Evaluate the model
    predictions = model.test(testset)
    print("\nCollaborative Filtering Model Performance:")
    accuracy.rmse(predictions)
    
    return model

def hybrid_recommendation(user_id, movie_id, movies, embeddings, cf_model, top_n=10):
    """
    Combines collaborative filtering and content-based filtering for recommendations.
    
    Args:
        user_id (int): ID of the user.
        movie_id (int): ID of the movie to base recommendations on.
        movies (DataFrame): DataFrame containing movie metadata.
        embeddings (torch.Tensor): BERT embeddings for movie titles.
        cf_model (SVD): Trained collaborative filtering model.
        top_n (int): Number of recommendations to return.
    
    Returns:
        DataFrame: Top N recommended movies.
    """
    # Collaborative Filtering: Predict ratings for all movies
    movie_ids = movies["movieId"].tolist()
    cf_predictions = {mid: cf_model.predict(user_id, mid).est for mid in movie_ids}
    
    # Content-Based Filtering: Find similar movies
    movie_idx = movies[movies["movieId"] == movie_id].index[0]
    movie_embedding = embeddings[movie_idx].unsqueeze(0)
    similarities = cosine_similarity(movie_embedding, embeddings)[0]
    
    # Combine CF and CB scores
    alpha = 0.7  # vekt for CF
    beta = 0.3   # vekt for CB
    hybrid_scores = [
        (mid, alpha * cf_predictions[mid] + beta * similarities[idx])
        for idx, mid in enumerate(movie_ids)
    ]
    
    # Sort by hybrid score
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_movies = [mid for mid, _ in hybrid_scores[:top_n]]
    recommended_movies = movies[movies["movieId"].isin(top_movies)].copy()
    recommended_movies["hybrid_score"] = [score for mid, score in hybrid_scores[:top_n]]
    
    return recommended_movies

def run_cli(movies, embeddings, cf_model):
    print("Velkommen til film-anbefalingssystemet!")
    print("Antall brukere: ", ratings['userId'].nunique())
    print("Antall filmer: ", len(movies))
    while True:
        try:
            user_id = int(input("Skriv inn bruker-ID (eller -1 for å avslutte): "))
            if user_id == -1:
                print("Avslutter...")
                break
            if user_id not in ratings['userId'].values:
                print("Ugyldig bruker-ID. Prøv igjen.")
                continue
            movie_id = int(input("Skriv inn movieId på en film du liker: "))
            if movie_id not in movies['movieId'].values:
                print("Ugyldig movieId. Prøv igjen.")
                continue
            recs = hybrid_recommendation(user_id, movie_id, movies, embeddings, cf_model, top_n=10)
            print("\nAnbefalte filmer:")
            print(recs[['movieId', 'title', 'hybrid_score']])
        except Exception as e:
            print(f"Feil: {e}. Prøv igjen.")

# Example usage
if __name__ == "__main__":
    # Load movie and ratings data
    movies = pd.read_csv("ml-32m/ml-32m/movies.csv")
    ratings = load_ratings("ml-32m/ml-32m/ratings.csv")
    
    # Train collaborative filtering model
    cf_model = train_collaborative_filtering_model(ratings)
    
    # Extract embeddings for movie titles
    movie_titles = movies["title"].tolist()
    embeddings_file = "movie_embeddings.pt"
    metadata_file = "movie_metadata.csv"
    
    if not torch.cuda.is_available():
        print("Running on CPU. This may take longer.")
    
    # Check if embeddings already exist
    try:
        embeddings = torch.load(embeddings_file)
        print("Loaded embeddings from file.")
    except FileNotFoundError:
        print("Embeddings file not found. Generating embeddings...")
        embeddings = extract_bert_embeddings(movie_titles, batch_size=32)
        save_metadata_and_embeddings(movies, embeddings, metadata_file, embeddings_file)
    
    # Run CLI for user interaction
    run_cli(movies, embeddings, cf_model)