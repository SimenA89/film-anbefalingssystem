import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Film-Anbefalingssystem med BERT og Hybrid Tilnærming\n",
                "\n",
                "Dette notebook implementerer et film-anbefalingssystem som kombinerer:\n",
                "- BERT-basert innholdsfiltrering\n",
                "- Kollaborativ filtrering (SVD)\n",
                "- Hybrid anbefalingsmodell\n",
                "\n",
                "## Installasjon av nødvendige pakker\n",
                "Kjør følgende celle for å installere nødvendige pakker hvis de ikke allerede er installert."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install transformers torch pandas scikit-learn surprise matplotlib seaborn"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Importering av biblioteker"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Importer nødvendige biblioteker\n",
                "from transformers import BertTokenizer, BertModel\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.metrics.pairwise import cosine_similarity\n",
                "from surprise import SVD, Dataset, Reader\n",
                "from surprise.model_selection import train_test_split\n",
                "from surprise import accuracy\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Sett opp plotting\n",
                "plt.style.use('seaborn')\n",
                "sns.set_palette('husl')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Hjelpefunksjoner\n",
                "\n",
                "Her definerer vi alle nødvendige funksjoner for systemet."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_bert_embeddings(texts, batch_size=32):\n",
                "    \"\"\"\n",
                "    Extracts BERT embeddings for a list of texts in batches.\n",
                "    \n",
                "    Args:\n",
                "        texts (list): List of text strings.\n",
                "        batch_size (int): Number of texts to process in each batch.\n",
                "    \n",
                "    Returns:\n",
                "        torch.Tensor: BERT embeddings.\n",
                "    \"\"\"\n",
                "    # Load pre-trained BERT model and tokenizer\n",
                "    tokenizer = BertTokenizer.from_pretrained(\"bert-base-uncased\")\n",
                "    model = BertModel.from_pretrained(\"bert-base-uncased\")\n",
                "    \n",
                "    # Move model to GPU if available\n",
                "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "    model = model.to(device)\n",
                "    \n",
                "    embeddings = []\n",
                "    \n",
                "    # Process texts in batches\n",
                "    for i in range(0, len(texts), batch_size):\n",
                "        batch_texts = texts[i:i + batch_size]\n",
                "        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors=\"pt\", max_length=256)\n",
                "        \n",
                "        # Move inputs to the same device as the model\n",
                "        inputs = {key: val.to(device) for key, val in inputs.items()}\n",
                "        \n",
                "        # Get BERT embeddings\n",
                "        with torch.no_grad():\n",
                "            outputs = model(**inputs)\n",
                "            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling\n",
                "            embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory\n",
                "    \n",
                "    # Concatenate all batch embeddings\n",
                "    return torch.cat(embeddings, dim=0)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def save_metadata_and_embeddings(movies, embeddings, metadata_file, embeddings_file):\n",
                "    \"\"\"\n",
                "    Saves movie metadata and embeddings to files.\n",
                "    \n",
                "    Args:\n",
                "        movies (DataFrame): DataFrame containing movie metadata.\n",
                "        embeddings (torch.Tensor): BERT embeddings for movie titles.\n",
                "        metadata_file (str): Path to save the metadata CSV file.\n",
                "        embeddings_file (str): Path to save the embeddings file.\n",
                "    \"\"\"\n",
                "    # Save metadata\n",
                "    movies.to_csv(metadata_file, index=False)\n",
                "    print(f\"Metadata saved to {metadata_file}\")\n",
                "    \n",
                "    # Save embeddings\n",
                "    torch.save(embeddings, embeddings_file)\n",
                "    print(f\"Embeddings saved to {embeddings_file}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def recommend_movies(movie_id, movies, embeddings, top_n=10):\n",
                "    \"\"\"\n",
                "    Recommends similar movies based on BERT embeddings.\n",
                "    \n",
                "    Args:\n",
                "        movie_id (int): ID of the movie to base recommendations on.\n",
                "        movies (DataFrame): DataFrame containing movie metadata.\n",
                "        embeddings (torch.Tensor): BERT embeddings for movie titles.\n",
                "        top_n (int): Number of recommendations to return.\n",
                "    \n",
                "    Returns:\n",
                "        DataFrame: Top N recommended movies.\n",
                "    \"\"\"\n",
                "    # Get the embedding for the given movie\n",
                "    movie_idx = movies[movies[\"movieId\"] == movie_id].index[0]\n",
                "    movie_embedding = embeddings[movie_idx].unsqueeze(0)\n",
                "    \n",
                "    # Compute cosine similarity with all other embeddings\n",
                "    similarities = cosine_similarity(movie_embedding, embeddings)[0]\n",
                "    \n",
                "    # Get top N similar movies\n",
                "    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclude the movie itself\n",
                "    recommended_movies = movies.iloc[similar_indices]\n",
                "    recommended_movies[\"similarity\"] = similarities[similar_indices]\n",
                "    \n",
                "    return recommended_movies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def load_ratings(file_path):\n",
                "    \"\"\"\n",
                "    Loads and preprocesses the ratings data.\n",
                "    \n",
                "    Args:\n",
                "        file_path (str): Path to the ratings CSV file.\n",
                "    \n",
                "    Returns:\n",
                "        DataFrame: Preprocessed ratings data.\n",
                "    \"\"\"\n",
                "    ratings = pd.read_csv(file_path)\n",
                "    print(\"\\nRatings Data:\")\n",
                "    print(ratings.head())\n",
                "    return ratings"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def train_collaborative_filtering_model(ratings):\n",
                "    \"\"\"\n",
                "    Trains a collaborative filtering model using SVD.\n",
                "    \n",
                "    Args:\n",
                "        ratings (DataFrame): User-item interaction data.\n",
                "    \n",
                "    Returns:\n",
                "        SVD: Trained collaborative filtering model.\n",
                "    \"\"\"\n",
                "    # Prepare the data for Surprise library\n",
                "    reader = Reader(rating_scale=(0.5, 5.0))\n",
                "    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)\n",
                "    \n",
                "    # Split the data into training and testing sets\n",
                "    trainset, testset = train_test_split(data, test_size=0.2)\n",
                "    \n",
                "    # Train the SVD model\n",
                "    model = SVD()\n",
                "    model.fit(trainset)\n",
                "    \n",
                "    # Evaluate the model\n",
                "    predictions = model.test(testset)\n",
                "    print(\"\\nCollaborative Filtering Model Performance:\")\n",
                "    accuracy.rmse(predictions)\n",
                "    \n",
                "    return model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def hybrid_recommendation(user_id, movie_id, movies, embeddings, cf_model, top_n=10):\n",
                "    \"\"\"\n",
                "    Combines collaborative filtering and content-based filtering for recommendations.\n",
                "    \n",
                "    Args:\n",
                "        user_id (int): ID of the user.\n",
                "        movie_id (int): ID of the movie to base recommendations on.\n",
                "        movies (DataFrame): DataFrame containing movie metadata.\n",
                "        embeddings (torch.Tensor): BERT embeddings for movie titles.\n",
                "        cf_model (SVD): Trained collaborative filtering model.\n",
                "        top_n (int): Number of recommendations to return.\n",
                "    \n",
                "    Returns:\n",
                "        DataFrame: Top N recommended movies.\n",
                "    \"\"\"\n",
                "    # Collaborative Filtering: Predict ratings for all movies\n",
                "    movie_ids = movies[\"movieId\"].tolist()\n",
                "    cf_predictions = {mid: cf_model.predict(user_id, mid).est for mid in movie_ids}\n",
                "    \n",
                "    # Content-Based Filtering: Find similar movies\n",
                "    movie_idx = movies[movies[\"movieId\"] == movie_id].index[0]\n",
                "    movie_embedding = embeddings[movie_idx].unsqueeze(0)\n",
                "    similarities = cosine_similarity(movie_embedding, embeddings)[0]\n",
                "    \n",
                "    # Combine CF and CB scores\n",
                "    alpha = 0.7  # vekt for CF\n",
                "    beta = 0.3   # vekt for CB\n",
                "    hybrid_scores = [\n",
                "        (mid, alpha * cf_predictions[mid] + beta * similarities[idx])\n",
                "        for idx, mid in enumerate(movie_ids)\n",
                "    ]\n",
                "    \n",
                "    # Sort by hybrid score\n",
                "    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)\n",
                "    \n",
                "    # Get top N recommendations\n",
                "    top_movies = [mid for mid, _ in hybrid_scores[:top_n]]\n",
                "    recommended_movies = movies[movies[\"movieId\"].isin(top_movies)].copy()\n",
                "    recommended_movies[\"hybrid_score\"] = [score for mid, score in hybrid_scores[:top_n]]\n",
                "    \n",
                "    return recommended_movies"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Dataanalyse og Forberedelse\n",
                "\n",
                "La oss først utforske datasettene og forberede dataene."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Last inn data\n",
                "movies = pd.read_csv(\"ml-32m/ml-32m/movies.csv\")\n",
                "ratings = load_ratings(\"ml-32m/ml-32m/ratings.csv\")\n",
                "\n",
                "# Vis informasjon om datasettene\n",
                "print(\"Filminformasjon:\")\n",
                "print(f\"Antall filmer: {len(movies)}\")\n",
                "print(\"\\nEksempel på filmer:\")\n",
                "display(movies.head())\n",
                "\n",
                "print(\"\\nVurderingsinformasjon:\")\n",
                "print(f\"Antall brukere: {ratings['userId'].nunique()}\")\n",
                "print(f\"Antall vurderinger: {len(ratings)}\")\n",
                "print(\"\\nEksempel på vurderinger:\")\n",
                "display(ratings.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Visualisering av Data\n",
                "\n",
                "La oss visualisere noen interessante mønstre i dataene."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribusjon av vurderinger\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.histplot(data=ratings, x='rating', bins=20)\n",
                "plt.title('Distribusjon av Filmvurderinger')\n",
                "plt.xlabel('Vurdering')\n",
                "plt.ylabel('Antall')\n",
                "plt.show()\n",
                "\n",
                "# Topp 10 mest vurderte filmer\n",
                "top_movies = ratings.groupby('movieId').size().sort_values(ascending=False).head(10)\n",
                "top_movies_titles = movies[movies['movieId'].isin(top_movies.index)]['title']\n",
                "\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.barplot(x=top_movies.values, y=top_movies_titles)\n",
                "plt.title('Topp 10 Mest Vurderte Filmer')\n",
                "plt.xlabel('Antall Vurderinger')\n",
                "plt.ylabel('Film')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Treningsmodeller\n",
                "\n",
                "Nå skal vi trene både kollaborativ filtrering og generere BERT-embeddings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tren kollaborativ filtrering\n",
                "print(\"Trener kollaborativ filtreringsmodell...\")\n",
                "cf_model = train_collaborative_filtering_model(ratings)\n",
                "\n",
                "# Generer eller last inn BERT-embeddings\n",
                "print(\"\\nGenererer/laster inn BERT-embeddings...\")\n",
                "movie_titles = movies[\"title\"].tolist()\n",
                "embeddings_file = \"movie_embeddings.pt\"\n",
                "metadata_file = \"movie_metadata.csv\"\n",
                "\n",
                "if not torch.cuda.is_available():\n",
                "    print(\"Kjører på CPU. Dette kan ta litt tid.\")\n",
                "\n",
                "try:\n",
                "    embeddings = torch.load(embeddings_file)\n",
                "    print(\"Lastet inn embeddings fra fil.\")\n",
                "except FileNotFoundError:\n",
                "    print(\"Embeddings-fil ikke funnet. Genererer embeddings...\")\n",
                "    embeddings = extract_bert_embeddings(movie_titles, batch_size=32)\n",
                "    save_metadata_and_embeddings(movies, embeddings, metadata_file, embeddings_file)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Test Anbefalingssystemet\n",
                "\n",
                "La oss teste systemet med noen eksempler."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def test_recommendations(user_id, movie_id):\n",
                "    print(f\"\\nAnbefalinger for bruker {user_id} basert på film {movie_id}:\")\n",
                "    recommendations = hybrid_recommendation(user_id, movie_id, movies, embeddings, cf_model, top_n=10)\n",
                "    display(recommendations[['movieId', 'title', 'hybrid_score']])\n",
                "\n",
                "# Test med noen eksempler\n",
                "test_recommendations(user_id=1, movie_id=1)  # Juster disse ID-ene basert på dine data\n",
                "test_recommendations(user_id=2, movie_id=2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Interaktiv Anbefaling\n",
                "\n",
                "Du kan nå teste systemet interaktivt ved å kjøre følgende celle og følge instruksjonene."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import clear_output\n",
                "\n",
                "def interactive_recommendation():\n",
                "    while True:\n",
                "        try:\n",
                "            user_id = int(input(\"Skriv inn bruker-ID (eller -1 for å avslutte): \"))\n",
                "            if user_id == -1:\n",
                "                print(\"Avslutter...\")\n",
                "                break\n",
                "            if user_id not in ratings['userId'].values:\n",
                "                print(\"Ugyldig bruker-ID. Prøv igjen.\")\n",
                "                continue\n",
                "                \n",
                "            movie_id = int(input(\"Skriv inn movieId på en film du liker: \"))\n",
                "            if movie_id not in movies['movieId'].values:\n",
                "                print(\"Ugyldig movieId. Prøv igjen.\")\n",
                "                continue\n",
                "                \n",
                "            clear_output(wait=True)\n",
                "            recs = hybrid_recommendation(user_id, movie_id, movies, embeddings, cf_model, top_n=10)\n",
                "            print(f\"\\nAnbefalinger for bruker {user_id} basert på film {movie_id}:\")\n",
                "            display(recs[['movieId', 'title', 'hybrid_score']])\n",
                "            \n",
                "        except Exception as e:\n",
                "            print(f\"Feil: {e}. Prøv igjen.\")\n",
                "\n",
                "interactive_recommendation()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Lagre notebook-filen
with open('movie_recommendation_system.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook-fil opprettet!") 