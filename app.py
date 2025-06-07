import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from movie_recommender import (
    load_ratings, train_collaborative_filtering_model,
    hybrid_recommendation, extract_bert_embeddings,
    save_metadata_and_embeddings
)

# Last inn miljøvariabler fra .env fil
load_dotenv()

# Hent API-nøkkel fra miljøvariabel
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    st.error("TMDB API-nøkkel ikke funnet. Vennligst opprett en .env fil med TMDB_API_KEY=din_nøkkel")
    st.stop()

# Sett opp sidetittel og konfigurasjon
st.set_page_config(
    page_title="Film-Anbefalingssystem",
    page_icon="🎬",
    layout="wide"
)

# Tilpasset CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .movie-card {
        background-color: #2c3e50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-title {
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 8px;
        color: #ecf0f1;
    }
    .movie-score {
        color: #3498db;
        font-weight: bold;
    }
    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for lasting av data og modeller
with st.sidebar:
    st.title("🎬 Film-Anbefalingssystem")
    st.markdown("---")
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.models_loaded = False
        st.session_state.user_preferences = []  # Ny: Lagre brukerpreferanser
    
    # Legg til en slider for å velge antall vurderinger
    sample_size = st.slider(
        "Antall vurderinger å bruke",
        min_value=100000,
        max_value=1000000,
        value=500000,
        step=100000,
        help="Velg hvor mange vurderinger som skal brukes for trening. Færre vurderinger = raskere trening, men kan påvirke kvaliteten."
    )
    
    # Ny seksjon for brukerpreferanser
    st.subheader("📝 Mine Filmpreferanser")
    st.markdown("Legg til filmer du liker for å få bedre anbefalinger")
    
    # Søkefelt for å legge til filmer
    preference_search = st.text_input("Søk etter film å legge til")
    if preference_search and st.session_state.data_loaded:
        search_results = st.session_state.movies[
            st.session_state.movies['title'].str.contains(preference_search, case=False)
        ].head(5)
        
        if not search_results.empty:
            selected_movie = st.selectbox(
                "Velg film",
                options=search_results['movieId'],
                format_func=lambda x: st.session_state.movies[
                    st.session_state.movies['movieId'] == x
                ]['title'].iloc[0]
            )
            
            rating = st.slider("Din vurdering", 1.0, 5.0, 3.0, 0.5)
            
            if st.button("Legg til film"):
                movie_title = st.session_state.movies[
                    st.session_state.movies['movieId'] == selected_movie
                ]['title'].iloc[0]
                
                # Legg til i brukerpreferanser
                st.session_state.user_preferences.append({
                    'movieId': selected_movie,
                    'title': movie_title,
                    'rating': rating
                })
                st.success(f"La til {movie_title} med vurdering {rating}")
    
    # Vis lagrede preferanser
    if st.session_state.user_preferences:
        st.markdown("### Mine Lagrede Filmer")
        for pref in st.session_state.user_preferences:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(pref['title'])
            with col2:
                if st.button("🗑️", key=f"delete_{pref['movieId']}"):
                    st.session_state.user_preferences = [
                        p for p in st.session_state.user_preferences 
                        if p['movieId'] != pref['movieId']
                    ]
                    st.rerun()
            st.write(f"Vurdering: {pref['rating']} ⭐")

    if st.button("Last inn data og tren modeller"):
        with st.spinner("Laster inn data og trener modeller..."):
            # Last inn data
            movies = pd.read_csv("ml-32m/ml-32m/movies.csv")
            ratings = load_ratings("ml-32m/ml-32m/ratings.csv", sample_size=sample_size)
            
            # Tren modell
            cf_model = train_collaborative_filtering_model(ratings, sample_size=sample_size)
            
            # Last eller generer embeddings
            try:
                embeddings = torch.load("movie_embeddings.pt")
                st.success("Lastet inn embeddings fra fil")
            except FileNotFoundError:
                with st.spinner("Genererer BERT-embeddings (dette kan ta litt tid)..."):
                    movie_titles = movies["title"].tolist()
                    embeddings = extract_bert_embeddings(movie_titles, batch_size=32)
                    save_metadata_and_embeddings(movies, embeddings, "movie_metadata.csv", "movie_embeddings.pt")
            
            # Lagre i session state
            st.session_state.movies = movies
            st.session_state.ratings = ratings
            st.session_state.cf_model = cf_model
            st.session_state.embeddings = embeddings
            st.session_state.data_loaded = True
            st.session_state.models_loaded = True
            st.success("Data og modeller lastet inn!")

# Hovedinnhold
st.title("🎬 Film-Anbefalingssystem")
st.markdown("""
    Dette systemet kombinerer BERT-basert innholdsfiltrering med kollaborativ filtrering 
    for å gi deg personlige film-anbefalinger.
""")

if not st.session_state.data_loaded:
    st.info("👈 Klikk på 'Last inn data og tren modeller' i sidepanelet for å komme i gang")
else:
    # To kolonner for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Velg bruker")
        user_id = st.number_input("Skriv inn bruker-ID", min_value=1, max_value=st.session_state.ratings['userId'].max())
        
        # Vis brukerstatistikk
        user_ratings = st.session_state.ratings[st.session_state.ratings['userId'] == user_id]
        if not user_ratings.empty:
            st.write(f"Antall vurderinger: {len(user_ratings)}")
            st.write(f"Gjennomsnittlig vurdering: {user_ratings['rating'].mean():.2f}")
            
            # Plot brukerens vurderingsfordeling
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=user_ratings, x='rating', bins=10)
            plt.title('Brukerens Vurderingsfordeling')
            st.pyplot(fig)
    
    with col2:
        st.subheader("Velg film")
        # Søkefelt for filmer
        search_query = st.text_input("Søk etter film")
        if search_query:
            search_results = st.session_state.movies[
                st.session_state.movies['title'].str.contains(search_query, case=False)
            ].head(10)
            if not search_results.empty:
                selected_movie = st.selectbox(
                    "Velg en film fra søkeresultatene",
                    options=search_results['movieId'],
                    format_func=lambda x: st.session_state.movies[
                        st.session_state.movies['movieId'] == x
                    ]['title'].iloc[0]
                )
            else:
                st.warning("Ingen filmer funnet")
                selected_movie = None
        else:
            selected_movie = None

    # Legg til TMDB API nøkkel og funksjon for å hente filmplakater
    # TMDB_API_KEY er nå hentet fra miljøvariabelen over

    def get_movie_poster(movie_title, year=None):
        """Henter filmplakat fra TMDB API"""
        import requests
        
        # Fjern årstallet fra tittelen hvis det finnes
        if year:
            search_title = movie_title.replace(f" ({year})", "")
        else:
            search_title = movie_title
        
        # Søk etter filmen
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={search_title}"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        # Returner en standard plakat hvis ingen finnes
        return "https://via.placeholder.com/500x750?text=Ingen+plakat+tilgjengelig"

    # Anbefalinger
    if st.button("Få anbefalinger"):
        with st.spinner("Genererer anbefalinger..."):
            if st.session_state.user_preferences:
                # Bruk brukerpreferanser for å generere anbefalinger
                all_recommendations = []
                for pref in st.session_state.user_preferences:
                    recs = hybrid_recommendation(
                        None,  # Ingen spesifikk bruker-ID
                        pref['movieId'],
                        st.session_state.movies,
                        st.session_state.embeddings,
                        st.session_state.cf_model,
                        top_n=5
                    )
                    # Juster score basert på brukerens vurdering
                    recs['hybrid_score'] *= (pref['rating'] / 5.0)
                    all_recommendations.append(recs)
                
                # Kombiner og sorter anbefalinger
                combined_recs = pd.concat(all_recommendations)
                combined_recs = combined_recs.groupby('movieId').agg({
                    'title': 'first',
                    'hybrid_score': 'mean'
                }).reset_index()
                
                # Fjern filmer som allerede er i brukerpreferanser
                user_movie_ids = [p['movieId'] for p in st.session_state.user_preferences]
                combined_recs = combined_recs[~combined_recs['movieId'].isin(user_movie_ids)]
                
                # Sorter og ta topp 10
                recommendations = combined_recs.sort_values('hybrid_score', ascending=False).head(10)
            else:
                st.warning("Legg til noen filmer i dine preferanser for å få personlige anbefalinger!")
                st.stop()

            st.subheader("🎯 Anbefalte filmer basert på dine preferanser")
            
            # Vis anbefalinger i et grid med plakater
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(recommendations.iterrows()):
                with cols[idx % 3]:
                    # Hent filmplakat
                    poster_url = get_movie_poster(movie['title'])
                    
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-poster" alt="{movie['title']}">
                            <div class="movie-title">{movie['title']}</div>
                            <div class="movie-score">Score: {movie['hybrid_score']:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Plot anbefalingenes fordeling
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=recommendations, x='hybrid_score', y='title')
            plt.title('Anbefalingenes Fordeling')
            plt.xlabel('Hybrid Score')
            plt.ylabel('Film')
            st.pyplot(fig)

    # Statistikk og informasjon
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Systemstatistikk")
        st.write(f"Totalt antall filmer: {len(st.session_state.movies)}")
        st.write(f"Totalt antall brukere: {st.session_state.ratings['userId'].nunique()}")
        st.write(f"Totalt antall vurderinger: {len(st.session_state.ratings)}")
        
        # Plot vurderingsfordeling
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=st.session_state.ratings, x='rating', bins=20)
        plt.title('Total Vurderingsfordeling')
        st.pyplot(fig)
    
    with col4:
        st.subheader("ℹ️ Om systemet")
        st.markdown("""
            Dette anbefalingssystemet bruker en hybrid tilnærming som kombinerer:
            - **BERT-basert innholdsfiltrering**: Analyserer filmtitler for å finne semantiske sammenhenger
            - **Kollaborativ filtrering**: Bruker brukervurderinger for å finne mønstre
            - **Hybrid scoring**: Kombinerer begge metodene for bedre anbefalinger
        """) 