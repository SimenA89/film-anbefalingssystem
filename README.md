<div align="center">
  <h1>🎬 Movie Recommendation System (Hybrid AI)</h1>
  <p>An intelligent movie recommendation engine built with Python, PyTorch, and Streamlit.</p>

  <a href="https://github.com/SimenA89/film-anbefalingssystem">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </a>
  <a href="https://github.com/SimenA89/film-anbefalingssystem">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  </a>
  <a href="https://github.com/SimenA89/film-anbefalingssystem">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  </a>
  <a href="https://github.com/SimenA89/film-anbefalingssystem">
    <img src="https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Transformers" />
  </a>
</div>

<br/>

## 🎯 Overview
This project is an advanced, hybrid movie recommendation system that goes beyond simple rating averages. It combines **Natural Language Processing (NLP)** through BERT embeddings with **Collaborative Filtering** (SVD) to provide highly personalized movie suggestions based on both the semantic content of movies and user behavior patterns.

> **Business Value:** Demonstrates the ability to build and deploy complex machine learning pipelines, integrate external APIs (TMDB), and create interactive data apps that end-users can actually interact with.

## ✨ Features
*   **Hybrid Recommendation Engine:** 
    *   **Content-Based Filtering:** Uses pre-trained BERT (Transformers) to extract semantic meaning from movie titles and metadata.
    *   **Collaborative Filtering:** Uses user rating patterns (Surprise/SVD) to recommend movies based on similar users.
*   **Interactive UI:** A clean, responsive dashboard built with Streamlit.
*   **Dynamic Posters:** Integrates with the TMDB API to fetch real-time movie posters and metadata.
*   **User Preferences:** Users can input their own ratings to instantly get tailored recommendations.

## 🚀 Live Demo
*(Optional: Add a link here if you deploy it to Streamlit Community Cloud! Example: `[Live Demo](https://your-app.streamlit.app)`)*

---

## 🛠️ Tech Stack
*   **Machine Learning:** PyTorch, HuggingFace Transformers (BERT), Scikit-learn, Surprise (SVD)
*   **Data Processing:** Pandas, Numpy
*   **Frontend/Deployment:** Streamlit
*   **Visualization:** Matplotlib, Seaborn
*   **APIs:** TMDB API

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/SimenA89/film-anbefalingssystem.git
cd film-anbefalingssystem
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset & API
1.  **Dataset:** Download the [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/) and extract it into a folder named `ml-32m` in the root directory.
2.  **TMDB API:** Get a free API key from [The Movie Database (TMDB)](https://www.themoviedb.org/settings/api).
3.  Create a `.env` file in the root directory and add your key:
    ```
    TMDB_API_KEY=your_api_key_here
    ```

### 4. Launch the App
```bash
streamlit run app.py
```

## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a pull request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.