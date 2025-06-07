# Film-Anbefalingssystem 🎬

Dette er et intelligent film-anbefalingssystem bygget med Python og Streamlit som kombinerer BERT-basert innholdsfiltrering med kollaborativ filtrering for å gi personlige film-anbefalinger.

## Funksjoner

- 🔍 Søk og utforsk filmer
- 🤖 Hybrid anbefalingssystem som kombinerer:
  - BERT-basert innholdsfiltrering
  - Kollaborativ filtrering
- 📊 Interaktiv brukergrensesnitt med Streamlit
- 🎯 Personlige anbefalinger basert på brukerpreferanser
- 🖼️ Filmplakater og detaljert informasjon

## Teknisk Stack

- Python 3.x
- Streamlit
- PyTorch
- Transformers (BERT)
- Pandas
- Scikit-learn
- Surprise (for kollaborativ filtrering)

## Installasjon

1. Klon repositoriet:
```bash
git clone https://github.com/SimenA89/film-anbefalingssystem.git
cd film-anbefalingssystem
```

2. Installer avhengigheter:
```bash
pip install -r requirements.txt
```

3. Last ned datasettet:
- Last ned MovieLens 32M datasettet fra [MovieLens](https://grouplens.org/datasets/movielens/)
- Pakk ut filene i en mappe kalt `ml-32m` i prosjektets rotmappe

4. Konfigurer TMDB API-nøkkel:
- Gå til [TMDB](https://www.themoviedb.org/settings/api) og opprett en konto
- Generer en API-nøkkel
- Opprett en `.env` fil i prosjektets rotmappe med følgende innhold:
```
TMDB_API_KEY=din_api_nøkkel_her
```

5. Start applikasjonen:
```bash
streamlit run app.py
```

## Bruk

1. Åpne applikasjonen i nettleseren
2. Last inn data og tren modeller ved å klikke på knappen i sidepanelet
3. Legg til filmer du liker i dine preferanser
4. Få personlige anbefalinger basert på dine preferanser

## Lisens

Dette prosjektet er lisensiert under MIT-lisensen - se [LICENSE](LICENSE) filen for detaljer.

## Bidrag

Bidrag er velkomne! Vennligst åpne en issue eller pull request for å foreslå endringer eller forbedringer. 