# Mängelmelder Konstanz - NLP-Analyse

**Projekt: Data Analysis (DLBDSEDA02_D)**  
**Autor: Tobias Seekatz**

Automatisierte Identifikation von Problemthemen aus Bürgerbeschwerden mittels NLP.

## 🎯 Zielsetzung

Analyse unstrukturierter Bürgerbeschwerden aus dem "Mängelmelder Konstanz" zur automatisierten Themenidentifikation.

## ✨ Features

- **Datenvorverarbeitung**: Normalisierung, Tokenisierung, Lemmatisierung (spaCy)
- **Vektorisierung**: CountVectorizer (BoW) und TF-IDF
- **Topic Modeling**: Latent Dirichlet Allocation (LDA)
- **N-Gramm-Analyse**: Bigramme für kontextuelle Wortpaare
- **Visualisierungen**: WordCloud, Plots, pyLDAvis

## 🚀 Schnellstart
```bash
# 1. Installation
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python -m spacy download de_core_news_sm

# 2. Daten ablegen
# Kopiere CSV nach: data/raw/maengelmelder_konstanz.csv

# 3. Ausführen
python main.py

# 4. Ergebnisse
ls output/visualizations/  # Bilder
open output/reports/lda_interactive.html  # Interaktiv
```

## 📊 Verwendung
```bash
# Standard
python main.py

# Mit Optionen
python main.py --input data/raw/meine_daten.csv --topics 7

# Hilfe
python main.py --help
```

## 🛠️ Software-Stack

- **Python 3.8+**
- **pandas**: Datenmanagement
- **spaCy**: Deutsches NLP & Lemmatisierung
- **scikit-learn**: Vektorisierung & LDA
- **matplotlib**: Visualisierung
- **wordcloud**: WordCloud-Erstellung
- **pyLDAvis**: Interaktive Topic-Visualisierung

## 📁 Projektstruktur

Maengelmelder-Konstanz/
├── src/                 # Hauptcode
├── data/raw/            # CSV-Daten
├── output/              # Ergebnisse
├── resources/           # Stoppwörter
├── main.py              # Einstiegspunkt
└── README.md

