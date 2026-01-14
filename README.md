# Mängelmelder Konstanz - NLP-Analyse

**Projekt: Data Analysis (DLBDSEDA02_D)**  
**Autor: Tobias Seekatz**  
**Sprache: Deutsch (NLP für deutsche Texte)**

Automatisierte Identifikation von Problemthemen aus Bürgerbeschwerden mittels Natural Language Processing (NLP). Das System analysiert unstrukturierte Texteingaben von Bürgern und extrahiert automatisch semantische Themen, häufig vorkommende Wortmuster und visualisiert diese interaktiv.

---

## 🎯 Zielsetzung

Das Projekt löst das Problem der manuellen Kategorisierung von tausenden Bürgerbeschwerden in der Stadt Konstanz. Mittels moderner NLP-Techniken werden:

- **Automatische Themenextraktion**: Identifiziert 5-N Hauptthemen aus unstrukturierten Texten
- **Kontextuelle Wortmuster**: Findet wiederkehrende Bigramme (z.B. „Straßenlaterne defekt")
- **Semantische Analyse**: Versteht Umlaute, Varianten und deutsche Grammatik
- **Interaktive Exploration**: Erlaubt explorative Datenanalyse über pyLDAvis Dashboard

### Beispiel-Erkenntnisse aus den Daten:
```
Top-Thema 1: Illegale Müllablagerung, abgestellte Fahrzeuge
Top-Thema 2: Gehwegschäden, Baumverschmutzung
Top-Thema 3: Radweginstandhaltung, Straßenbeleuchtung
Top-Thema 4: Fußgänger-/Radfahrer-Sicherheit
Top-Thema 5: Defekte Straßenlaternen (häufigste Beschwerde: 377x Wort „defekt")
```

## ✨ Features

### 1. **Datenvorverarbeitung (Text Cleaning)**
- Unicode-Normalisierung (NFC) für deutsche Umlaute (ä, ö, ü, ß)
- Entfernung von Satzzeichen & Zahlen
- Automatische Stoppwort-Filterung (195 deutsche Stoppwörter)
- **Lemmatisierung**: spaCy-Modell `de_core_news_sm` mit NLTK-Fallback
- **Auto-Download**: Modell wird automatisch heruntergeladen falls nicht vorhanden

### 2. **Vektorisierung**
- **CountVectorizer**: Bag-of-Words (BoW) Darstellung
- **TF-IDF**: Term Frequency-Inverse Document Frequency für gewichtete Features
- Maximale Features: 500 (konfigurierbar)
- Dokumentfrequenz-Filterung: min 2 Dokumente, max 95%

### 3. **Topic Modeling (LDA)**
- **Latent Dirichlet Allocation** mit scikit-learn
- 5 Themen (konfigurierbar via `--topics` Flag)
- 20 Iterationen (bis zur Konvergenz)
- Extraktion der Top-10 Wörter pro Thema

### 4. **N-Gramm-Analyse**
- Bigramme (2er-Wortpaare) werden extrahiert
- Top-20 Bigramme mit Häufigkeitszähler
- Ermöglicht Erkennung von häufigen Phrasen (z.B. „Straßenlaterne defekt")

### 5. **Visualisierungen**
- **WordCloud**: Größe der Wörter proportional zur Häufigkeit
- **Top-Wörter Bar-Plot**: Horizontales Balkendiagramm der 20 häufigsten Wörter
- **pyLDAvis HTML-Dashboard**: Interaktive 2D-Themenvisualisierung (t-SNE Reduktion)
  - Klicke auf Themen-Blasen zum Erkunden
  - Sieh relevante Wörter pro Thema
  - Themenverteilung über alle Dokumente

## 🚀 Schnellstart

### Installation (5 Minuten)

#### Option 1: Mit Skript (empfohlen)
```bash
# Klone das Repository
git clone https://github.com/Toby2906/Maengelmelder-Konstanz.git
cd Maengelmelder-Konstanz

# Erstelle Virtual Environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# oder: .venv\Scripts\activate  # Windows

# Installiere Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Manuelle spaCy-Installation (normalerweise nicht nötig!)
# python -m spacy download de_core_news_sm
```

#### Option 2: Mit Docker
```bash
docker build -t maengelmelder-nlp .
docker run -v $(pwd)/data:/app/data maengelmelder-nlp python main.py
```

### Daten vorbereiten

1. CSV-Datei mit Bürgerbeschwerden (UTF-8 encoding):
   ```csv
   ID,Kategorie,Beschreibung
   1,Müll,"Illegale Müllablagerung am Hafenplatz"
   2,Straße,"Straßenlaterne vor Haus 42 ist defekt"
   ...
   ```

2. Kopiere in: `data/raw/maengelmelder_konstanz.csv`

### Analyse starten

```bash
# Standard: 5 Themen
python main.py

# Mit benutzerdefinierten Parametern
python main.py --input data/raw/custom_data.csv --topics 7

# Debug-Modus (verbose logging)
python main.py --verbose
```

### Ergebnisse anschauen

```bash
# Statische Visualisierungen
ls output/visualizations/
# → wordcloud.png          (Wort-Häufigkeit)
# → top_words.png          (Top-20 Wörter)

# Interaktives Dashboard öffnen
cd output/reports
python -m http.server 8000
# Browser: http://localhost:8000/lda_interactive.html

# Rohe Ergebnisse (JSON)
cat output/analysis/analysis_results.json
```

## 📊 Verwendung & Konfiguration

### Kommandozeilen-Argumente

```bash
# Alle verfügbaren Optionen
python main.py --help

# Beispiele:
python main.py --topics 10                              # 10 statt 5 Themen
python main.py --input my_data.csv --topics 7          # Custom CSV + 7 Themen
python main.py --verbose                               # Debug-Output
```

### Konfiguration ändern (src/config.py)

```python
# Text-Vorverarbeitung
MIN_WORD_LENGTH = 3          # Nur Wörter ≥ 3 Zeichen
REMOVE_PUNCTUATION = True    # Satzzeichen entfernen
USE_LEMMATIZATION = True     # Lemmatisierung ein/aus

# Vektorisierung
VECTORIZER_MAX_FEATURES = 500
VECTORIZER_MIN_DF = 2        # Min. 2 Dokumente pro Token

# Topic Modeling
LDA_N_TOPICS = 5
LDA_MAX_ITER = 20

# Visualisierung
WORDCLOUD_MAX_WORDS = 100
PLOT_DPI = 150              # Bildauflösung
```

### Wichtige Hinweise

> ⚠️ **spaCy-Modell**: Fehlt `de_core_news_sm`, wird automatisch heruntergeladen. Falls das fehlschlägt:
> ```bash
> python -m spacy download de_core_news_sm
> ```
> Fallback: NLTK SnowballStemmer wird als Lemmatizer verwendet.

> 💡 **pyLDAvis**: Optional für interaktive HTML-Visualisierung. Falls fehlend:
> ```bash
> pip install pyLDAvis
> ```
> Ohne pyLDAvis: Analyse läuft normal, nur kein interaktives Dashboard.

> 📈 **Performance**: Bei >5000 Texten dauert LDA länger. Nutze `--topics 3` für schnellere Tests.

---

## 🛠️ Architektur & Module

### Dateistruktur

```
Maengelmelder-Konstanz/
│
├── main.py                    # Einstiegspunkt, orchestriert Pipeline
├── src/
│   ├── __init__.py
│   ├── config.py              # Zentrale Konfiguration
│   ├── data_loader.py         # CSV-Laden & Basis-Statistiken
│   ├── preprocessor.py        # Text-Cleaning, Lemmatisierung
│   ├── vectorizer.py          # CountVectorizer & TF-IDF
│   ├── topic_modeler.py       # LDA-Training
│   ├── analyzer.py            # N-Gramm, Top-Wörter
│   └── visualizer.py          # WordCloud, Plots, pyLDAvis
│
├── data/
│   ├── raw/                   # Eingabe (CSV)
│   └── processed/             # Temporäre Outputs
│
├── resources/
│   └── stopwords_de.txt       # 195 deutsche Stoppwörter
│
├── output/
│   ├── visualizations/        # WordCloud, Plots (PNG)
│   ├── reports/               # pyLDAvis HTML
│   └── analysis/              # JSON mit Ergebnissen
│
├── requirements.txt           # Python-Dependencies
└── README.md                  # Diese Datei
```

### Pipeline-Flow

```
1. Daten laden (CSV) → 2132 Texte
      ↓
2. Vorverarbeitung (spaCy/NLTK) → 2127 saubere Texte, 5109 Tokens
      ↓
3. Vektorisierung (CountVectorizer) → 2127×500 Matrix
      ↓
4. LDA Topic Modeling → 5 Themen
      ↓
5. N-Gramm Analyse → Top-20 Bigramme
      ↓
6. Visualisierungen & Export → PNG + HTML + JSON
```

---

## 🔧 Software-Stack Detailliert

| Komponente | Paket | Version | Zweck |
|-----------|-------|---------|-------|
| NLP | `spacy` | 3.7+ | Lemmatisierung, Tokenisierung |
| NLP (Fallback) | `nltk` | 3.8+ | Stemming (wenn spaCy fehlt) |
| ML | `scikit-learn` | 1.3+ | Vektorisierung, LDA |
| Daten | `pandas` | 2.0+ | CSV-Verarbeitung, Statistiken |
| Visualisierung | `matplotlib` | 3.8+ | Plots |
| Visualisierung | `wordcloud` | 1.9+ | WordCloud-Erstellung |
| Visualisierung | `pyLDAvis` | 3.4+ | Interaktive Topic-Visualisierung |
| Datenverarbeitung | `numpy` | 1.24+ | Numerische Operationen |
| Datenverarbeitung | `scipy` | 1.17+ | Wissenschaftliche Funktionen |

---

## 📈 Beispiel-Output

### Analyse-Ergebnisse (analysis_results.json)
```json
{
  "data_stats": {
    "total_texts": 2132,
    "avg_words": 18.4,
    "avg_length": 133.7
  },
  "preprocessing_stats": {
    "total_texts": 2127,
    "unique_tokens": 5109,
    "avg_tokens_per_text": 9.4
  },
  "topics": {
    "Thema 1": ["steht", "müll", "strasse", "liegt", "abgestellt", ...],
    "Thema 2": ["schild", "gehweg", "baum", "liegt", "fussweg", ...],
    ...
  },
  "top_bigrams": {
    "strassenlaterne defekt": 54,
    "lamp defekt": 42,
    "laterne defekt": 32,
    ...
  },
  "top_words": {
    "defekt": 377,
    "strasse": 251,
    "strassenlaterne": 236,
    ...
  }
}
```

### Visualisierungen
- `wordcloud.png`: Visuelle Darstellung der Wort-Häufigkeiten
- `top_words.png`: Top-20 Wörter als Balkendiagramm
- `lda_interactive.html`: Interaktives pyLDAvis Dashboard (öffne im Browser)

---

## 🐛 Troubleshooting

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: No module named 'spacy'` | `pip install -r requirements.txt` |
| `OSError: No such file or directory: 'data/raw/...'` | Prüfe CSV-Pfad, muss UTF-8 sein |
| `Empty vocabulary after vectorization` | Stoppwort-Filter zu aggressiv? Prüfe `config.py` |
| `pyLDAvis nicht verfügbar` | `pip install pyLDAvis` |
| `spaCy Modell 'de_core_news_sm' nicht gefunden` | Auto-Download startet, oder manuell: `python -m spacy download de_core_news_sm` |
| Analyse ist sehr langsam | Reduziere `VECTORIZER_MAX_FEATURES` oder nutze `--topics 3` |

---

## 📚 Ressourcen & Dokumentation

- **spaCy Docs**: https://spacy.io
- **scikit-learn LDA**: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
- **pyLDAvis Docs**: https://pyldavis.readthedocs.io/
- **Deutsche Stoppwörter**: `resources/stopwords_de.txt` (195 Wörter)

---

## 📝 Lizenz & Autor

**Autor**: Tobias Seekatz  
**Projekt**: Data Analysis (DLBDSEDA02_D)  
**Datum**: Januar 2026  
**Lizenz**: MIT (siehe LICENSE Datei)

---

## 🚀 Weitere Verbesserungen (Roadmap)

- [ ] Gensim LDA für bessere Performance bei großen Datasets
- [ ] Themenveränderung über Zeit analysieren
- [ ] Export in andere Formate (XML, CSV mit Zuordnung)
- [ ] REST-API für Echtzeit-Klassifizierung
- [ ] Mehrsprachigkeit (Englisch, Französisch)
- [ ] Sentiment-Analyse der Beschwerden

---

## 🛠️ Basis-Software-Stack

- **Python 3.8+**
- **pandas**: Datenmanagement
- **spaCy**: Deutsches NLP & Lemmatisierung
- **scikit-learn**: Vektorisierung & LDA
- **matplotlib**: Visualisierung
- **wordcloud**: WordCloud-Erstellung
- **pyLDAvis**: Interaktive Topic-Visualisierung

## 📁 Projekt-Übersicht

Maengelmelder-Konstanz/
├── src/                 # Hauptcode
├── data/raw/            # CSV-Daten
├── output/              # Ergebnisse
├── resources/           # Stoppwörter
├── main.py              # Einstiegspunkt
└── README.md

