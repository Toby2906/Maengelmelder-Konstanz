# Maengelmelder-Konstanz

NLP-Analyse von Bürgerbeschwerden aus Konstanz.

## Projektstruktur

```
Maengelmelder-Konstanz/
├── README.md                         # Dieses Dokument
├── LICENSE                           # Lizenz
├── requirements.txt                  # Python-Abhängigkeiten
├── .gitignore                        # Git-Einstellungen
├── .flake8                           # Linting-Konfiguration
├── .coveragerc                       # Code-Coverage-Konfiguration
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions Workflow
├── data/
│   └── maengelmelder_konstanz.csv    # Eingangsdaten (CSV)
├── nlp/                              # Python-Paket
│   ├── __init__.py
│   ├── fetch.py                      # CSV-Lader
│   ├── preprocess.py                 # Vorverarbeitung
│   ├── topics.py                     # Topic-Extraktion
│   └── utils.py                      # Hilfsfunktionen
├── main.py                           # Einstiegspunkt
├── nlp_analyse.py                    # Legacy-Einstiegspunkt
├── stopwords_de.txt                  # Deutsche Stoppwörter
├── scripts/                          # Zusätzliche Skripte
└── tests/
    ├── __init__.py
    └── test_preprocess.py            # Unit-Tests
```

## Schnellstart

### Installation

```bash
# Virtual Environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Ausführung

```bash
# Standard (verwendet data/maengelmelder_konstanz.csv)
python3 main.py

# Mit Custom-Datenpfad und 7 Themen
python3 main.py path/to/data.csv -t 7

# Legacy-Einstiegspunkt
python3 nlp_analyse.py
```

## Module

| Datei | Funktion |
|-------|----------|
| `nlp/fetch.py` | CSV-Lader mit pandas-Fallback |
| `nlp/preprocess.py` | Stopwörter, Tokenisierung, Lemmatisierung |
| `nlp/topics.py` | Vektorisierung (Count/TF-IDF), LDA, N-Gramme |
| `nlp/utils.py` | Feature-Flags für optionale Libraries |

### Code-Beispiel

```python
from nlp.fetch import load_data_from_csv
from nlp.preprocess import prepare_stopwords_and_lemmatizer, preprocess_text
from nlp.topics import vectorize_texts, extract_topics_lda

texts = load_data_from_csv('data/maengelmelder_konstanz.csv')
stop_words, stop_phrases, _ = prepare_stopwords_and_lemmatizer()
clean = [preprocess_text(t, stop_words, stop_phrases) for t in texts]
count_matrix, *_ = vectorize_texts(clean)
topics = extract_topics_lda(count_matrix, num_topics=5)
```

## Testing & Linting

```bash
# Tests ausführen
pytest -v

# Mit Code-Coverage
pytest --cov=nlp --cov-report=html

# Linting mit flake8
flake8
```

## CI/CD

GitHub Actions Workflow unter [.github/workflows/ci.yml](.github/workflows/ci.yml):
- Installiert Dependencies
- Führt pytest aus
- Lädt Coverage-Report als Artefakt

## Abhängigkeiten

Optional (automatische Erkennung):
- `pandas` — besseres CSV-Handling
- `scikit-learn` — Vektorisierung & LDA
- `nltk` — Tokenisierung & Lemmatisierung
- `gensim` — Alternative Topic-Modellierung

# Maengelmelder-Konstanz

NLP-Analyse von Bürgerbeschwerden aus Konstanz.

**Kurzübersicht**
- Code befindet sich im Package `src/` (früher `nlp/`).
- Stopwörter liegen in `Stoppwörter/stopwords_de.txt`.
- Ergebnisse (Bilder, HTML) werden in `out/` geschrieben.

**Wesentliche Funktionen**
- Vorverarbeitung: Stopwörter, Normalisierung, optional Lemmatizer/Stemming (spaCy oder NLTK Snowball).
- Vektorisierung: Bag-of-Words (CountVectorizer) und TF‑IDF (TfidfVectorizer).
- Themenextraktion: LDA via scikit‑learn.
- N‑Gram Analyse (Bigramme) und Top‑Wort‑Listen.
- Visualisierungen: `wordcloud.png`, `top_words.png` und pyLDAvis HTML (`pyldavis_lda.html`) in `out/`.

## Projektstruktur (kurz)

```
Maengelmelder-Konstanz/
├── README.md
├── requirements.txt
├── data/                 # Eingabedaten (CSV)
├── Stoppwörter/          # Deutsche Stopwörter (stopwords_de.txt)
├── src/                  # Hauptpaket (ersetzt früheres `nlp/`)
│   ├── fetch.py
│   ├── preprocess.py
│   ├── topics.py
│   ├── utils.py
│   └── main.py           # Orchestrator: `run()` und CLI
├── out/                  # Ausgabe (Visualisierungen)
└── .github/              # CI/Workflows
```

## Schnellstart

### 1) Virtual Environment & Installation

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\\Scripts\\activate     # Windows
python -m pip install -r requirements.txt
```

Hinweis: `spaCy` ist in `requirements.txt` enthalten, das deutsche Modell `de_core_news_sm` muss separat installiert werden, falls du spaCy-Lemmatisierung nutzen willst:

```bash
python -m spacy download de_core_news_sm
```

### 2) Ausführen

Standardlauf (nutzt `data/maengelmelder_konstanz.csv`):

```bash
python -m src.main
```

Mit Optionen:

```bash
python -m src.main data/maengelmelder_konstanz.csv -t 5 -o out
```

Nach erfolgreichem Lauf findest du die Visualisierungen in `out/`:
- `wordcloud.png`
- `top_words.png`
- `pyldavis_lda.html` (falls `pyLDAvis` und LDA verfügbar sind)

## Hinweise zu Abhängigkeiten
- `pandas`: optional, für bequemeres CSV-Handling
- `scikit-learn`: für Vektorisierung und LDA
- `nltk`: Fallback für Stemming/Lemmatisierung (Snowball)
- `spacy`: bevorzugte Lemmatisierung (Empfohlen: `de_core_news_sm`)
- `matplotlib`, `wordcloud`, `pyLDAvis`: Visualisierungen

## Entfernte / veraltete Dateien
- Das alte `nlp/` Package wurde in `src/` migriert.
- Legacy‑Einstiegspunkte (`nlp_analyse.py`) und einige Root‑Skripte wurden entfernt oder ersetzt.

## Entwicklung
- Tests: `pytest`
- Linting: `flake8`

---

Wenn du möchtest, erstelle ich noch einen kurzen `CONTRIBUTING.md` mit Ablauf für lokale Tests und Pull Requests.

**Notebook zur Anzeige**

- In `out/` wurde ein Notebook `view_pyldavis.ipynb` hinzugefügt, das mehrere Möglichkeiten zeigt, die pyLDAvis-HTML (`out/pyldavis_lda.html`) im Notebook einzubetten oder per IFrame anzuzeigen.
- Öffnen: `jupyter lab out/view_pyldavis.ipynb` oder in VS Code die Datei öffnen.
- Die Zellen demonstrieren: direkte Einbettung mit `IPython.display.HTML`, IFrame-Anzeige über einen lokalen HTTP-Server und ein interaktives `ipywidgets.HTML`-Beispiel.

**pyLDAvis Anzeige**

- Nach einem erfolgreichen Lauf (`python -m src.main`) findest du die interaktive Visualisierung in `out/pyldavis_lda.html`.
- Zum lokalen Betrachten empfehle ich:
```
python -m http.server 8000 -d out
# dann: http://localhost:8000/pyldavis_lda.html
```

**Hinweis zu Logs**
- Temporäre Laufprotokolle wie `run.log` wurden entfernt, damit das Repository sauber bleibt.
