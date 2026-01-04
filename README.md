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

Siehe [requirements.txt](requirements.txt).

---

**Hinweis:** Alte Dateien im Root-Verzeichnis (`fetch.py`, `preprocess.py`, `topics.py`, `utils.py`, `beispieldaten.csv`) sind veraltet und können gelöscht werden.
