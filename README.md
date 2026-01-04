# Maengelmelder-Konstanz

Beispielprojekt zur Analyse von Bürgerbeschwerden.

Neu: Das Skript wurde in mehrere Module aufgeteilt, um die Wartbarkeit zu verbessern:

- `main.py`      : Orchestriert den Ablauf (Laden → Vorverarbeitung → Themen)
- `fetch.py`     : Funktionen zum Laden von Daten (CSV-Fallback und pandas)
- `preprocess.py`: Stopwörter, Vorverarbeitung und Hilfsfunktionen
- `topics.py`    : Vektorisierung und Themen-/N-Gramm-Analysen
- `utils.py`     : Feature-Flags für optionale Bibliotheken und kleine Helfer

Einfacher Start:

```bash
python3 nlp_analyse.py
```

Hinweis:
- Wenn du `scikit-learn`, `pandas` oder `nltk` installiert hast, nutzt das Programm diese Bibliotheken automatisch für bessere Ergebnisse.
- Die Datei `maengelmelder_konstanz.csv` wird standardmäßig geladen. Du kannst einen anderen Pfad als Argument an `main.py` übergeben.

Beispiel (einen anderen CSV-Pfad verwenden):

```bash
python3 main.py data/meine_daten.csv
```

Konflikt-Hinweis: Diese Repository-Historie wurde mit dem Remote zusammengeführt. Falls du Probleme beim Push/Pull hast, siehe die Git-Anweisungen im Projekt-Root.

Weitere Hinweise zur Modulentrennung findest du in den Dateien selbst.
# M-ngelmelder-Konstanz
DLBDSEDA02_D Data Analysis Aufgabe Nr. 1
<<<<<<< HEAD

## Kurz: Commit-Anleitung für `stopwords_de.txt`

Falls du die alphabetisch sortierte Stopwortliste ins Git-Repository übernehmen möchtest, führe lokal im Projektordner folgende Befehle aus:

```bash
git add stopwords_de.txt
git commit -m "Add alphabetically sorted German stopwords list"
git push
```

Optional: Passe die Commit-Nachricht an deine Konvention an.
=======
# Maengelmelder-Konstanz

Beispielprojekt zur Analyse von Bürgerbeschwerden.

Dieses Repository enthält ein kleines NLP-Toolkit zum Einlesen von CSV-Daten, zur Vorverarbeitung und zur Themenextraktion.

Schnellstart
-----------

```bash
python3 nlp_analyse.py
# oder mit eigenem Pfad und 7 Themen:
python3 main.py data/meine_daten.csv -t 7
```

Paketstruktur (`nlp/`)
----------------------

Die wichtigsten Module:

- `nlp/fetch.py` — CSV-Lader mit Pandas-Fallback (`load_data_from_csv(filepath)`)
- `nlp/preprocess.py` — Stopwort-Lader und `preprocess_text(text, stop_words, stop_phrases)`
- `nlp/topics.py` — Vektorisierung, LDA-Topic-Extraction und N‑Gram-Analyse
- `nlp/utils.py` — Feature-Flags (`HAS_PANDAS`, `HAS_SKLEARN`, ...)

Beispiel (Programmierschnittstelle):

```python
from nlp.fetch import load_data_from_csv
from nlp.preprocess import prepare_stopwords_and_lemmatizer, preprocess_text
from nlp.topics import vectorize_texts, extract_topics_lda

texts = load_data_from_csv('maengelmelder_konstanz.csv')
stop_words, stop_phrases, _ = prepare_stopwords_and_lemmatizer()
clean = [preprocess_text(t, stop_words, stop_phrases) for t in texts]
count_matrix, *_ = vectorize_texts(clean)
topics = extract_topics_lda(count_matrix, None, num_topics=5)
```

Dependencies
------------

Optional (für bessere Ergebnisse): `pandas`, `scikit-learn`, `nltk`, `gensim`. Eine Übersicht ist in `requirements.txt` enthalten.

Continuous Integration (CI)
--------------------------

Ein GitHub Actions Workflow wurde hinzugefügt unter `.github/workflows/ci.yml`. Er installiert Abhängigkeiten aus `requirements.txt` und führt `pytest` für Pull Requests und Pushes auf `main` aus.

Tests
-----

Ein einfacher Test mit `pytest` liegt unter `tests/test_preprocess.py`. Führe sie lokal mit:

```bash
pytest -q
```

Weitere Hinweise
---------------

- Bei Merge-Konflikten (z. B. beim README) haben wir die Datei bereinigt und die Paketstruktur dokumentiert.
- Wenn du möchtest, richte ich noch GitHub Actions für Linting oder Coverage ein.

Linting & Coverage
------------------

Die CI wurde erweitert: der Workflow führt jetzt `flake8` zur statischen Prüfung aus und läuft `pytest` mit Coverage. Die Coverage-XML wird als Artefakt an den CI-Lauf angehängt.

Lokaler Check:

```bash
pip install -r requirements.txt
flake8
pytest --cov=nlp
```
