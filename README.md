# Mängelmelder Konstanz - NLP-Analyse mit Coherence Score


**Projekt: Data Analysis (DLBDSEDA02_D)**  
**Autor: Tobias Seekatz**

Automatisierte Identifikation von Problemthemen aus Bürgerbeschwerden mit **automatischer Topic-Anzahl Optimierung** mittels Coherence Score.

## 🎯 Zielsetzung

Analyse unstrukturierter Bürgerbeschwerden aus dem "Mängelmelder Konstanz" zur automatisierten Themenidentifikation. **Neu:** Automatische Bestimmung der optimalen Anzahl von Topics (k) durch Coherence Score (c_v metric).

## ✨ Features

### Kern-Funktionalität
- **Datenvorverarbeitung**: Normalisierung, Tokenisierung, Lemmatisierung (spaCy)
- **Vektorisierung**: CountVectorizer (BoW) und TF-IDF
- **Topic Modeling**: Latent Dirichlet Allocation (LDA)
- **N-Gramm-Analyse**: Bigramme für kontextuelle Wortpaare
- **Visualisierungen**: WordCloud, Plots, pyLDAvis

### 🆕 Neu: Coherence Score Optimierung
- **Automatische k-Optimierung**: Findet optimale Topic-Anzahl
- **Coherence Score (c_v)**: Evaluiert Themen-Kohärenz mit gensim
- **Visualisierung**: Coherence Score über verschiedene k-Werte
- **Empfehlungssystem**: Interpretation der Coherence-Qualität

## 🚀 Schnellstart

### Installation

```bash
# 1. Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# 2. Dependencies
pip install -r requirements.txt

# 3. spaCy Modell
python -m spacy download de_core_news_sm

# 4. NLTK Daten (optional)
python -c "import nltk; nltk.download('stopwords')"
```

### Daten vorbereiten

```bash
# Kopiere deine CSV-Datei nach:
# data/raw/maengelmelder_konstanz.csv
```

### Ausführung

#### Variante 1: Automatische Topic-Optimierung (Empfohlen)

```bash
python main.py --auto-optimize
```

**Was passiert:**
1. Testet verschiedene Topic-Anzahlen (k=2 bis k=15)
2. Berechnet Coherence Score für jedes k
3. Wählt optimales k automatisch
4. Führt vollständige Analyse durch
5. Speichert Coherence-Plot

#### Variante 2: Nur Optimierung (ohne vollständige Analyse)

```bash
python main.py --optimize-topics
```

**Empfohlen für:**
- Erste Exploration der Daten
- Schnelle k-Bestimmung
- Vergleich verschiedener Datensätze

#### Variante 3: Manuelle Topic-Anzahl

```bash
python main.py --topics 5
```

**Nutze wenn:**
- Du die optimale Anzahl bereits kennst
- Zeitersparnis wichtig ist
- Reproduzierbare Ergebnisse benötigt werden

#### Variante 4: Custom Optimierungs-Bereich

```bash
python main.py --auto-optimize --min-topics 3 --max-topics 12
```

### Alle Optionen

```bash
# Hilfe anzeigen
python main.py --help

# Mit eigener CSV
python main.py --input data/raw/meine_daten.csv --auto-optimize

# Custom Range + Verbose
python main.py --auto-optimize --min-topics 2 --max-topics 20 --verbose
```

## 📊 Verwendungsbeispiele

### Beispiel 1: Standard-Workflow

```bash
# Automatische Optimierung
python main.py --auto-optimize

# Ausgabe:
# Optimale Topic-Anzahl: k=7
# Coherence Score: 0.4523
# (Gute Themen-Kohärenz)
```

### Beispiel 2: Vergleich verschiedener k-Werte

```bash
# Teste breiten Bereich
python main.py --optimize-topics --min-topics 2 --max-topics 20

# Ergebnis:
# coherence_scores.png zeigt Kurve
# topic_optimization.json enthält alle Scores
```

### Beispiel 3: Reproduzierbare Analyse

```bash
# Verwende vorher ermitteltes optimales k
python main.py --topics 7

## 📈 Coherence Score Interpretation

Der Coherence Score (c_v metric) misst die semantische Kohärenz von Topics:

| Score | Qualität | Bedeutung |
|-------|----------|-----------|
| **> 0.5** | Exzellent | Sehr kohärente, interpretierbare Topics |
| **0.4 - 0.5** | Gut | Klare Themen, gute Interpretierbarkeit |
| **0.3 - 0.4** | Akzeptabel | Brauchbare Themen, teilweise überlappend |
| **< 0.3** | Niedrig | Schwache Kohärenz, mehr Daten nötig |

**Faktoren die Coherence beeinflussen:**
- Datenmenge (mehr Texte = bessere Scores)
- Datenqualität (saubere Vorverarbeitung wichtig)
- Domänen-Homogenität (ähnliche Themen = höhere Scores)
- k-Wert (zu viele/wenige Topics senken Score)

## 📁 Output-Struktur

Nach der Analyse:

```
output/
├── visualizations/
│   ├── coherence_scores.png      # NEU: k-Optimierung
│   ├── wordcloud.png
│   ├── top_words.png
│   └── top_bigrams.png
├── reports/
│   └── lda_interactive.html      # pyLDAvis
└── analysis/
    ├── analysis_results.json     # Haupt-Ergebnisse
    └── topic_optimization.json   # NEU: Coherence Scores
```

## 🛠️ Software-Stack

### Core
- **Python 3.8+**
- **pandas**: Datenmanagement
- **spaCy**: Deutsches NLP & Lemmatisierung
- **scikit-learn**: Vektorisierung & LDA

### Neu: Topic-Optimierung
- **gensim**: Coherence Score Berechnung
- **gensim.models.CoherenceModel**: c_v metric

### Visualisierung
- **matplotlib**: Plots & Diagramme
- **wordcloud**: WordCloud-Erstellung
- **pyLDAvis**: Interaktive Topic-Visualisierung

## 📖 Methodologie

### 1. Datenvorverarbeitung
- Unicode-Normalisierung (NFC)
- Lowercase-Transformation
- Stoppwort-Entfernung (erweiterte deutsche Liste)
- Lemmatisierung (spaCy de_core_news_sm)
- Tokenfilterung (Länge, Sonderzeichen)

### 2. Vektorisierung
- **Bag-of-Words (BoW)**: CountVectorizer
  - max_df=0.95 (max. 95% Dokument-Frequenz)
  - min_df=2 (min. 2 Dokumente)
  - max_features=500
- **TF-IDF**: Term Frequency-Inverse Document Frequency

### 3. Topic Modeling
- **Algorithmus**: Latent Dirichlet Allocation (LDA)
- **Optimierung**: Coherence Score (c_v metric)
- **Parameter**: 
  - n_components: 2-15 (auto-optimiert)
  - max_iter: 20
  - random_state: 42 (Reproduzierbarkeit)

### 4. Evaluation
- **Coherence Score (c_v)**: Semantische Kohärenz
- **Perplexity**: Alternative Metrik (Fallback)
- **N-Gramm-Analyse**: Kontextuelle Validierung

## 🔬 Wissenschaftlicher Kontext

**Coherence Score nach Röder et al. (2015):**
- Misst menschliche Interpretierbarkeit von Topics
- c_v metric korreliert am besten mit menschlichem Urteil
- Basiert auf word co-occurrence patterns

**Zitat:**
> "Topic coherence measures provide an automatic evaluation of topic quality based on the semantic similarity between high-scoring words in topics."

**Literatur:**
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. WSDM '15.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR.

## 🧪 Testing

```bash
# Installation mit Dev-Dependencies
pip install -r requirements-dev.txt

# Tests ausführen
pytest

# Mit Coverage
pytest --cov=src --cov-report=html
```

## 📊 Performance

**Laufzeit-Vergleich** (1000 Texte):

| Modus | Laufzeit | k-Werte getestet |
|-------|----------|------------------|
| Manuell (k=5) | ~10 Sekunden | 1 |
| Auto-Optimize (k=2-15) | ~2-3 Minuten | 14 |
| Optimize-Only | ~1-2 Minuten | 14 |

**Empfehlung:**
- Erste Analyse: `--optimize-topics` (schnell, nur k-Bestimmung)
- Produktion: `--topics K` mit ermitteltem k (schnell, reproduzierbar)
- Exploration: `--auto-optimize` (komplett, inkl. k-Optimierung)
