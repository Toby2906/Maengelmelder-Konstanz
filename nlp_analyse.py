#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP-Analyse von Bürgerbeschwerden für die Stadt Konstanz
Aufgabe: Textanalyse zur Themenestraktion
"""

import re
import sys
import csv
from collections import Counter

# Versuche optionale Bibliotheken zu importieren; falls nicht vorhanden,
# arbeiten wir mit Fallbacks auf der Standardbibliothek.
HAS_PANDAS = False
HAS_NLTK = False
HAS_SKLEARN = False
HAS_GENSIM = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from gensim import corpora
    from gensim.models import LdaModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


def load_data_from_csv(filepath):
    """Laden der Daten aus CSV-Datei."""
    texts = []
    
    if HAS_PANDAS:
        try:
            # Versuche CSV-Format mit Komma als Separator
            df = pd.read_csv(filepath, sep=',', quotechar='"', dtype=str)
            # Spalte mit Index 2 (3. Spalte) enthält die Beschwerdentexte
            if len(df.columns) > 2:
                texts = df.iloc[:, 2].dropna().astype(str).tolist()
            else:
                print(f"Warnung: Datei hat weniger als 3 Spalten")
        except Exception as e:
            print(f"Fehler beim Laden mit pandas: {e}")
            # Fallback auf csv-Modul
            texts = load_data_csv_module(filepath)
    else:
        texts = load_data_csv_module(filepath)
    
    return texts


def load_data_csv_module(filepath):
    """Fallback: Daten mit csv-Modul laden."""
    texts = []
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for i, row in enumerate(reader):
                if len(row) > 2:
                    text = row[2].strip()
                    if text:  # Nur nicht-leere Texte
                        texts.append(text)
    except FileNotFoundError:
        print(f"Fehler: Datei '{filepath}' nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim CSV-Lesen: {e}")
    
    return texts


# 2. Daten laden
print("Lade Daten...")
texts = load_data_from_csv('beispieldaten.csv')
print(f"Geladen: {len(texts)} Texte\n")

# 3. NLTK-Ressourcen und Stopwörter vorbereiten
print("Initialisiere Vorverarbeitung...")

def load_local_stopwords(path='stopwords_de.txt'):
    """Lade lokale Stopwort-Liste aus Datei.

    Rückgabe: (set(single_words), list(phrase_stopwords)) oder None
    - Kommentare (Zeilen, die mit '#') werden ignoriert
    - Normalisiert auf NFC, kleingeschrieben
    - Trennt einwortige Stopwörter und mehrwortige Phrasen
    """
    try:
        import unicodedata
        with open(path, encoding='utf-8') as f:
            raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
        normalized = [unicodedata.normalize('NFC', ln).lower() for ln in raw]
        single = set()
        phrases = []
        for w in normalized:
            if ' ' in w:
                # collapse multiple spaces
                phrases.append(re.sub(r'\s+', ' ', w).strip())
            else:
                single.add(w)
        return single, phrases
    except Exception:
        return None
if HAS_NLTK:
    def ensure_nltk_resources():
        """Stellt sicher, dass benötigte NLTK-Ressourcen vorhanden sind."""
        resources = ['stopwords', 'wordnet', 'punkt']
        for res in resources:
            try:
                nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
            except LookupError:
                try:
                    nltk.download(res, quiet=True)
                except Exception as e:
                    print(f"Warnung: Konnte {res} nicht laden: {e}")

    ensure_nltk_resources()

    # Stopwörter: zuerst lokale Datei prüfen, sonst NLTK verwenden
    local_sw = load_local_stopwords()
    if local_sw is not None:
        stop_words, stop_phrases = local_sw
    else:
        try:
            stop_words = set(stopwords.words('german'))
        except Exception:
            try:
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('german'))
            except Exception:
                stop_words = set()
        stop_phrases = []

    # Lemmatizer
    class _IdentityLemmatizer:
        """Fallback Lemmatizer für Deutsch"""
        def lemmatize(self, w, pos='n'):
            return w

    lemmatizer = _IdentityLemmatizer()
else:
    # Falls vorhanden, lokale Stopwörter laden
    local_sw = load_local_stopwords()
    if local_sw is not None:
        stop_words, stop_phrases = local_sw
    else:
        # Minimaler deutscher Stopwortsatz als Fallback
        stop_words = set([
            "und", "der", "die", "das", "ein", "eine", "in", "auf", "mit", "für",
            "ist", "sind", "bei", "zu", "von", "an", "dem", "den", "des", "denen",
            "aber", "oder", "nicht", "kann", "auch", "so", "wie", "über", "vor"
        ])
        stop_phrases = []


def preprocess_text(text):
    """Vorverarbeitung der Texte, um 'saubere Texte' zu erhalten."""
    if not isinstance(text, str):
        return ""
    import unicodedata
    # 1. Normalisieren + Kleinbuchstaben
    text = unicodedata.normalize('NFC', text).lower()
    # 2. Entferne mehrwortige Stopphrasen vor der Tokenisierung
    try:
        for phrase in stop_phrases:
            # ganze Phrase als Wortgrenze ersetzen
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", ' ', text)
    except Exception:
        pass
    # 3. Entfernung von Sonderzeichen (behalte aber Umlaute)
    text = re.sub(r"[^a-zäöüß\s]", ' ', text)
    # 4. Tokenisierung
    tokens = text.split()
    # 5. Entfernung von Stoppwörtern und zu kurze Wörter
    processed_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(processed_tokens)


print("Vorverarbeite Texte...")
clean_texts = [preprocess_text(text) for text in texts]
clean_texts = [t for t in clean_texts if t.strip()]  # Leere Texte entfernen

print(f"Bereinigt: {len(clean_texts)} Texte\n")

if len(clean_texts) == 0:
    print("Fehler: Keine Texte zum Verarbeiten. Bitte prüfe die Datei 'beispieldaten.csv'")
    sys.exit(1)


# 4. Vektorisierung der Texte (mindestens zwei Techniken)
print("=" * 60)
print("VEKTORISIERUNG DER TEXTE")
print("=" * 60)

count_matrix = None
tfidf_matrix = None
feature_names_count = None
feature_names_tfidf = None

if HAS_SKLEARN:
    # Ansatz 1: CountVectorizer (Bag of Words)
    print("\n1. CountVectorizer (Bag of Words)")
    print("-" * 40)
    try:
        count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
        count_matrix = count_vectorizer.fit_transform(clean_texts)
        feature_names_count = count_vectorizer.get_feature_names_out()
        print(f"   Matrix-Größe: {count_matrix.shape}")
        print(f"   Features: {len(feature_names_count)}")
    except Exception as e:
        print(f"   Fehler: {e}")

    # Ansatz 2: TF-IDF Vectorizer
    print("\n2. TF-IDF Vectorizer")
    print("-" * 40)
    try:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=500)
        tfidf_matrix = tfidf_vectorizer.fit_transform(clean_texts)
        feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
        print(f"   Matrix-Größe: {tfidf_matrix.shape}")
        print(f"   Features: {len(feature_names_tfidf)}")
    except Exception as e:
        print(f"   Fehler: {e}")
else:
    print("\nWARNUNG: scikit-learn nicht installiert — verwende Fallback-Statistiken\n")
    # Einfache Wort-Frequenzen als Fallback
    from collections import Counter
    word_counts = Counter()
    for doc in clean_texts:
        word_counts.update(doc.split())
    feature_names_count = list(dict(word_counts.most_common(100)).keys())


# 5. Themenextraktion (mindestens zwei Techniken)
print("\n" + "=" * 60)
print("THEMENEXTRAKTION")
print("=" * 60)

NUM_TOPICS = 5

## Technik 1: LDA mit scikit-learn
if HAS_SKLEARN and count_matrix is not None:
    print(f"\n1. Latent Dirichlet Allocation (LDA) - {NUM_TOPICS} Themen")
    print("-" * 40)
    try:
        lda_model = LatentDirichletAllocation(
            n_components=NUM_TOPICS,
            max_iter=20,
            random_state=42,
            verbose=0
        )
        lda_model.fit(count_matrix)
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names_count[i] for i in top_indices]
            scores = [f"{feature_names_count[i]}({topic[i]:.2f})" 
                     for i in top_indices]
            print(f"\n   Thema {topic_idx + 1}:")
            print(f"   {', '.join(top_words)}")
    except Exception as e:
        print(f"   Fehler: {e}")
else:
    print("\n1. LDA - nicht verfügbar (scikit-learn erforderlich)")

## Technik 2: N-Gramm-Analyse (Bigramme)
print(f"\n2. N-Gramm-Analyse (Bigramme)")
print("-" * 40)

if HAS_SKLEARN:
    try:
        ngram_vectorizer = CountVectorizer(
            ngram_range=(2, 2),
            max_features=100,
            min_df=2
        )
        ngram_matrix = ngram_vectorizer.fit_transform(clean_texts)
        ngram_sums = ngram_matrix.sum(axis=0).A1
        
        # Top 10 Bigramme
        top_indices = ngram_sums.argsort()[-10:][::-1]
        top_bigrams = [(ngram_vectorizer.get_feature_names_out()[i], int(ngram_sums[i])) 
                       for i in top_indices]
        
        print("\n   Top 10 Bigramme:")
        for i, (bigram, count) in enumerate(top_bigrams, 1):
            print(f"   {i:2d}. '{bigram}' ({count}x)")
    except Exception as e:
        print(f"   Fehler: {e}")
else:
    # Einfacher Bigram-Fallback
    print("\n   Top 10 Bigramme (Fallback):")
    bigram_counts = Counter()
    for doc in clean_texts:
        tokens = doc.split()
        for i in range(len(tokens) - 1):
            bigram_counts[f"{tokens[i]} {tokens[i+1]}"] += 1
    
    for i, (bg, cnt) in enumerate(bigram_counts.most_common(10), 1):
        print(f"   {i:2d}. '{bg}' ({cnt}x)")


# 6. Top-Wörter (Häufigkeitsanalyse)
print(f"\n3. Top-Wörter (Häufigkeitsanalyse)")
print("-" * 40)

word_freq = Counter()
for doc in clean_texts:
    word_freq.update(doc.split())

print("\n   Top 15 Wörter:")
for i, (word, count) in enumerate(word_freq.most_common(15), 1):
    print(f"   {i:2d}. '{word}' ({count}x)")


print("\n" + "=" * 60)
print("ANALYSE ABGESCHLOSSEN")
print("=" * 60)
