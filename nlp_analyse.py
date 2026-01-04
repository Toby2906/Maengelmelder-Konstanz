#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy entrypoint — ruft `main.run()` auf."""
from main import run


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'maengelmelder_konstanz.csv'
    sys.exit(run(path))
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
    print("Fehler: Keine Texte zum Verarbeiten. Bitte prüfe die Datei 'maengelmelder_konstanz.csv'")
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
