"""Vektorisierung und Themen-/N-Gramm-Analysen.

Erweitert um einfache Visualisierungen via matplotlib, wordcloud und pyLDAvis
wenn diese verfügbar sind. Ergebnisse werden in den `out`-Ordner geschrieben.
"""
import os
from collections import Counter
from typing import List, Tuple, Optional
from .utils import HAS_SKLEARN

if HAS_SKLEARN:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation


def vectorize_texts(clean_texts: List[str]):
    count_matrix = None
    tfidf_matrix = None
    feature_names_count = None
    feature_names_tfidf = None
    count_vectorizer = None
    tfidf_vectorizer = None

    if HAS_SKLEARN:
        try:
            count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
            count_matrix = count_vectorizer.fit_transform(clean_texts)
            feature_names_count = count_vectorizer.get_feature_names_out()
        except Exception:
            count_matrix = None

        try:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=500)
            tfidf_matrix = tfidf_vectorizer.fit_transform(clean_texts)
            feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
        except Exception:
            tfidf_matrix = None
    else:
        # Fallback: einfache Wortfrequenzen
        word_counts = Counter()
        for doc in clean_texts:
            word_counts.update(doc.split())
        feature_names_count = list(dict(word_counts.most_common(100)).keys())

    return (
        count_matrix,
        tfidf_matrix,
        feature_names_count,
        feature_names_tfidf,
        count_vectorizer,
        tfidf_vectorizer,
    )


def extract_topics_lda(count_matrix, count_vectorizer, feature_names_count, num_topics=5):
    """Führt LDA aus und gibt (topics, lda_model) zurück.

    `topics` ist eine Liste von (idx, [words]).
    """
    if not HAS_SKLEARN or count_matrix is None:
        print('LDA nicht verfügbar (scikit-learn erforderlich oder keine Count-Matrix)')
        return [], None
    try:
        lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=20, random_state=42)
        lda_model.fit(count_matrix)
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names_count[i] for i in top_indices]
            topics.append((topic_idx + 1, top_words))
        return topics, lda_model
    except Exception as e:
        print(f"Fehler bei LDA: {e}")
        return [], None


def extract_ngrams(clean_texts: List[str], top_n=10):
    if HAS_SKLEARN:
        try:
            vec = CountVectorizer(ngram_range=(2, 2), max_features=100, min_df=2)
            mat = vec.fit_transform(clean_texts)
            sums = mat.sum(axis=0).A1
            top_indices = sums.argsort()[-top_n:][::-1]
            top_bigrams = [(vec.get_feature_names_out()[i], int(sums[i])) for i in top_indices]
            return top_bigrams
        except Exception:
            pass
    # Fallback
    bigram_counts = Counter()
    for doc in clean_texts:
        tokens = doc.split()
        for i in range(len(tokens) - 1):
            bigram_counts[f"{tokens[i]} {tokens[i+1]}"] += 1
    return bigram_counts.most_common(top_n)


def top_words(clean_texts: List[str], top_n=15):
    freq = Counter()
    for doc in clean_texts:
        freq.update(doc.split())
    return freq.most_common(top_n)


def visualize_and_save(clean_texts: List[str], count_vectorizer, count_matrix, lda_model, out_dir: str = 'out'):
    """Erzeuge und speichere Visualisierungen in `out_dir`.

    - Wordcloud (`wordcloud.png`)
    - Top-Wörter Balkendiagramm (`top_words.png`)
    - pyLDAvis HTML (`pyldavis_lda.html`) falls möglich
    """
    os.makedirs(out_dir, exist_ok=True)

    # Wordcloud
    try:
        from wordcloud import WordCloud
        wc_text = " ".join(clean_texts)
        wc = WordCloud(width=1200, height=600, background_color='white').generate(wc_text)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'wordcloud.png'), dpi=150)
        plt.close()
    except Exception:
        pass

    # Top-Wörter Balkendiagramm
    try:
        top = top_words(clean_texts, top_n=20)
        words = [w for w, c in top][::-1]
        counts = [c for w, c in top][::-1]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.barh(words, counts)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'top_words.png'), dpi=150)
        plt.close()
    except Exception:
        pass

    # pyLDAvis (falls vorhanden und ein LDA-Model verfügbar ist)
    if lda_model is not None and count_vectorizer is not None:
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
            panel = pyLDAvis.sklearn.prepare(lda_model, count_matrix, count_vectorizer)
            pyLDAvis.save_html(panel, os.path.join(out_dir, 'pyldavis_lda.html'))
        except Exception:
            pass
