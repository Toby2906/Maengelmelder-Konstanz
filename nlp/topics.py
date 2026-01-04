"""Vektorisierung und Themen-/N-Gramm-Analysen."""
from collections import Counter
from typing import List, Tuple
from nlp.utils import HAS_SKLEARN

if HAS_SKLEARN:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation


def vectorize_texts(clean_texts: List[str]):
    count_matrix = None
    tfidf_matrix = None
    feature_names_count = None
    feature_names_tfidf = None

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

    return count_matrix, tfidf_matrix, feature_names_count, feature_names_tfidf


def extract_topics_lda(count_matrix, feature_names_count, num_topics=5):
    if not HAS_SKLEARN or count_matrix is None:
        print('LDA nicht verfügbar (scikit-learn erforderlich)')
        return
    try:
        lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=20, random_state=42)
        lda_model.fit(count_matrix)
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names_count[i] for i in top_indices]
            topics.append((topic_idx + 1, top_words))
        return topics
    except Exception as e:
        print(f"Fehler bei LDA: {e}")
        return


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
