"""Vektorisierung und Themen-/N-Gramm-Analysen mit Visualisierungen.

Erweitert um matplotlib, wordcloud und pyLDAvis Visualisierungen.
Ergebnisse werden im out-Ordner gespeichert.
"""
import os
import logging
from collections import Counter
from typing import List, Tuple, Optional
from .utils import HAS_SKLEARN

if HAS_SKLEARN:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

# Konfiguration für Vektorisierung
VECTORIZER_CONFIG = {
    'max_df': 0.95,        # Ignoriere Wörter die in >95% der Dokumente vorkommen
    'min_df': 2,           # Ignoriere Wörter die in <2 Dokumenten vorkommen
    'max_features': 500,   # Maximale Anzahl Features
}


def vectorize_texts(clean_texts: List[str]):
    """Vektorisiere Texte mit CountVectorizer und TF-IDF.
    
    Args:
        clean_texts: Liste von vorverarbeiteten Texten
    
    Returns:
        Tuple: (count_matrix, tfidf_matrix, feature_names_count,
                feature_names_tfidf, count_vectorizer, tfidf_vectorizer)
    """
    count_matrix = None
    tfidf_matrix = None
    feature_names_count = None
    feature_names_tfidf = None
    count_vectorizer = None
    tfidf_vectorizer = None

    if HAS_SKLEARN:
        # CountVectorizer (Bag of Words)
        try:
            count_vectorizer = CountVectorizer(**VECTORIZER_CONFIG)
            count_matrix = count_vectorizer.fit_transform(clean_texts)
            feature_names_count = count_vectorizer.get_feature_names_out()
            logger.info(f"CountVectorizer: {count_matrix.shape[0]} Dokumente, "
                       f"{count_matrix.shape[1]} Features")
        except Exception as e:
            logger.error(f"CountVectorizer fehlgeschlagen: {e}")
            count_matrix = None

        # TF-IDF Vectorizer
        try:
            tfidf_vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
            tfidf_matrix = tfidf_vectorizer.fit_transform(clean_texts)
            feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
            logger.info(f"TF-IDF: {tfidf_matrix.shape[0]} Dokumente, "
                       f"{tfidf_matrix.shape[1]} Features")
        except Exception as e:
            logger.error(f"TF-IDF Vektorisierung fehlgeschlagen: {e}")
            tfidf_matrix = None
    else:
        logger.warning("scikit-learn nicht verfügbar - Fallback auf einfache Frequenzen")
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


def extract_topics_lda(
    count_matrix,
    count_vectorizer,
    feature_names_count,
    num_topics: int = 5,
    max_iter: int = 20,
    random_state: int = 42
) -> Tuple[List[Tuple[int, List[str]]], Optional[LatentDirichletAllocation]]:
    """Führe Latent Dirichlet Allocation (LDA) aus.
    
    Args:
        count_matrix: Scipy sparse matrix von CountVectorizer
        count_vectorizer: Fitted CountVectorizer
        feature_names_count: Feature names array
        num_topics: Anzahl zu extrahierender Themen
        max_iter: Maximale Iterationen für LDA
        random_state: Random seed für Reproduzierbarkeit
    
    Returns:
        Tuple: (topics, lda_model)
            topics: Liste von (topic_idx, [top_words])
            lda_model: Fitted LDA model oder None
    """
    if not HAS_SKLEARN or count_matrix is None:
        logger.warning('LDA nicht verfügbar (scikit-learn erforderlich oder keine Count-Matrix)')
        return [], None
    
    try:
        logger.info(f"Starte LDA mit {num_topics} Themen...")
        
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1  # Nutze alle CPU-Kerne
        )
        
        lda_model.fit(count_matrix)
        
        # Extrahiere Top-Wörter pro Thema
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            # Top 10 Wörter pro Thema
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names_count[i] for i in top_indices]
            topics.append((topic_idx + 1, top_words))
        
        logger.info(f"✓ LDA erfolgreich: {num_topics} Themen extrahiert")
        return topics, lda_model
        
    except Exception as e:
        logger.error(f"Fehler bei LDA: {e}", exc_info=True)
        return [], None


def extract_ngrams(
    clean_texts: List[str],
    top_n: int = 10,
    ngram_range: Tuple[int, int] = (2, 2)
) -> List[Tuple[str, int]]:
    """Extrahiere häufigste N-Gramme aus Texten.
    
    Args:
        clean_texts: Liste von vorverarbeiteten Texten
        top_n: Anzahl der häufigsten N-Gramme
        ngram_range: Tuple (min_n, max_n) für N-Gramm-Größe
    
    Returns:
        Liste von (ngram, count) Tuples
    """
    if HAS_SKLEARN:
        try:
            vec = CountVectorizer(
                ngram_range=ngram_range,
                max_features=100,
                min_df=2
            )
            mat = vec.fit_transform(clean_texts)
            sums = mat.sum(axis=0).A1
            top_indices = sums.argsort()[-top_n:][::-1]
            top_ngrams = [
                (vec.get_feature_names_out()[i], int(sums[i]))
                for i in top_indices
            ]
            return top_ngrams
        except Exception as e:
            logger.warning(f"N-Gramm-Extraktion mit sklearn fehlgeschlagen: {e}")
    
    # Fallback: Manuelle Bigramm-Extraktion
    logger.info("Fallback auf manuelle N-Gramm-Extraktion")
    ngram_counts = Counter()
    
    for doc in clean_texts:
        tokens = doc.split()
        
        # Generiere N-Gramme
        min_n, max_n = ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngram_counts[ngram] += 1
    
    return ngram_counts.most_common(top_n)


def top_words(clean_texts: List[str], top_n: int = 15) -> List[Tuple[str, int]]:
    """Extrahiere häufigste Einzelwörter.
    
    Args:
        clean_texts: Liste von vorverarbeiteten Texten
        top_n: Anzahl der häufigsten Wörter
    
    Returns:
        Liste von (word, count) Tuples
    """
    freq = Counter()
    for doc in clean_texts:
        freq.update(doc.split())
    return freq.most_common(top_n)


def visualize_and_save(
    clean_texts: List[str],
    count_vectorizer,
    count_matrix,
    lda_model,
    out_dir: str = 'out'
) -> None:
    """Erzeuge und speichere Visualisierungen.
    
    Erstellt:
    - wordcloud.png: Word Cloud der häufigsten Wörter
    - top_words.png: Balkendiagramm der Top-Wörter
    - pyldavis_lda.html: Interaktive LDA-Visualisierung (falls verfügbar)
    
    Args:
        clean_texts: Liste von vorverarbeiteten Texten
        count_vectorizer: Fitted CountVectorizer
        count_matrix: Document-term matrix
        lda_model: Fitted LDA model
        out_dir: Ausgabe-Verzeichnis
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Erstelle Visualisierungen in: {out_dir}")

    # 1. Word Cloud
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        wc_text = " ".join(clean_texts)
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate(wc_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Häufigste Wörter', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = os.path.join(out_dir, 'wordcloud.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Word Cloud gespeichert: {output_path}")
        
    except ImportError:
        logger.warning("wordcloud nicht installiert. Überspringe Word Cloud. "
                      "Installiere mit: pip install wordcloud")
    except Exception as e:
        logger.error(f"Word Cloud Fehler: {e}", exc_info=True)

    # 2. Top-Wörter Balkendiagramm
    try:
        import matplotlib.pyplot as plt
        
        top = top_words(clean_texts, top_n=20)
        words = [w for w, c in top][::-1]  # Reverse für bessere Darstellung
        counts = [c for w, c in top][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(words, counts, color='steelblue')
        plt.xlabel('Häufigkeit', fontsize=12)
        plt.title('Top 20 Wörter', fontsize=14, pad=15)
        plt.tight_layout()
        
        output_path = os.path.join(out_dir, 'top_words.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Top-Wörter Diagramm gespeichert: {output_path}")
        
    except ImportError:
        logger.warning("matplotlib nicht installiert. Überspringe Diagramme. "
                      "Installiere mit: pip install matplotlib")
    except Exception as e:
        logger.error(f"Top-Wörter Plot Fehler: {e}", exc_info=True)

    # 3. pyLDAvis Interaktive Visualisierung
    if lda_model is not None and count_vectorizer is not None:
        try:
            import pyLDAvis
            
            # Versuche sklearn-spezifischen Wrapper
            try:
                import pyLDAvis.sklearn as sklearn_vis
                panel = sklearn_vis.prepare(
                    lda_model,
                    count_matrix,
                    count_vectorizer,
                    mds='tsne'
                )
                logger.info("Verwende pyLDAvis.sklearn")
                
            except (ImportError, AttributeError):
                # Fallback: Manuelle Vorbereitung
                logger.info("Fallback auf manuelles pyLDAvis.prepare")
                
                try:
                    import numpy as np
                    from scipy import sparse
                    
                    # Topic-term distributions (n_topics x n_terms)
                    topic_term_dists = lda_model.components_
                    topic_term_dists = topic_term_dists / topic_term_dists.sum(axis=1)[:, np.newaxis]
                    
                    # Document-topic distributions (n_docs x n_topics)
                    doc_topic_dists = lda_model.transform(count_matrix)
                    
                    # Document lengths and term frequencies
                    if sparse.issparse(count_matrix):
                        doc_lengths = np.array(count_matrix.sum(axis=1)).ravel()
                        term_frequency = np.array(count_matrix.sum(axis=0)).ravel()
                    else:
                        doc_lengths = count_matrix.sum(axis=1)
                        term_frequency = count_matrix.sum(axis=0)
                    
                    # Vocabulary
                    vocab = count_vectorizer.get_feature_names_out()
                    
                    panel = pyLDAvis.prepare(
                        topic_term_dists,
                        doc_topic_dists,
                        doc_lengths,
                        vocab,
                        term_frequency,
                        mds='tsne'
                    )
                    
                except Exception as prep_error:
                    logger.error(f"Manuelle pyLDAvis Vorbereitung fehlgeschlagen: {prep_error}")
                    raise
            
            # Speichere HTML
            output_path = os.path.join(out_dir, 'pyldavis_lda.html')
            pyLDAvis.save_html(panel, output_path)
            logger.info(f"✓ pyLDAvis Visualisierung gespeichert: {output_path}")
            logger.info("  Öffne mit: python -m http.server 8000 -d out")
            
        except ImportError:
            logger.warning("pyLDAvis nicht installiert. Überspringe interaktive Visualisierung. "
                          "Installiere mit: pip install pyldavis")
        except Exception as e:
            logger.error(f"pyLDAvis Fehler: {e}", exc_info=True)
    else:
        logger.info("Überspringe pyLDAvis (kein LDA-Modell verfügbar)")