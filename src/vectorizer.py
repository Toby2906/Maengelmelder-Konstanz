"""Vektorisierung mit CountVectorizer und TF-IDF."""
import logging
from typing import Tuple, Optional, List
from .config import Config

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.error("scikit-learn nicht verfügbar!")


class TextVectorizer:
    """Vektorisiert Texte mit CountVectorizer und TF-IDF."""
    
    def __init__(self):
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.count_matrix = None
        self.tfidf_matrix = None
        self.feature_names_count = None
        self.feature_names_tfidf = None
    
    def fit_transform(self, texts: List[str]) -> Tuple:
        """Vektorisiere Texte.
        
        Returns:
            (count_matrix, tfidf_matrix, features_count, features_tfidf,
             count_vectorizer, tfidf_vectorizer)
        """
        if not HAS_SKLEARN:
            logger.error("scikit-learn erforderlich!")
            return None, None, None, None, None, None
        
        logger.info("Vektorisiere Texte...")
        
        # CountVectorizer (Bag of Words)
        self.count_vectorizer = CountVectorizer(
            max_features=Config.VECTORIZER_MAX_FEATURES,
            max_df=Config.VECTORIZER_MAX_DF,
            min_df=Config.VECTORIZER_MIN_DF,
            ngram_range=Config.VECTORIZER_NGRAM_RANGE
        )
        
        self.count_matrix = self.count_vectorizer.fit_transform(texts)
        self.feature_names_count = self.count_vectorizer.get_feature_names_out()
        
        logger.info(f"✓ CountVectorizer: {self.count_matrix.shape}")
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.VECTORIZER_MAX_FEATURES,
            max_df=Config.VECTORIZER_MAX_DF,
            min_df=Config.VECTORIZER_MIN_DF,
            ngram_range=Config.VECTORIZER_NGRAM_RANGE
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names_tfidf = self.tfidf_vectorizer.get_feature_names_out()
        
        logger.info(f"✓ TF-IDF: {self.tfidf_matrix.shape}")
        
        return (
            self.count_matrix,
            self.tfidf_matrix,
            self.feature_names_count,
            self.feature_names_tfidf,
            self.count_vectorizer,
            self.tfidf_vectorizer
        )