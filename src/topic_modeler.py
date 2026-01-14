"""Topic Modeling mit LDA."""
import logging
from typing import List, Tuple, Optional
from .config import Config

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TopicModeler:
    """LDA Topic Modeling."""
    
    def __init__(self, n_topics: int = None):
        self.n_topics = n_topics or Config.LDA_N_TOPICS
        self.lda_model = None
        self.topics = []
    
    def fit(self, count_matrix, feature_names) -> List[Tuple[int, List[str]]]:
        """Führe LDA aus.
        
        Returns:
            Liste von (topic_id, [top_words])
        """
        if not HAS_SKLEARN:
            logger.error("scikit-learn erforderlich!")
            return []
        
        logger.info(f"Starte LDA mit {self.n_topics} Themen...")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=Config.LDA_MAX_ITER,
            random_state=Config.LDA_RANDOM_STATE,
            n_jobs=Config.LDA_N_JOBS
        )
        
        self.lda_model.fit(count_matrix)
        
        # Extrahiere Top-Wörter
        self.topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-Config.LDA_TOP_WORDS:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            self.topics.append((topic_idx + 1, top_words))
        
        logger.info(f"✓ {self.n_topics} Themen extrahiert")
        return self.topics
    
    def get_topics_dict(self) -> dict:
        """Gib Themen als Dictionary zurück."""
        return {
            f"Thema {idx}": words
            for idx, words in self.topics
        }