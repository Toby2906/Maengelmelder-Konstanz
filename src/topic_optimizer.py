"""Topic-Anzahl Optimierung mit Coherence Score.

Findet automatisch die optimale Anzahl von Topics (k) für LDA
mittels Coherence Score (c_v metric).
"""
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

# Prüfe Verfügbarkeit
try:
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.error("scikit-learn nicht verfügbar!")

try:
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    logger.warning("gensim nicht verfügbar - Coherence Score nicht möglich")


class TopicOptimizer:
    """Optimiere Topic-Anzahl mit Coherence Score."""
    
    def __init__(
        self,
        min_topics: int = 2,
        max_topics: int = 15,
        step: int = 1
    ):
        """Initialisiere Optimizer.
        
        Args:
            min_topics: Minimale Anzahl Topics
            max_topics: Maximale Anzahl Topics
            step: Schrittgröße
        """
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.step = step
        self.coherence_scores: Dict[int, float] = {}
        self.models: Dict[int, LatentDirichletAllocation] = {}
        self.optimal_k: Optional[int] = None
        self.optimal_score: Optional[float] = None
    
    def optimize(
        self,
        texts: List[str],
        count_matrix,
        count_vectorizer
    ) -> Tuple[int, Dict[int, float]]:
        """Finde optimale Topic-Anzahl.
        
        Args:
            texts: Liste von vorverarbeiteten Texten
            count_matrix: Document-term matrix
            count_vectorizer: Fitted CountVectorizer
        
        Returns:
            (optimal_k, coherence_scores_dict)
        """
        if not HAS_SKLEARN:
            logger.error("scikit-learn erforderlich!")
            return Config.LDA_N_TOPICS, {}
        
        logger.info(
            f"Starte Topic-Optimierung: k={self.min_topics}..{self.max_topics}"
        )
        
        # Tokenisiere Texte für Gensim
        tokenized_texts = [text.split() for text in texts]
        
        # Erstelle Gensim Dictionary
        if HAS_GENSIM:
            dictionary = Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Teste verschiedene k-Werte
        k_values = range(self.min_topics, self.max_topics + 1, self.step)
        
        for k in k_values:
            logger.info(f"  Teste k={k}...")
            
            # Trainiere LDA-Modell
            lda_model = LatentDirichletAllocation(
                n_components=k,
                max_iter=Config.LDA_MAX_ITER,
                random_state=Config.LDA_RANDOM_STATE,
                n_jobs=Config.LDA_N_JOBS
            )
            
            lda_model.fit(count_matrix)
            self.models[k] = lda_model
            
            # Berechne Coherence Score
            if HAS_GENSIM:
                coherence = self._calculate_coherence_gensim(
                    lda_model,
                    tokenized_texts,
                    dictionary,
                    corpus
                )
            else:
                # Fallback: Perplexity-basierte Metrik
                coherence = -lda_model.perplexity(count_matrix)
            
            self.coherence_scores[k] = coherence
            logger.info(f"    → Coherence Score: {coherence:.4f}")
        
        # Finde Optimum
        self.optimal_k = max(self.coherence_scores, key=self.coherence_scores.get)
        self.optimal_score = self.coherence_scores[self.optimal_k]
        
        logger.info("")
        logger.info(f"✓ Optimale Topic-Anzahl: k={self.optimal_k}")
        logger.info(f"  Coherence Score: {self.optimal_score:.4f}")
        
        return self.optimal_k, self.coherence_scores
    
    def _calculate_coherence_gensim(
        self,
        lda_model,
        tokenized_texts: List[List[str]],
        dictionary,
        corpus
    ) -> float:
        """Berechne Coherence Score mit Gensim.
        
        Args:
            lda_model: Trainiertes LDA-Modell
            tokenized_texts: Tokenisierte Texte
            dictionary: Gensim Dictionary
            corpus: Gensim Corpus
        
        Returns:
            Coherence Score (c_v metric)
        """
        try:
            # Extrahiere Topics aus sklearn LDA
            feature_names = list(dictionary.token2id.keys())
            topics = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_indices = topic.argsort()[-10:][::-1]
                # Map sklearn indices zu Gensim tokens
                top_words = []
                for idx in top_indices:
                    if idx < len(feature_names):
                        top_words.append(feature_names[idx])
                topics.append(top_words)
            
            # Berechne Coherence
            coherence_model = CoherenceModel(
                topics=topics,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            return coherence_model.get_coherence()
            
        except Exception as e:
            logger.warning(f"Gensim Coherence fehlgeschlagen: {e}")
            # Fallback auf Perplexity
            return -lda_model.perplexity(corpus) if hasattr(lda_model, 'perplexity') else 0.0
    
    def get_optimal_model(self):
        """Gib optimales LDA-Modell zurück.
        
        Returns:
            LDA-Modell mit optimaler Topic-Anzahl
        """
        if self.optimal_k and self.optimal_k in self.models:
            return self.models[self.optimal_k]
        return None
    
    def plot_coherence_scores(self, save_path: str = None):
        """Plotte Coherence Scores.
        
        Args:
            save_path: Pfad zum Speichern (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            k_values = sorted(self.coherence_scores.keys())
            scores = [self.coherence_scores[k] for k in k_values]
            
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, scores, marker='o', linewidth=2, markersize=8)
            
            # Markiere Optimum
            if self.optimal_k:
                plt.axvline(
                    x=self.optimal_k,
                    color='red',
                    linestyle='--',
                    label=f'Optimal k={self.optimal_k}'
                )
                plt.plot(
                    self.optimal_k,
                    self.optimal_score,
                    'r*',
                    markersize=15
                )
            
            plt.xlabel('Anzahl Topics (k)', fontsize=12)
            plt.ylabel('Coherence Score', fontsize=12)
            plt.title('Topic-Anzahl Optimierung: Coherence Score', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
                logger.info(f"✓ Coherence Plot gespeichert: {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib nicht verfügbar - Plot übersprungen")
        except Exception as e:
            logger.error(f"Fehler beim Plotten: {e}")
    
    def get_summary(self) -> dict:
        """Gib Zusammenfassung zurück.
        
        Returns:
            Dictionary mit Optimierungsergebnissen
        """
        return {
            "optimal_k": self.optimal_k,
            "optimal_coherence_score": self.optimal_score,
            "coherence_scores": self.coherence_scores,
            "tested_range": {
                "min": self.min_topics,
                "max": self.max_topics,
                "step": self.step
            },
            "recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Gib Empfehlung basierend auf Coherence Score.
        
        Returns:
            Empfehlungstext
        """
        if not self.optimal_score:
            return "Keine Optimierung durchgeführt"
        
        if self.optimal_score > 0.5:
            return "Exzellente Themen-Kohärenz"
        elif self.optimal_score > 0.4:
            return "Gute Themen-Kohärenz"
        elif self.optimal_score > 0.3:
            return "Akzeptable Themen-Kohärenz"
        else:
            return "Niedrige Kohärenz - eventuell mehr Daten benötigt"


def find_optimal_topics(
    texts: List[str],
    count_matrix,
    count_vectorizer,
    min_topics: int = 2,
    max_topics: int = 15
) -> Tuple[int, Dict[int, float]]:
    """Convenience-Funktion zur Topic-Optimierung.
    
    Args:
        texts: Vorverarbeitete Texte
        count_matrix: Document-term matrix
        count_vectorizer: Fitted CountVectorizer
        min_topics: Minimale Anzahl Topics
        max_topics: Maximale Anzahl Topics
    
    Returns:
        (optimal_k, coherence_scores)
    
    Example:
        >>> optimal_k, scores = find_optimal_topics(texts, matrix, vectorizer)
        >>> print(f"Optimale Topic-Anzahl: {optimal_k}")
    """
    optimizer = TopicOptimizer(min_topics, max_topics)
    return optimizer.optimize(texts, count_matrix, count_vectorizer)
