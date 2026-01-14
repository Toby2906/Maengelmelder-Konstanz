"""N-Gramm Analyse."""
import logging
from collections import Counter
from typing import List, Tuple
from .config import Config

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TextAnalyzer:
    """Analysiere Texte (N-Gramme, Top-Wörter)."""
    
    def extract_ngrams(self, texts: List[str], n: int = 2, top_n: int = None) -> List[Tuple[str, int]]:
        """Extrahiere N-Gramme.
        
        Args:
            texts: Liste von Texten
            n: N-Gramm-Größe (2 = Bigramme)
            top_n: Anzahl Top-N-Gramme
        
        Returns:
            Liste von (ngram, count)
        """
        top_n = top_n or Config.NGRAM_TOP_N
        
        if HAS_SKLEARN:
            try:
                vec = CountVectorizer(ngram_range=(n, n), max_features=100, min_df=2)
                matrix = vec.fit_transform(texts)
                sums = matrix.sum(axis=0).A1
                top_indices = sums.argsort()[-top_n:][::-1]
                return [(vec.get_feature_names_out()[i], int(sums[i])) for i in top_indices]
            except:
                pass
        
        # Fallback
        ngram_counts = Counter()
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngram_counts[ngram] += 1
        
        return ngram_counts.most_common(top_n)
    
    def top_words(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, int]]:
        """Extrahiere Top-Wörter.
        
        Returns:
            Liste von (word, count)
        """
        freq = Counter()
        for text in texts:
            freq.update(text.split())
        return freq.most_common(top_n)