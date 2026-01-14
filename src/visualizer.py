"""Visualisierungen erstellen."""
import logging
import os
from typing import List
from .config import Config

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import pyLDAvis
    import pyLDAvis.sklearn
    HAS_PYLDAVIS = True
except ImportError:
    HAS_PYLDAVIS = False


class Visualizer:
    """Erstelle Visualisierungen."""
    
    def create_wordcloud(self, texts: List[str], filename: str = "wordcloud.png"):
        """Erstelle WordCloud."""
        if not HAS_WORDCLOUD or not HAS_MATPLOTLIB:
            logger.warning("wordcloud oder matplotlib nicht verfügbar")
            return
        
        logger.info("Erstelle WordCloud...")
        
        text_combined = " ".join(texts)
        wc = WordCloud(
            width=Config.WORDCLOUD_WIDTH,
            height=Config.WORDCLOUD_HEIGHT,
            background_color=Config.WORDCLOUD_BACKGROUND,
            max_words=Config.WORDCLOUD_MAX_WORDS,
            colormap=Config.WORDCLOUD_COLORMAP
        ).generate(text_combined)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud - Häufigste Wörter", fontsize=16)
        plt.tight_layout()
        
        path = Config.VISUALIZATIONS_DIR / filename
        plt.savefig(path, dpi=Config.PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✓ WordCloud gespeichert: {path}")
    
    def create_top_words_plot(self, top_words, filename: str = "top_words.png"):
        """Erstelle Balkendiagramm für Top-Wörter."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib nicht verfügbar")
            return
        
        logger.info("Erstelle Top-Wörter Plot...")
        
        words = [w for w, c in top_words][::-1]
        counts = [c for w, c in top_words][::-1]
        
        plt.figure(figsize=Config.PLOT_FIGSIZE)
        plt.barh(words, counts, color="steelblue")
        plt.xlabel("Häufigkeit", fontsize=12)
        plt.title(f"Top {len(words)} Wörter", fontsize=14)
        plt.tight_layout()
        
        path = Config.VISUALIZATIONS_DIR / filename
        plt.savefig(path, dpi=Config.PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✓ Top-Wörter Plot gespeichert: {path}")
    
    def create_pyldavis(self, lda_model, count_matrix, count_vectorizer, filename: str = "lda_interactive.html"):
        """Erstelle pyLDAvis Visualisierung."""
        if not HAS_PYLDAVIS:
            logger.warning("pyLDAvis nicht verfügbar")
            return
        
        logger.info("Erstelle pyLDAvis Visualisierung...")
        
        try:
            panel = pyLDAvis.sklearn.prepare(
                lda_model,
                count_matrix,
                count_vectorizer,
                mds=Config.PYLDAVIS_MDS
            )
            
            path = Config.REPORTS_DIR / filename
            pyLDAvis.save_html(panel, str(path))
            
            logger.info(f"✓ pyLDAvis gespeichert: {path}")
            logger.info(f"  Öffne mit: python -m http.server -d output/reports/")
        except Exception as e:
            logger.error(f"pyLDAvis Fehler: {e}")