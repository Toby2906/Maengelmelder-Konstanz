"""Text-Vorverarbeitung mit spaCy und NLTK.

Implementiert vollständige NLP-Pipeline: Normalisierung, Tokenisierung,
Lemmatisierung, Stoppwort-Entfernung.
"""
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Set, Tuple, Optional

from .config import Config

logger = logging.getLogger(__name__)

# Prüfe Verfügbarkeit von Bibliotheken
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords as nltk_stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


class Preprocessor:
    """Text-Vorverarbeitung für deutsche Texte."""
    
    def __init__(self):
        """Initialisiere Preprocessor."""
        self.stopwords: Set[str] = set()
        self.stopword_phrases: List[str] = []
        self.lemmatizer = None
        
        # Lade Stoppwörter
        self._load_stopwords()
        
        # Initialisiere Lemmatizer
        self._init_lemmatizer()
    
    def _load_stopwords(self):
        """Lade deutsche Stoppwörter."""
        # Aus Datei laden
        if Config.STOPWORDS_PATH.exists():
            try:
                with open(Config.STOPWORDS_PATH, encoding="utf-8") as f:
                    lines = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                
                # Normalisiere
                normalized = [
                    unicodedata.normalize("NFC", line).lower()
                    for line in lines
                ]
                
                # Trenne Einzelwörter und Phrasen
                for word in normalized:
                    if " " in word:
                        self.stopword_phrases.append(word)
                    else:
                        self.stopwords.add(word)
                
                logger.info(
                    f"✓ Stoppwörter geladen: {len(self.stopwords)} Wörter, "
                    f"{len(self.stopword_phrases)} Phrasen"
                )
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Stoppwörter: {e}")
        
        # Ergänze mit NLTK (falls verfügbar)
        if HAS_NLTK:
            try:
                nltk_stops = set(nltk_stopwords.words("german"))
                self.stopwords.update(nltk_stops)
                logger.debug(f"NLTK Stoppwörter hinzugefügt: {len(nltk_stops)}")
            except LookupError:
                logger.debug("NLTK stopwords nicht verfügbar")
    
    def _init_lemmatizer(self):
        """Initialisiere Lemmatizer (spaCy bevorzugt)."""
        if Config.USE_LEMMATIZATION:
            # Versuche spaCy
            if HAS_SPACY:
                try:
                    nlp = spacy.load(
                        Config.SPACY_MODEL,
                        exclude=Config.SPACY_DISABLE
                    )
                    
                    class SpacyLemmatizer:
                        def __init__(self, nlp_model):
                            self.nlp = nlp_model
                        
                        def lemmatize(self, word: str) -> str:
                            try:
                                doc = self.nlp.make_doc(word)
                                return doc[0].lemma_ if len(doc) > 0 else word
                            except:
                                return word
                    
                    self.lemmatizer = SpacyLemmatizer(nlp)
                    logger.info(f"✓ spaCy Lemmatizer initialisiert ({Config.SPACY_MODEL})")
                    return
                    
                except OSError:
                    logger.warning(
                        f"spaCy Modell '{Config.SPACY_MODEL}' nicht gefunden. "
                        f"Installiere mit: python -m spacy download {Config.SPACY_MODEL}"
                    )
            
            # Fallback: NLTK Stemmer
            if HAS_NLTK:
                try:
                    stemmer = SnowballStemmer("german")
                    
                    class StemmerWrapper:
                        def __init__(self, stemmer_instance):
                            self.stemmer = stemmer_instance
                        
                        def lemmatize(self, word: str) -> str:
                            try:
                                return self.stemmer.stem(word)
                            except:
                                return word
                    
                    self.lemmatizer = StemmerWrapper(stemmer)
                    logger.info("✓ NLTK Stemmer initialisiert (Fallback)")
                    return
                except:
                    pass
        
        # Identity Fallback
        class IdentityLemmatizer:
            def lemmatize(self, word: str) -> str:
                return word
        
        self.lemmatizer = IdentityLemmatizer()
        logger.info("ℹ Keine Lemmatisierung (Identity)")
    
    def preprocess(self, text: str) -> str:
        """Vorverarbeite einen Text.
        
        Args:
            text: Eingabetext
        
        Returns:
            Vorverarbeiteter Text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 1. Unicode-Normalisierung
        text = unicodedata.normalize("NFC", text).lower()
        
        # 2. Entferne Stoppwort-Phrasen
        for phrase in self.stopword_phrases:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            text = re.sub(pattern, " ", text)
        
        # 3. Entferne Sonderzeichen (behalte deutsche Umlaute)
        if Config.REMOVE_PUNCTUATION:
            text = re.sub(r"[^a-zäöüß\s]", " ", text)
        
        # 4. Normalisiere Whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # 5. Tokenisierung und Filterung
        tokens = text.split()
        processed_tokens = []
        
        for token in tokens:
            # Filtere Stoppwörter
            if token in self.stopwords:
                continue
            
            # Filtere nach Länge
            if len(token) < Config.MIN_WORD_LENGTH:
                continue
            if len(token) > Config.MAX_WORD_LENGTH:
                continue
            
            # Lemmatisierung
            if self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return " ".join(processed_tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Vorverarbeite mehrere Texte.
        
        Args:
            texts: Liste von Texten
        
        Returns:
            Liste von vorverarbeiteten Texten
        """
        logger.info(f"Vorverarbeite {len(texts)} Texte...")
        
        processed = []
        for i, text in enumerate(texts):
            cleaned = self.preprocess(text)
            if cleaned:
                processed.append(cleaned)
            
            # Progress
            if Config.ENABLE_PROGRESS_BAR and (i + 1) % 100 == 0:
                logger.debug(f"  Verarbeitet: {i + 1}/{len(texts)}")
        
        removed = len(texts) - len(processed)
        logger.info(
            f"✓ Vorverarbeitung abgeschlossen: {len(processed)} Texte "
            f"({removed} leer/nur Stoppwörter)"
        )
        
        return processed
    
    def get_stats(self, texts: List[str]) -> dict:
        """Gib Vorverarbeitungs-Statistiken zurück.
        
        Args:
            texts: Liste von Texten
        
        Returns:
            Dictionary mit Statistiken
        """
        if not texts:
            return {}
        
        # Token-Statistiken
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        
        unique_tokens = set(all_tokens)
        
        return {
            "total_texts": len(texts),
            "total_tokens": len(all_tokens),
            "unique_tokens": len(unique_tokens),
            "avg_tokens_per_text": len(all_tokens) / len(texts),
            "vocabulary_richness": len(unique_tokens) / len(all_tokens) if all_tokens else 0,
        }


def preprocess_texts(texts: List[str]) -> List[str]:
    """Convenience-Funktion für Vorverarbeitung.
    
    Args:
        texts: Liste von Texten
    
    Returns:
        Liste von vorverarbeiteten Texten
    """
    preprocessor = Preprocessor()
    return preprocessor.preprocess_batch(texts)