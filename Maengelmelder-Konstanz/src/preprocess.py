"""Vorverarbeitung: Stoppwörter laden und Texte reinigen."""
import re
import logging
from typing import Tuple, Set, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_local_stopwords(path: Optional[str] = None) -> Tuple[Set[str], List[str]]:
    """Lade lokale Stopwort-Liste. Standard: Stoppwörter/stopwords_de.txt im Projektstamm."""
    try:
        import unicodedata
        if path is None:
            path = str(Path(__file__).resolve().parents[1] / 'Stoppwörter' / 'stopwords_de.txt')
        
        if not Path(path).exists():
            logger.warning(f"Stopwords-Datei nicht gefunden: {path}")
            return set(), []
            
        with open(path, encoding='utf-8') as f:
            raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
        
        normalized = [unicodedata.normalize('NFC', ln).lower() for ln in raw]
        single = set()
        phrases = []
        
        for w in normalized:
            if ' ' in w:
                phrases.append(re.sub(r'\s+', ' ', w).strip())
            else:
                single.add(w)
        
        logger.info(f"Stopwords geladen: {len(single)} Einzelwörter, {len(phrases)} Phrasen")
        return single, phrases
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der Stopwords: {e}", exc_info=True)
        return set(), []


def prepare_stopwords_and_lemmatizer():
    """Bereite Stoppwörter und (falls möglich) einen Lemmatizer oder Stemmer vor.

    Priorität:
    1. spaCy Lemmatizer (beste Qualität)
    2. NLTK SnowballStemmer (gute Performance)
    3. Identity Fallback (keine Lemmatisierung)
    
    Returns:
        Tuple[Set[str], List[str], Lemmatizer]: (stop_words, stop_phrases, lemmatizer)
    """
    stop_words, stop_phrases = load_local_stopwords()

    # 1. Versuche spaCy Lemmatizer (bevorzugt)
    try:
        import spacy
        try:
            nlp = spacy.load('de_core_news_sm', exclude=['parser', 'ner'])
            
            class _SpacyLemmatizer:
                """Optimierter spaCy Lemmatizer für einzelne Wörter."""
                def __init__(self, nlp_model):
                    self.nlp = nlp_model
                
                def lemmatize(self, word: str, pos: str = 'n') -> str:
                    """Lemmatisiere ein einzelnes Wort.
                    
                    Args:
                        word: Zu lemmatisierendes Wort
                        pos: Part-of-speech Tag (wird ignoriert bei spaCy)
                    
                    Returns:
                        Lemmatisiertes Wort
                    """
                    try:
                        # Verwende make_doc statt nlp() für bessere Performance
                        doc = self.nlp.make_doc(word)
                        if len(doc) > 0:
                            return doc[0].lemma_
                        return word
                    except Exception:
                        return word

            lemmatizer = _SpacyLemmatizer(nlp)
            logger.info("Verwende spaCy Lemmatizer (de_core_news_sm)")
            return stop_words, stop_phrases, lemmatizer
            
        except OSError:
            logger.warning("spaCy Modell 'de_core_news_sm' nicht installiert. "
                         "Installiere mit: python -m spacy download de_core_news_sm")
        except Exception as e:
            logger.warning(f"spaCy konnte nicht initialisiert werden: {e}")
    except ImportError:
        logger.info("spaCy nicht verfügbar")

    # 2. Versuche NLTK Snowball Stemmer
    try:
        from nltk.stem import SnowballStemmer
        from nltk.corpus import stopwords
        
        try:
            if not stop_words:
                nltk_stops = set(stopwords.words('german'))
                stop_words.update(nltk_stops)
                logger.info(f"NLTK Stopwords hinzugefügt: {len(nltk_stops)} Wörter")
        except LookupError:
            logger.warning("NLTK stopwords nicht verfügbar. "
                         "Führe aus: python -m nltk.downloader stopwords")
        except Exception as e:
            logger.warning(f"NLTK Stopwords konnten nicht geladen werden: {e}")
        
        stemmer = SnowballStemmer('german')

        class _StemmerWrapper:
            """Wrapper für NLTK Snowball Stemmer."""
            def __init__(self, stemmer_instance):
                self.stemmer = stemmer_instance
            
            def lemmatize(self, word: str, pos: str = 'n') -> str:
                """Stemme ein Wort.
                
                Args:
                    word: Zu stemmendes Wort
                    pos: Part-of-speech Tag (wird ignoriert)
                
                Returns:
                    Gestemmtes Wort
                """
                try:
                    return self.stemmer.stem(word)
                except Exception:
                    return word

        lemmatizer = _StemmerWrapper(stemmer)
        logger.info("Verwende NLTK SnowballStemmer (Fallback)")
        return stop_words, stop_phrases, lemmatizer
        
    except ImportError:
        logger.info("NLTK nicht verfügbar")
    except Exception as e:
        logger.warning(f"NLTK Stemmer konnte nicht initialisiert werden: {e}")

    # 3. Fallback: Identity (keine Lemmatisierung)
    class _IdentityLemmatizer:
        """Fallback: Keine Lemmatisierung/Stemming."""
        def lemmatize(self, word: str, pos: str = 'n') -> str:
            """Gib Wort unverändert zurück."""
            return word

    logger.warning("Keine Lemmatisierung verfügbar - verwende Identity-Fallback")
    return stop_words, stop_phrases, _IdentityLemmatizer()


def preprocess_text(
    text: str,
    stop_words: Set[str],
    stop_phrases: List[str],
    lemmatizer=None
) -> str:
    """Vorverarbeitung eines Textes.
    
    Schritte:
    1. Unicode-Normalisierung
    2. Lowercase
    3. Entfernung von Stoppwort-Phrasen
    4. Entfernung von Sonderzeichen (außer deutschen Umlauten)
    5. Tokenisierung
    6. Filterung: Stoppwörter, kurze Wörter (≤2 Zeichen)
    7. Lemmatisierung (optional)
    
    Args:
        text: Eingabetext
        stop_words: Set von Stoppwörtern
        stop_phrases: Liste von Stoppwort-Phrasen
        lemmatizer: Optionaler Lemmatizer mit .lemmatize() Methode
    
    Returns:
        Vorverarbeiteter Text (space-separated tokens)
    """
    if not isinstance(text, str):
        logger.warning(f"Ungültiger Texttyp: {type(text)}")
        return ""
    
    if not text.strip():
        return ""
    
    try:
        import unicodedata
        
        # 1. Unicode-Normalisierung (NFC = composed form)
        text = unicodedata.normalize('NFC', text).lower()
        
        # 2. Entferne Stoppwort-Phrasen
        for phrase in stop_phrases:
            # Word boundary für exakte Phrasen-Matches
            pattern = r"\b" + re.escape(phrase) + r"\b"
            text = re.sub(pattern, ' ', text)
        
        # 3. Entferne Sonderzeichen (behalte deutsche Umlaute)
        text = re.sub(r"[^a-zäöüß\s]", ' ', text)
        
        # 4. Normalisiere Whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 5. Tokenisierung
        tokens = text.split()
        
        # 6. Filterung und Lemmatisierung
        processed_tokens = []
        for token in tokens:
            # Filtere Stoppwörter
            if token in stop_words:
                continue
            
            # Filtere sehr kurze Wörter
            if len(token) <= 2:
                continue
            
            # Lemmatisierung (falls verfügbar)
            if lemmatizer is not None:
                try:
                    token = lemmatizer.lemmatize(token)
                except Exception as e:
                    logger.debug(f"Lemmatisierung fehlgeschlagen für '{token}': {e}")
            
            processed_tokens.append(token)
        
        return " ".join(processed_tokens)
        
    except Exception as e:
        logger.error(f"Fehler bei Textvorverarbeitung: {e}", exc_info=True)
        return ""