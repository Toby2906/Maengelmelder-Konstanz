"""Vorverarbeitung: Stopwörter laden und Texte reinigen."""
import re
from typing import Tuple, Set, List, Optional
from pathlib import Path


def load_local_stopwords(path: Optional[str] = None) -> Tuple[Set[str], List[str]]:
    """Lade lokale Stopwort-Liste. Standard: Stoppwörter/stopwords_de.txt im Projektstamm."""
    try:
        import unicodedata
        if path is None:
            path = str(Path(__file__).resolve().parents[1] / 'Stoppwörter' / 'stopwords_de.txt')
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
        return single, phrases
    except Exception:
        return set(), []


def prepare_stopwords_and_lemmatizer():
    """Bereite Stopwörter und (falls möglich) einen Lemmatizer oder Stemmer vor.

    Versucht einen SnowballStemmer ('german') über NLTK zu verwenden. Falls das
    nicht verfügbar ist, wird ein Identity-Lemmatizer zurückgegeben.
    """
    stop_words, stop_phrases = load_local_stopwords()
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            try:
                if not stop_words:
                    stop_words = set(stopwords.words('german'))
            except Exception:
                pass
        except Exception:
            pass

        try:
            from nltk.stem import SnowballStemmer
            stemmer = SnowballStemmer('german')

            class _StemmerWrapper:
                def lemmatize(self, w, pos='n'):
                    try:
                        return stemmer.stem(w)
                    except Exception:
                        return w

            lemmatizer = _StemmerWrapper()
            return stop_words, stop_phrases, lemmatizer
        except Exception:
            pass
    except Exception:
        pass

    class _IdentityLemmatizer:
        def lemmatize(self, w, pos='n'):
            return w

    return stop_words, stop_phrases, _IdentityLemmatizer()


def preprocess_text(text: str, stop_words: set, stop_phrases: List[str], lemmatizer=None) -> str:
    if not isinstance(text, str):
        return ""
    import unicodedata
    text = unicodedata.normalize('NFC', text).lower()
    try:
        for phrase in stop_phrases:
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", ' ', text)
    except Exception:
        pass
    text = re.sub(r"[^a-zäöüß\s]", ' ', text)
    tokens = text.split()
    processed_tokens = []
    for t in tokens:
        if t in stop_words or len(t) <= 2:
            continue
        if lemmatizer is not None:
            try:
                t = lemmatizer.lemmatize(t)
            except Exception:
                pass
        processed_tokens.append(t)
    return " ".join(processed_tokens)
