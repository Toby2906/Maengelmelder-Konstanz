"""Vorverarbeitung: Stopwörter laden und Texte reinigen."""
import re
from typing import Tuple, Set, List

def load_local_stopwords(path: str = 'stopwords_de.txt') -> Tuple[Set[str], List[str]]:
    try:
        import unicodedata
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
        return None


def prepare_stopwords_and_lemmatizer():
    """Bereite Stopwörter und (falls möglich) einen Lemmatizer vor."""
    try:
        import nltk
        from nltk.corpus import stopwords
        def ensure_nltk_resources():
            resources = ['stopwords', 'wordnet', 'punkt']
            for res in resources:
                try:
                    nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
                except LookupError:
                    try:
                        nltk.download(res, quiet=True)
                    except Exception:
                        pass
        ensure_nltk_resources()
        local = load_local_stopwords()
        if local is not None:
            stop_words, stop_phrases = local
        else:
            try:
                stop_words = set(stopwords.words('german'))
            except Exception:
                stop_words = set()
            stop_phrases = []

        class _IdentityLemmatizer:
            def lemmatize(self, w, pos='n'):
                return w

        lemmatizer = _IdentityLemmatizer()
        return stop_words, stop_phrases, lemmatizer
    except Exception:
        # Fallbacks
        local = load_local_stopwords()
        if local is not None:
            stop_words, stop_phrases = local
        else:
            stop_words = set([
                "und", "der", "die", "das", "ein", "eine", "in", "auf", "mit", "für",
                "ist", "sind", "bei", "zu", "von", "an", "dem", "den", "des",
                "aber", "oder", "nicht", "kann", "auch", "so", "wie", "über", "vor"
            ])
            stop_phrases = []
        class _IdentityLemmatizer:
            def lemmatize(self, w, pos='n'):
                return w
        return stop_words, stop_phrases, _IdentityLemmatizer()


def preprocess_text(text: str, stop_words: set, stop_phrases: List[str]) -> str:
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
    processed_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(processed_tokens)
