"""Utility: Feature flags for optional libraries and small helpers."""
HAS_PANDAS = False
HAS_NLTK = False
HAS_SKLEARN = False
HAS_GENSIM = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    pd = None

try:
    import nltk
    HAS_NLTK = True
except Exception:
    nltk = None

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except Exception:
    CountVectorizer = None
    TfidfVectorizer = None
    LatentDirichletAllocation = None

try:
    from gensim import corpora
    from gensim.models import LdaModel
    HAS_GENSIM = True
except Exception:
    corpora = None
    LdaModel = None

def print_flags():
    print(f"HAS_PANDAS={HAS_PANDAS}, HAS_NLTK={HAS_NLTK}, HAS_SKLEARN={HAS_SKLEARN}, HAS_GENSIM={HAS_GENSIM}")
