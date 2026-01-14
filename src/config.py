"""Zentrale Konfiguration für Mängelmelder Konstanz Analyse.

Alle Einstellungen für die NLP-Pipeline an einem Ort.
"""
from pathlib import Path


class Config:
    """Haupt-Konfiguration."""
    
    # Projekt-Pfade
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "output"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"
    RESOURCES_DIR = BASE_DIR / "resources"
    
    # Daten
    DEFAULT_CSV_PATH = RAW_DATA_DIR / "maengelmelder_konstanz.csv"
    STOPWORDS_PATH = RESOURCES_DIR / "stopwords_de.txt"
    
    # CSV-Einstellungen
    CSV_ENCODING = "utf-8"
    CSV_DELIMITER = ","
    CSV_TEXT_COLUMN = 2  # Index der Spalte mit Beschreibungen
    
    # Text-Vorverarbeitung
    MIN_WORD_LENGTH = 3  # Minimale Wortlänge
    MAX_WORD_LENGTH = 50  # Maximale Wortlänge
    REMOVE_NUMBERS = True  # Zahlen entfernen
    REMOVE_PUNCTUATION = True  # Satzzeichen entfernen
    
    # spaCy Einstellungen
    SPACY_MODEL = "de_core_news_sm"  # Deutsches Modell
    SPACY_DISABLE = ["parser", "ner"]  # Deaktiviere nicht benötigte Komponenten
    USE_LEMMATIZATION = True  # Lemmatisierung aktivieren
    
    # Vektorisierung
    VECTORIZER_MAX_FEATURES = 500  # Maximale Anzahl Features
    VECTORIZER_MAX_DF = 0.95  # Maximale Dokument-Frequenz (95%)
    VECTORIZER_MIN_DF = 2  # Minimale Dokument-Frequenz (mindestens 2 Dokumente)
    VECTORIZER_NGRAM_RANGE = (1, 1)  # Nur Unigramme für Vektorisierung
    
    # Topic Modeling (LDA)
    LDA_N_TOPICS = 5  # Anzahl Themen
    LDA_MAX_ITER = 20  # Maximale Iterationen
    LDA_RANDOM_STATE = 42  # Reproduzierbarkeit
    LDA_N_JOBS = -1  # Nutze alle CPU-Kerne
    LDA_TOP_WORDS = 10  # Top-N Wörter pro Thema
    
    # N-Gramm Analyse
    NGRAM_RANGE = (2, 2)  # Bigramme
    NGRAM_TOP_N = 20  # Top-20 N-Gramme
    
    # Visualisierung
    WORDCLOUD_WIDTH = 1200
    WORDCLOUD_HEIGHT = 600
    WORDCLOUD_BACKGROUND = "white"
    WORDCLOUD_MAX_WORDS = 100
    WORDCLOUD_COLORMAP = "viridis"
    
    PLOT_FIGSIZE = (12, 8)
    PLOT_DPI = 150
    PLOT_STYLE = "seaborn-v0_8-darkgrid"
    
    # pyLDAvis
    PYLDAVIS_MDS = "tsne"  # Dimensionsreduktion (tsne oder pcoa)
    
    # Logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Performance
    ENABLE_PROGRESS_BAR = True  # Fortschrittsbalken anzeigen
    ENABLE_TIMING = True  # Laufzeit messen
    
    # Export
    SAVE_VISUALIZATIONS = True  # Bilder speichern
    SAVE_JSON = True  # JSON-Exports
    SAVE_HTML_REPORT = True  # HTML-Report
    
    @classmethod
    def create_directories(cls):
        """Erstelle alle notwendigen Verzeichnisse."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.VISUALIZATIONS_DIR,
            cls.REPORTS_DIR,
            cls.ANALYSIS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_summary(cls) -> dict:
        """Gib Konfigurations-Zusammenfassung zurück."""
        return {
            "data": {
                "csv_path": str(cls.DEFAULT_CSV_PATH),
                "text_column": cls.CSV_TEXT_COLUMN,
                "encoding": cls.CSV_ENCODING,
            },
            "preprocessing": {
                "min_word_length": cls.MIN_WORD_LENGTH,
                "lemmatization": cls.USE_LEMMATIZATION,
                "spacy_model": cls.SPACY_MODEL,
            },
            "vectorization": {
                "max_features": cls.VECTORIZER_MAX_FEATURES,
                "max_df": cls.VECTORIZER_MAX_DF,
                "min_df": cls.VECTORIZER_MIN_DF,
            },
            "lda": {
                "n_topics": cls.LDA_N_TOPICS,
                "max_iter": cls.LDA_MAX_ITER,
                "top_words": cls.LDA_TOP_WORDS,
            },
            "output": {
                "visualizations": str(cls.VISUALIZATIONS_DIR),
                "reports": str(cls.REPORTS_DIR),
                "analysis": str(cls.ANALYSIS_DIR),
            },
        }


# Erstelle Verzeichnisse beim Import
Config.create_directories()