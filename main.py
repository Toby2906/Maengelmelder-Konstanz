#!/usr/bin/env python3
"""Hauptprogramm für Mängelmelder Konstanz NLP-Analyse.

Verwendung:
    python main.py
    python main.py --input data/raw/maengelmelder_konstanz.csv --topics 5
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizer import TextVectorizer
from src.topic_modeler import TopicModeler
from src.analyzer import TextAnalyzer
from src.visualizer import Visualizer


def setup_logging(verbose: bool = False):
    """Konfiguriere Logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    )


def main(args):
    """Hauptfunktion."""
    start_time = time.time()
    
    print("="*60)
    print("MÄNGELMELDER KONSTANZ - NLP-ANALYSE")
    print("Projekt: Data Analysis (DLBDSEDA02_D)")
    print("="*60)
    print()
    
    # 1. DATEN LADEN
    print("📥 Schritt 1/6: Daten laden")
    loader = DataLoader(args.input)
    texts = loader.load()
    
    if not texts:
        print("❌ Fehler: Keine Daten geladen!")
        return 1
    
    stats = loader.get_stats()
    print(f"   ✓ {stats['total_texts']} Texte geladen")
    print(f"   ℹ Durchschnitt: {stats['avg_words']:.1f} Wörter pro Text")
    print()
    
    # 2. VORVERARBEITUNG
    print("🔧 Schritt 2/6: Text-Vorverarbeitung")
    preprocessor = Preprocessor()
    clean_texts = preprocessor.preprocess_batch(texts)
    
    if not clean_texts:
        print("❌ Fehler: Keine Texte nach Vorverarbeitung!")
        return 1
    
    prep_stats = preprocessor.get_stats(clean_texts)
    print(f"   ✓ {prep_stats['total_texts']} Texte vorverarbeitet")
    print(f"   ℹ Vokabular: {prep_stats['unique_tokens']} eindeutige Tokens")
    print()
    
    # 3. VEKTORISIERUNG
    print("🔢 Schritt 3/6: Vektorisierung")
    vectorizer = TextVectorizer()
    count_matrix, tfidf_matrix, features_count, features_tfidf, count_vec, tfidf_vec = vectorizer.fit_transform(clean_texts)
    print()
    
    # 4. TOPIC MODELING
    print(f"🎯 Schritt 4/6: Topic Modeling (LDA mit {args.topics} Themen)")
    topic_modeler = TopicModeler(n_topics=args.topics)
    topics = topic_modeler.fit(count_matrix, features_count)
    
    print("\n   📊 Extrahierte Themen:")
    for topic_id, words in topics:
        print(f"      Thema {topic_id}: {', '.join(words[:5])}...")
    print()
    
    # 5. N-GRAMM ANALYSE
    print("📈 Schritt 5/6: N-Gramm Analyse")
    analyzer = TextAnalyzer()
    bigrams = analyzer.extract_ngrams(clean_texts, n=2, top_n=10)
    top_words = analyzer.top_words(clean_texts, top_n=20)
    
    print("   Top-10 Bigramme:")
    for i, (bg, cnt) in enumerate(bigrams[:10], 1):
        print(f"      {i:2d}. '{bg}' ({cnt}x)")
    print()
    
    # 6. VISUALISIERUNGEN
    print("📊 Schritt 6/6: Visualisierungen erstellen")
    visualizer = Visualizer()
    visualizer.create_wordcloud(clean_texts)
    visualizer.create_top_words_plot(top_words)
    visualizer.create_pyldavis(topic_modeler.lda_model, count_matrix, count_vec)
    print()
    
    # EXPORT
    if Config.SAVE_JSON:
        results = {
            "data_stats": stats,
            "preprocessing_stats": prep_stats,
            "topics": topic_modeler.get_topics_dict(),
            "top_bigrams": dict(bigrams[:20]),
            "top_words": dict(top_words),
        }
        
        output_path = Config.ANALYSIS_DIR / "analysis_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Ergebnisse gespeichert: {output_path}")
    
    # ZUSAMMENFASSUNG
    elapsed = time.time() - start_time
    print()
    print("="*60)
    print("✅ ANALYSE ABGESCHLOSSEN")
    print("="*60)
    print(f"⏱️  Laufzeit: {elapsed:.2f} Sekunden")
    print(f"📁 Outputs:")
    print(f"   • Visualisierungen: {Config.VISUALIZATIONS_DIR}/")
    print(f"   • Reports: {Config.REPORTS_DIR}/")
    print(f"   • Analysen: {Config.ANALYSIS_DIR}/")
    print()
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mängelmelder Konstanz - NLP-Analyse"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(Config.DEFAULT_CSV_PATH),
        help="Pfad zur CSV-Datei"
    )
    parser.add_argument(
        "--topics", "-t",
        type=int,
        default=Config.LDA_N_TOPICS,
        help="Anzahl LDA-Themen"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ausführliches Logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    sys.exit(main(args))