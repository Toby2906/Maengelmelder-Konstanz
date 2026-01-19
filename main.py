#!/usr/bin/env python3
"""Hauptprogramm für Mängelmelder Konstanz NLP-Analyse.

Automatische Topic-Anzahl Optimierung mit Coherence Score.

Verwendung:
    python main.py                          # Auto-Optimierung
    python main.py --topics 5               # Manuelle Topic-Anzahl
    python main.py --optimize-topics        # Nur Optimierung, dann stoppen
    python main.py --min-topics 2 --max-topics 10  # Custom Range
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
from src.topic_optimizer import TopicOptimizer
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


def print_header():
    """Zeige Projekt-Header."""
    print("="*70)
    print("MÄNGELMELDER KONSTANZ - NLP-ANALYSE MIT COHERENCE SCORE OPTIMIERUNG")
    print("Projekt: Data Analysis (DLBDSEDA02_D)")
    print("Autor: Tobias Seekatz")
    print("="*70)
    print()


def main(args):
    """Hauptfunktion."""
    start_time = time.time()
    print_header()
    
    # ========================================================================
    # PHASE 1: DATEN LADEN
    # ========================================================================
    print("📥 Phase 1/7: Daten laden")
    print("-" * 70)
    loader = DataLoader(args.input)
    texts = loader.load()
    
    if not texts:
        print("❌ Fehler: Keine Daten geladen!")
        return 1
    
    stats = loader.get_stats()
    print(f"   ✓ {stats['total_texts']} Texte geladen")
    print(f"   ℹ Durchschnittliche Länge: {stats['avg_words']:.1f} Wörter/Text")
    print(f"   ℹ Bereich: {stats['min_words']} - {stats['max_words']} Wörter")
    print()
    
    # ========================================================================
    # PHASE 2: VORVERARBEITUNG
    # ========================================================================
    print("🔧 Phase 2/7: Text-Vorverarbeitung")
    print("-" * 70)
    preprocessor = Preprocessor()
    clean_texts = preprocessor.preprocess_batch(texts)
    
    if not clean_texts:
        print("❌ Fehler: Keine Texte nach Vorverarbeitung!")
        return 1
    
    prep_stats = preprocessor.get_stats(clean_texts)
    print(f"   ✓ {prep_stats['total_texts']} Texte vorverarbeitet")
    print(f"   ℹ Gesamt-Tokens: {prep_stats['total_tokens']:,}")
    print(f"   ℹ Eindeutige Tokens: {prep_stats['unique_tokens']:,}")
    print(f"   ℹ Vokabular-Reichtum: {prep_stats['vocabulary_richness']:.3f}")
    print()
    
    # ========================================================================
    # PHASE 3: VEKTORISIERUNG
    # ========================================================================
    print("🔢 Phase 3/7: Vektorisierung (BoW & TF-IDF)")
    print("-" * 70)
    vectorizer = TextVectorizer()
    count_matrix, tfidf_matrix, features_count, features_tfidf, count_vec, tfidf_vec = \
        vectorizer.fit_transform(clean_texts)
    print()
    
    # ========================================================================
    # PHASE 4: TOPIC-ANZAHL OPTIMIERUNG (NEU!)
    # ========================================================================
    optimal_k = None
    coherence_scores = {}
    
    if args.auto_optimize or args.optimize_topics:
        print("🎯 Phase 4/7: Topic-Anzahl Optimierung (Coherence Score)")
        print("-" * 70)
        
        optimizer = TopicOptimizer(
            min_topics=args.min_topics,
            max_topics=args.max_topics,
            step=1
        )
        
        optimal_k, coherence_scores = optimizer.optimize(
            clean_texts,
            count_matrix,
            count_vec
        )
        
        # Zeige Ergebnisse
        print("\n   📊 Coherence Scores:")
        for k in sorted(coherence_scores.keys()):
            marker = " ⭐" if k == optimal_k else ""
            print(f"      k={k:2d}: {coherence_scores[k]:.4f}{marker}")
        
        print(f"\n   ✓ Empfohlene Topic-Anzahl: k={optimal_k}")
        print(f"     ({optimizer._get_recommendation()})")
        
        # Speichere Coherence Plot
        optimizer.plot_coherence_scores(
            save_path=str(Config.VISUALIZATIONS_DIR / "coherence_scores.png")
        )
        
        # Speichere Optimierungs-Ergebnisse
        optimization_results = optimizer.get_summary()
        with open(Config.ANALYSIS_DIR / "topic_optimization.json", "w") as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False)
        
        print()
        
        # Falls nur Optimierung gewünscht
        if args.optimize_topics:
            print("="*70)
            print("✅ TOPIC-OPTIMIERUNG ABGESCHLOSSEN")
            print(f"   Empfehlung: Verwende --topics {optimal_k}")
            print("="*70)
            return 0
        
        # Verwende optimales k für LDA
        n_topics = optimal_k
    else:
        print("🎯 Phase 4/7: Topic-Anzahl (manuell)")
        print("-" * 70)
        n_topics = args.topics
        print(f"   ℹ Verwende manuelle Anzahl: k={n_topics}")
        print("   💡 Tipp: Verwende --auto-optimize für automatische Optimierung")
        print()
    
    # ========================================================================
    # PHASE 5: TOPIC MODELING
    # ========================================================================
    print(f"📚 Phase 5/7: Topic Modeling (LDA mit k={n_topics})")
    print("-" * 70)
    
    topic_modeler = TopicModeler(n_topics=n_topics)
    topics = topic_modeler.fit(count_matrix, features_count)
    
    print("\n   📊 Extrahierte Themen:")
    for topic_id, words in topics:
        print(f"      Thema {topic_id}: {', '.join(words[:7])}")
    print()
    
    # ========================================================================
    # PHASE 6: N-GRAMM ANALYSE
    # ========================================================================
    print("📈 Phase 6/7: N-Gramm Analyse & Top-Wörter")
    print("-" * 70)
    
    analyzer = TextAnalyzer()
    bigrams = analyzer.extract_ngrams(clean_texts, n=2, top_n=15)
    top_words = analyzer.top_words(clean_texts, top_n=20)
    
    print("   Top-10 Bigramme:")
    for i, (bg, cnt) in enumerate(bigrams[:10], 1):
        print(f"      {i:2d}. '{bg}' ({cnt}x)")
    
    print("\n   Top-10 Einzelwörter:")
    for i, (word, cnt) in enumerate(top_words[:10], 1):
        print(f"      {i:2d}. '{word}' ({cnt}x)")
    print()
    
    # ========================================================================
    # PHASE 7: VISUALISIERUNGEN
    # ========================================================================
    print("📊 Phase 7/7: Visualisierungen & Export")
    print("-" * 70)
    
    visualizer = Visualizer()
    visualizer.create_wordcloud(clean_texts)
    visualizer.create_top_words_plot(top_words)
    visualizer.create_pyldavis(topic_modeler.lda_model, count_matrix, count_vec)
    
    # Erstelle auch Bigram-Plot
    if bigrams:
        visualizer.create_top_words_plot(
            bigrams[:20],
            filename="top_bigrams.png"
        )
    
    print()
    
    # ========================================================================
    # EXPORT ERGEBNISSE
    # ========================================================================
    if Config.SAVE_JSON:
        results = {
            "metadata": {
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_topics_used": n_topics,
                "optimized": args.auto_optimize or args.optimize_topics,
            },
            "data_statistics": stats,
            "preprocessing_statistics": prep_stats,
            "topics": topic_modeler.get_topics_dict(),
            "top_bigrams": dict(bigrams[:20]),
            "top_words": dict(top_words[:20]),
        }
        
        # Füge Optimierungs-Daten hinzu
        if coherence_scores:
            results["topic_optimization"] = {
                "optimal_k": optimal_k,
                "coherence_scores": coherence_scores,
                "recommendation": optimizer._get_recommendation()
            }
        
        output_path = Config.ANALYSIS_DIR / "analysis_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ JSON-Ergebnisse gespeichert: {output_path}")
    
    # ========================================================================
    # ZUSAMMENFASSUNG
    # ========================================================================
    elapsed = time.time() - start_time
    
    print()
    print("="*70)
    print("✅ ANALYSE ERFOLGREICH ABGESCHLOSSEN")
    print("="*70)
    print()
    print("📊 Zusammenfassung:")
    print(f"   • Texte analysiert: {len(clean_texts)}")
    print(f"   • Vokabular-Größe: {prep_stats['unique_tokens']:,} eindeutige Tokens")
    print(f"   • Themen extrahiert: {n_topics}")
    if optimal_k:
        print(f"   • Optimale Topic-Anzahl: {optimal_k} (Coherence: {coherence_scores[optimal_k]:.4f})")
    print(f"   • Laufzeit: {elapsed:.2f} Sekunden")
    print()
    print("📁 Outputs:")
    print(f"   • Visualisierungen: {Config.VISUALIZATIONS_DIR}/")
    if optimal_k:
        print(f"     - coherence_scores.png (Topic-Optimierung)")
    print(f"     - wordcloud.png")
    print(f"     - top_words.png")
    print(f"     - top_bigrams.png")
    print(f"   • Reports: {Config.REPORTS_DIR}/")
    print(f"     - lda_interactive.html (Öffne im Browser)")
    print(f"   • Analysen: {Config.ANALYSIS_DIR}/")
    print(f"     - analysis_results.json")
    if optimal_k:
        print(f"     - topic_optimization.json")
    print()
    print("💡 Tipp:")
    print(f"   Öffne interaktiven Report: python -m http.server -d output/reports/")
    print()
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mängelmelder Konstanz - NLP-Analyse mit Coherence Score Optimierung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Automatische Topic-Anzahl Optimierung
  python main.py --auto-optimize
  
  # Nur Optimierung durchführen (ohne vollständige Analyse)
  python main.py --optimize-topics
  
  # Manuelle Topic-Anzahl
  python main.py --topics 5
  
  # Custom Optimierungs-Bereich
  python main.py --auto-optimize --min-topics 3 --max-topics 12
  
  # Mit eigener CSV-Datei
  python main.py --input data/raw/meine_daten.csv --auto-optimize
        """
    )
    
    # Eingabe
    parser.add_argument(
        "--input", "-i",
        default=str(Config.DEFAULT_CSV_PATH),
        help="Pfad zur CSV-Datei (Standard: data/raw/maengelmelder_konstanz.csv)"
    )
    
    # Topic-Anzahl
    topic_group = parser.add_mutually_exclusive_group()
    topic_group.add_argument(
        "--topics", "-t",
        type=int,
        default=None,
        help="Manuelle Topic-Anzahl (deaktiviert Auto-Optimierung)"
    )
    topic_group.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatische Optimierung der Topic-Anzahl mit Coherence Score"
    )
    topic_group.add_argument(
        "--optimize-topics",
        action="store_true",
        help="Nur Topic-Optimierung durchführen, dann beenden"
    )
    
    # Optimierungs-Parameter
    parser.add_argument(
        "--min-topics",
        type=int,
        default=2,
        help="Minimale Topic-Anzahl für Optimierung (Standard: 2)"
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=15,
        help="Maximale Topic-Anzahl für Optimierung (Standard: 15)"
    )
    
    # Sonstiges
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ausführliches Logging (DEBUG-Level)"
    )
    
    args = parser.parse_args()
    
    # Validierung
    if args.topics is None and not args.auto_optimize and not args.optimize_topics:
        # Default: Auto-Optimierung
        args.auto_optimize = True
    
    if args.min_topics >= args.max_topics:
        print("❌ Fehler: --min-topics muss kleiner als --max-topics sein!")
        sys.exit(1)
    
    # Setup & Run
    setup_logging(args.verbose)
    sys.exit(main(args))
