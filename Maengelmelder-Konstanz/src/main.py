"""Haupteinstiegspunkt für NLP-Analyse.

Orchestriert das Laden, Vorverarbeiten, Vektorisieren und Analysieren
von Bürgerbeschwerden aus CSV-Dateien.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wenn als Skript ausgeführt, stelle sicher dass src im Path ist
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetch import load_data_from_csv
from src.preprocess import prepare_stopwords_and_lemmatizer, preprocess_text
from src.topics import (
    vectorize_texts,
    extract_topics_lda,
    extract_ngrams,
    top_words,
    visualize_and_save,
)


def sanitize_output_path(out_dir: str, project_root: Optional[Path] = None) -> Path:
    """Validiere und bereinige Output-Pfad gegen Path Traversal.
    
    Args:
        out_dir: Benutzer-angegebener Output-Pfad
        project_root: Projekt-Root (optional, wird automatisch erkannt)
    
    Returns:
        Validierter absoluter Pfad
    
    Raises:
        ValueError: Wenn Pfad außerhalb des Projekts liegt
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    
    # Resolve zu absolutem Pfad
    path = Path(out_dir).resolve()
    
    # Prüfe ob innerhalb des Projekts oder explizit erlaubt
    # Erlaube auch absolute Pfade wenn sie explizit angegeben werden
    if not out_dir.startswith('/') and not out_dir.startswith('\\'):
        # Relativer Pfad - stelle sicher er ist im Projekt
        try:
            path.relative_to(project_root)
        except ValueError:
            raise ValueError(
                f"Output-Pfad '{out_dir}' liegt außerhalb des Projekts. "
                f"Verwende einen Pfad relativ zu: {project_root}"
            )
    
    return path


def run(
    filepath: str = 'data/maengelmelder_konstanz.csv',
    num_topics: int = 5,
    out_dir: str = 'out'
) -> int:
    """Führe komplette NLP-Analyse-Pipeline aus.
    
    Args:
        filepath: Pfad zur CSV-Datei mit Bürgerbeschwerden
        num_topics: Anzahl der zu extrahierenden Themen (LDA)
        out_dir: Ausgabe-Verzeichnis für Visualisierungen
    
    Returns:
        Exit-Code (0 = Erfolg, 1 = Fehler)
    """
    try:
        # 1. Validiere und erstelle Output-Verzeichnis
        logger.info("="*60)
        logger.info("NLP-ANALYSE: MÄNGELMELDER KONSTANZ")
        logger.info("="*60)
        
        try:
            output_path = sanitize_output_path(out_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output-Verzeichnis: {output_path}")
        except ValueError as e:
            logger.error(f"Ungültiger Output-Pfad: {e}")
            return 1
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Output-Verzeichnisses: {e}")
            return 1

        # 2. Lade Daten
        logger.info(f"Lade Daten aus: {filepath}")
        
        if not Path(filepath).exists():
            logger.error(f"Datei nicht gefunden: {filepath}")
            return 1
        
        texts = load_data_from_csv(filepath)
        
        if not texts:
            logger.error(
                f"Keine Texte aus '{filepath}' geladen. "
                "Prüfe Dateiformat (CSV mit mindestens 3 Spalten, Text in Spalte 3)."
            )
            return 1
        
        logger.info(f"✓ Geladen: {len(texts)} Texte")

        # 3. Vorverarbeitung vorbereiten
        logger.info("\nInitialisiere Vorverarbeitung...")
        stop_words, stop_phrases, lemmatizer = prepare_stopwords_and_lemmatizer()
        logger.info(f"✓ Stoppwörter: {len(stop_words)} Einzelwörter, {len(stop_phrases)} Phrasen")

        # 4. Texte vorverarbeiten
        logger.info("Vorverarbeite Texte...")
        clean_texts = []
        
        for i, text in enumerate(texts):
            cleaned = preprocess_text(text, stop_words, stop_phrases, lemmatizer)
            if cleaned.strip():
                clean_texts.append(cleaned)
            
            # Progress-Anzeige bei vielen Texten
            if (i + 1) % 100 == 0:
                logger.info(f"  Verarbeitet: {i + 1}/{len(texts)}")
        
        if not clean_texts:
            logger.error(
                "Keine Texte nach Vorverarbeitung übrig. "
                "Möglicherweise enthält die Datei nur Stoppwörter."
            )
            return 1
        
        removed = len(texts) - len(clean_texts)
        logger.info(f"✓ Bereinigt: {len(clean_texts)} Texte ({removed} leer/nur Stoppwörter)")

        # 5. Vektorisierung
        logger.info("\n" + "="*60)
        logger.info("VEKTORISIERUNG")
        logger.info("="*60)
        
        (count_matrix, tfidf_matrix, feature_names_count,
         feature_names_tfidf, count_vectorizer, tfidf_vectorizer) = vectorize_texts(clean_texts)
        
        if count_matrix is not None:
            logger.info(f"✓ Count Matrix: {count_matrix.shape}")
        if tfidf_matrix is not None:
            logger.info(f"✓ TF-IDF Matrix: {tfidf_matrix.shape}")

        # 6. Themenextraktion (LDA)
        logger.info("\n" + "="*60)
        logger.info(f"THEMENEXTRAKTION (LDA mit {num_topics} Themen)")
        logger.info("="*60)
        
        topics, lda_model = extract_topics_lda(
            count_matrix,
            count_vectorizer,
            feature_names_count,
            num_topics=num_topics
        )
        
        if topics:
            for idx, words in topics:
                logger.info(f"\n   Thema {idx}: {', '.join(words)}")
        else:
            logger.warning("⚠ LDA konnte keine Themen extrahieren")

        # 7. N-Gramm-Analyse
        logger.info("\n" + "="*60)
        logger.info("N-GRAMM-ANALYSE (Bigramme)")
        logger.info("="*60)
        
        bigrams = extract_ngrams(clean_texts, top_n=10)
        for i, (bg, cnt) in enumerate(bigrams, 1):
            logger.info(f"   {i:2d}. '{bg}' ({cnt}x)")

        # 8. Top-Wörter
        logger.info("\n" + "="*60)
        logger.info("TOP-WÖRTER")
        logger.info("="*60)
        
        top_w = top_words(clean_texts, top_n=15)
        for i, (w, c) in enumerate(top_w, 1):
            logger.info(f"   {i:2d}. '{w}' ({c}x)")

        # 9. Visualisierungen erstellen
        logger.info("\n" + "="*60)
        logger.info("VISUALISIERUNGEN")
        logger.info("="*60)
        
        try:
            visualize_and_save(
                clean_texts,
                count_vectorizer,
                count_matrix,
                lda_model,
                out_dir=str(output_path)
            )
            logger.info(f"✓ Visualisierungen gespeichert in: {output_path}")
        except Exception as e:
            logger.error(f"⚠ Visualisierung fehlgeschlagen: {e}", exc_info=True)

        # 10. Zusammenfassung
        logger.info("\n" + "="*60)
        logger.info("ANALYSE ABGESCHLOSSEN")
        logger.info("="*60)
        logger.info(f"Eingabe: {filepath}")
        logger.info(f"Texte verarbeitet: {len(clean_texts)}")
        logger.info(f"Themen extrahiert: {len(topics) if topics else 0}")
        logger.info(f"Output: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nAnalyse durch Benutzer abgebrochen")
        return 1
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
        return 1


def main():
    """CLI-Einstiegspunkt."""
    parser = argparse.ArgumentParser(
        description='NLP-Analyse von Bürgerbeschwerden (Mängelmelder Konstanz)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard-Analyse
  python -m src.main
  
  # Mit spezifischer CSV-Datei
  python -m src.main data/meine_daten.csv
  
  # Mit 7 Themen und anderem Output-Ordner
  python -m src.main data/daten.csv -t 7 -o results
  
  # Verbose-Modus (mehr Logs)
  python -m src.main -v
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        default='data/maengelmelder_konstanz.csv',
        help='Pfad zur CSV-Datei (Standard: data/maengelmelder_konstanz.csv)'
    )
    
    parser.add_argument(
        '-t', '--topics',
        type=int,
        default=5,
        help='Anzahl der zu extrahierenden Themen (Standard: 5)'
    )
    
    parser.add_argument(
        '-o', '--out',
        default='out',
        help='Ausgabe-Verzeichnis für Visualisierungen (Standard: out)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose-Modus (mehr Logging)'
    )
    
    args = parser.parse_args()
    
    # Setze Logging-Level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validiere Argumente
    if args.topics < 1:
        logger.error("Anzahl der Themen muss mindestens 1 sein")
        sys.exit(1)
    
    if args.topics > 50:
        logger.warning(f"Sehr hohe Anzahl von Themen ({args.topics}). Empfohlen: 3-10")
    
    # Führe Analyse aus
    exit_code = run(args.input, num_topics=args.topics, out_dir=args.out)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()