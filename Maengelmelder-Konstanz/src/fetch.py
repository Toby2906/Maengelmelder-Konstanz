"""Daten laden (CSV).

Unterstützt pandas (bevorzugt) und csv-Modul als Fallback.
"""
import csv
import logging
from typing import List, Optional, Generator
from pathlib import Path
from .utils import HAS_PANDAS

if HAS_PANDAS:
    import pandas as pd

logger = logging.getLogger(__name__)


def load_data_csv_module(filepath: str) -> List[str]:
    """Lade Texte aus CSV mit Python's csv-Modul.
    
    Erwartet CSV mit mindestens 3 Spalten, Text in Spalte 3 (Index 2).
    
    Args:
        filepath: Pfad zur CSV-Datei
    
    Returns:
        Liste von Texten (ohne leere Strings)
    """
    texts = []
    
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            
            # Optional: Erste Zeile als Header überspringen
            try:
                header = next(reader)
                logger.debug(f"CSV Header: {header}")
            except StopIteration:
                logger.warning(f"CSV-Datei ist leer: {filepath}")
                return []
            
            for row_num, row in enumerate(reader, start=2):  # Start bei 2 (nach Header)
                if len(row) > 2:
                    text = row[2].strip()
                    if text:
                        texts.append(text)
                elif len(row) > 0:
                    logger.warning(f"Zeile {row_num}: Weniger als 3 Spalten ({len(row)} Spalten)")
                    
    except FileNotFoundError:
        logger.error(f"Datei nicht gefunden: {filepath}")
    except UnicodeDecodeError as e:
        logger.error(f"Encoding-Fehler in {filepath}: {e}. Versuche anderes Encoding.")
    except Exception as e:
        logger.error(f"Fehler beim CSV-Lesen mit csv-Modul: {e}", exc_info=True)
    
    return texts


def load_data_from_csv(filepath: str, text_column: int = 2) -> List[str]:
    """Lade Texte aus CSV; verwende pandas wenn verfügbar, sonst csv-Modul.
    
    Args:
        filepath: Pfad zur CSV-Datei
        text_column: Index der Spalte mit Texten (0-basiert, Standard: 2)
    
    Returns:
        Liste von Texten (ohne leere Strings)
    """
    texts = []
    
    if not Path(filepath).exists():
        logger.error(f"Datei nicht gefunden: {filepath}")
        return []
    
    if HAS_PANDAS:
        try:
            # Versuche mit verschiedenen Encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(
                        filepath,
                        sep=',',
                        quotechar='"',
                        dtype=str,
                        encoding=encoding
                    )
                    logger.info(f"CSV erfolgreich geladen mit {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("Konnte CSV mit keinem Standard-Encoding laden")
                return load_data_csv_module(filepath)
            
            # Validiere Spaltenanzahl
            if df.shape[1] <= text_column:
                logger.error(
                    f"CSV hat nur {df.shape[1]} Spalten, "
                    f"benötigt mindestens {text_column + 1} für Textextraktion"
                )
                return []
            
            # Extrahiere Texte aus angegebener Spalte
            texts = df.iloc[:, text_column].dropna().astype(str).tolist()
            
            # Entferne leere Strings
            texts = [t.strip() for t in texts if t.strip()]
            
            logger.info(
                f"pandas: {len(texts)} Texte aus Spalte {text_column} "
                f"({df.shape[0]} Zeilen total)"
            )
            
        except Exception as e:
            logger.warning(f"Fehler beim Laden mit pandas: {e}")
            logger.info("Fallback auf csv-Modul...")
            texts = load_data_csv_module(filepath)
    else:
        logger.info("pandas nicht verfügbar, verwende csv-Modul")
        texts = load_data_csv_module(filepath)
    
    if not texts:
        logger.warning(
            f"Keine Texte aus '{filepath}' geladen. "
            "Prüfe ob Datei korrekt formatiert ist (CSV mit Text in Spalte 3)."
        )
    
    return texts


def load_data_from_csv_chunked(
    filepath: str,
    text_column: int = 2,
    chunksize: int = 1000
) -> Generator[str, None, None]:
    """Generator: Lade große CSV-Dateien in Chunks (nur mit pandas).
    
    Nützlich für sehr große Dateien um Memory-Verbrauch zu reduzieren.
    
    Args:
        filepath: Pfad zur CSV-Datei
        text_column: Index der Spalte mit Texten (0-basiert)
        chunksize: Anzahl Zeilen pro Chunk
    
    Yields:
        Einzelne Texte
    
    Example:
        >>> for text in load_data_from_csv_chunked('large_file.csv'):
        ...     process(text)
    """
    if not HAS_PANDAS:
        logger.error("Chunked loading benötigt pandas. Installiere: pip install pandas")
        return
    
    if not Path(filepath).exists():
        logger.error(f"Datei nicht gefunden: {filepath}")
        return
    
    try:
        chunk_iterator = pd.read_csv(
            filepath,
            sep=',',
            quotechar='"',
            dtype=str,
            encoding='utf-8',
            chunksize=chunksize
        )
        
        total_texts = 0
        for chunk_num, chunk in enumerate(chunk_iterator, start=1):
            if chunk.shape[1] <= text_column:
                logger.warning(
                    f"Chunk {chunk_num}: Nur {chunk.shape[1]} Spalten, "
                    f"benötige mindestens {text_column + 1}"
                )
                continue
            
            texts = chunk.iloc[:, text_column].dropna().astype(str)
            
            for text in texts:
                text = text.strip()
                if text:
                    total_texts += 1
                    yield text
            
            if chunk_num % 10 == 0:
                logger.debug(f"Verarbeitet: {chunk_num} Chunks, {total_texts} Texte")
        
        logger.info(f"Chunked loading abgeschlossen: {total_texts} Texte")
        
    except Exception as e:
        logger.error(f"Fehler beim Chunked Loading: {e}", exc_info=True)


def validate_csv_structure(filepath: str, expected_columns: int = 3) -> bool:
    """Validiere CSV-Struktur vor dem Laden.
    
    Args:
        filepath: Pfad zur CSV-Datei
        expected_columns: Erwartete Mindestanzahl von Spalten
    
    Returns:
        True wenn Struktur valide ist, sonst False
    """
    if not Path(filepath).exists():
        logger.error(f"Datei nicht gefunden: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            if len(header) < expected_columns:
                logger.error(
                    f"CSV hat nur {len(header)} Spalten, "
                    f"erwartet mindestens {expected_columns}"
                )
                return False
            
            logger.info(f"CSV-Struktur valide: {len(header)} Spalten")
            logger.debug(f"Header: {header}")
            return True
            
    except Exception as e:
        logger.error(f"Fehler bei CSV-Validierung: {e}")
        return False