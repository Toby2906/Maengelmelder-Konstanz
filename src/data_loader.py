"""Daten-Loader für CSV-Dateien.

Lädt Bürgerbeschwerden aus CSV mit Multi-Encoding-Support.
"""
import csv
import logging
from pathlib import Path
from typing import List, Optional

from .config import Config

logger = logging.getLogger(__name__)

# Prüfe ob pandas verfügbar ist
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.info("pandas nicht verfügbar - verwende csv-Modul")


class DataLoader:
    """Lädt Textdaten aus CSV-Dateien."""
    
    def __init__(self, filepath: Optional[str] = None):
        """Initialisiere DataLoader.
        
        Args:
            filepath: Pfad zur CSV-Datei (optional)
        """
        self.filepath = Path(filepath) if filepath else Config.DEFAULT_CSV_PATH
        self.texts: List[str] = []
    
    def load(self) -> List[str]:
        """Lade Texte aus CSV.
        
        Returns:
            Liste von Texten
        """
        if not self.filepath.exists():
            logger.error(f"Datei nicht gefunden: {self.filepath}")
            return []
        
        logger.info(f"Lade Daten aus: {self.filepath}")
        
        if HAS_PANDAS:
            self.texts = self._load_with_pandas()
        else:
            self.texts = self._load_with_csv()
        
        if self.texts:
            logger.info(f"✓ {len(self.texts)} Texte geladen")
        else:
            logger.warning("⚠ Keine Texte geladen")
        
        return self.texts
    
    def _load_with_pandas(self) -> List[str]:
        """Lade mit pandas (bevorzugt).
        
        Returns:
            Liste von Texten
        """
        # Versuche verschiedene Encodings
        encodings = [Config.CSV_ENCODING, "latin-1", "iso-8859-1", "cp1252"]
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    self.filepath,
                    sep=Config.CSV_DELIMITER,
                    encoding=encoding,
                    dtype=str
                )
                logger.debug(f"CSV geladen mit {encoding} encoding")
                
                # Prüfe Spaltenanzahl
                if df.shape[1] <= Config.CSV_TEXT_COLUMN:
                    logger.error(
                        f"CSV hat nur {df.shape[1]} Spalten, "
                        f"benötige mindestens {Config.CSV_TEXT_COLUMN + 1}"
                    )
                    return []
                
                # Extrahiere Texte
                texts = (
                    df.iloc[:, Config.CSV_TEXT_COLUMN]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                
                # Filtere leere Texte
                texts = [t for t in texts if t]
                
                logger.debug(f"pandas: {len(texts)} Texte aus {df.shape[0]} Zeilen")
                return texts
                
            except UnicodeDecodeError:
                logger.debug(f"Encoding {encoding} fehlgeschlagen, probiere nächstes...")
                continue
            except Exception as e:
                logger.error(f"Fehler beim Laden mit pandas: {e}")
                return []
        
        logger.error("Konnte CSV mit keinem Encoding laden")
        return []
    
    def _load_with_csv(self) -> List[str]:
        """Lade mit csv-Modul (Fallback).
        
        Returns:
            Liste von Texten
        """
        texts = []
        
        try:
            with open(
                self.filepath,
                newline="",
                encoding=Config.CSV_ENCODING
            ) as f:
                reader = csv.reader(
                    f,
                    delimiter=Config.CSV_DELIMITER
                )
                
                # Überspringe Header
                try:
                    header = next(reader)
                    logger.debug(f"CSV Header: {header}")
                except StopIteration:
                    logger.warning("CSV ist leer")
                    return []
                
                # Lese Daten
                for row_num, row in enumerate(reader, start=2):
                    if len(row) > Config.CSV_TEXT_COLUMN:
                        text = row[Config.CSV_TEXT_COLUMN].strip()
                        if text:
                            texts.append(text)
                    elif len(row) > 0:
                        logger.debug(
                            f"Zeile {row_num}: Nur {len(row)} Spalten"
                        )
            
            logger.debug(f"csv-Modul: {len(texts)} Texte geladen")
            return texts
            
        except Exception as e:
            logger.error(f"Fehler beim CSV-Lesen: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Gib Statistiken zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        if not self.texts:
            return {}
        
        text_lengths = [len(t) for t in self.texts]
        word_counts = [len(t.split()) for t in self.texts]
        
        return {
            "total_texts": len(self.texts),
            "avg_length": sum(text_lengths) / len(text_lengths),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths),
            "avg_words": sum(word_counts) / len(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }
    
    def save_processed(self, texts: List[str], filename: str = "processed.csv"):
        """Speichere verarbeitete Texte.
        
        Args:
            texts: Liste von Texten
            filename: Dateiname
        """
        output_path = Config.PROCESSED_DATA_DIR / filename
        
        try:
            if HAS_PANDAS:
                df = pd.DataFrame({"text": texts})
                df.to_csv(output_path, index=False, encoding="utf-8")
            else:
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["text"])
                    for text in texts:
                        writer.writerow([text])
            
            logger.info(f"✓ Verarbeitete Texte gespeichert: {output_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")


def load_data(filepath: Optional[str] = None) -> List[str]:
    """Convenience-Funktion zum Laden von Daten.
    
    Args:
        filepath: Pfad zur CSV-Datei
    
    Returns:
        Liste von Texten
    """
    loader = DataLoader(filepath)
    return loader.load()