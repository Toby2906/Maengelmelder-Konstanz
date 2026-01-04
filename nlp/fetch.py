"""Daten laden (CSV)."""
import csv
from typing import List
from nlp.utils import HAS_PANDAS

if HAS_PANDAS:
    import pandas as pd


def load_data_csv_module(filepath: str) -> List[str]:
    texts = []
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                if len(row) > 2:
                    text = row[2].strip()
                    if text:
                        texts.append(text)
    except FileNotFoundError:
        print(f"Fehler: Datei '{filepath}' nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim CSV-Lesen: {e}")
    return texts


def load_data_from_csv(filepath: str) -> List[str]:
    """Lade Texte aus CSV; verwende pandas wenn verfügbar, sonst csv-Modul."""
    texts = []
    if HAS_PANDAS:
        try:
            df = pd.read_csv(filepath, sep=',', quotechar='"', dtype=str)
            if len(df.columns) > 2:
                texts = df.iloc[:, 2].dropna().astype(str).tolist()
            else:
                print("Warnung: Datei hat weniger als 3 Spalten")
        except Exception as e:
            print(f"Fehler beim Laden mit pandas: {e}")
            texts = load_data_csv_module(filepath)
    else:
        texts = load_data_csv_module(filepath)
    return texts
