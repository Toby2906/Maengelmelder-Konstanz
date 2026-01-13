#!/usr/bin/env python3
"""Legacy-Einstiegspunkt für NLP-Analyse.

Dieser Einstiegspunkt ist veraltet. Verwende stattdessen:
    python -m src.main

Wird aus Kompatibilitätsgründen beibehalten.
"""
import sys
from pathlib import Path

# Sicherstellen, dass src im Path ist für Imports
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.main import run


def main():
    """Legacy-Einstiegspunkt - ruft src.main.run() auf."""
    import warnings
    warnings.warn(
        "nlp_analyse.py ist veraltet. Verwende stattdessen: python -m src.main",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Parse einfache CLI-Argumente
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/maengelmelder_konstanz.csv'
    
    # Führe Hauptanalyse aus
    exit_code = run(path)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()