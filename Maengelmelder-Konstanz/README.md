# Mängelmelder Konstanz - NLP Analyse

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

NLP-Analyse von Bürgerbeschwerden mit Topic Modeling.

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m src.main
```

## Features

- Topic Modeling (LDA)
- Visualisierungen
- N-Gramm Analyse
- Tests & CI/CD

## Verwendung

```bash
python -m src.main data.csv -t 5 -o out
```

## Entwicklung

```bash
pip install -r requirements-dev.txt
pytest --cov=src
```

## Lizenz

MIT - siehe LICENSE
