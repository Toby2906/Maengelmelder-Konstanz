#!/usr/bin/env python3
"""Legacy entrypoint — ruft `src.main.run()` auf."""
from .main import run


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/maengelmelder_konstanz.csv'
    sys.exit(run(path))
