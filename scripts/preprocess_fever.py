#!/usr/bin/env python3
"""CLI: preprocess FEVER (download wiki, build index, emit JSONL)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halludet.data.preprocess import main

if __name__ == "__main__":
    main()
