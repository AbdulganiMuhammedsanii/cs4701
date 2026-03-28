#!/usr/bin/env python3
"""Fine-tune the claim–evidence verifier (RoBERTa 3-way)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halludet.train.train_classifier import main

if __name__ == "__main__":
    main()
