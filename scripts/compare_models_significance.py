#!/usr/bin/env python3
"""Paired bootstrap on per-example accuracy for comparing verification models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tqdm import tqdm

from halludet.eval.metrics import load_jsonl
from halludet.eval.significance import pairwise_bootstrap_accuracy, per_example_correct
from halludet.verify.checkpoint_utils import checkpoint_has_classifier_weights
from halludet.verify.embedding_baseline import EmbeddingBaseline
from halludet.verify.finetuned import FineTunedVerifier
from halludet.verify.nli_pretrained import NLIClassifier

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Bootstrap accuracy differences between models")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--embedding-threshold", type=float, default=0.35)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--nli-model", type=str, default="typeform/distilbert-base-uncased-mnli")
    p.add_argument("--finetuned-dir", type=Path, default=Path("checkpoints/verify_roberta"))
    p.add_argument(
        "--models",
        type=str,
        default="embedding,nli,finetuned",
        help="Comma-separated subset of: embedding,nli,finetuned",
    )
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    y_true = [r["label"] for r in rows]

    want = {s.strip() for s in args.models.split(",") if s.strip()}
    correct_by_model: dict[str, np.ndarray] = {}

    if "embedding" in want:
        logger.info("Embedding predictions…")
        emb = EmbeddingBaseline(model_name=args.embedding_model)
        y_pred = [
            emb.predict_three_way(r["claim"], r["evidence"], threshold=args.embedding_threshold)
            for r in tqdm(rows, desc="embedding")
        ]
        correct_by_model["embedding"] = per_example_correct(y_true, y_pred)

    if "nli" in want:
        logger.info("NLI predictions…")
        nli = NLIClassifier(model_name=args.nli_model)
        y_pred = [nli.predict_three_way(r["claim"], r["evidence"]) for r in tqdm(rows, desc="nli")]
        correct_by_model["nli"] = per_example_correct(y_true, y_pred)

    if "finetuned" in want:
        ft_dir = args.finetuned_dir
        if (
            ft_dir.is_dir()
            and (ft_dir / "config.json").is_file()
            and checkpoint_has_classifier_weights(ft_dir)
        ):
            logger.info("Fine-tuned predictions…")
            ft = FineTunedVerifier(str(ft_dir))
            from halludet.eval.metrics import collapse_to_supported_vs_unsupported

            if ft.num_labels == 2:
                yt = collapse_to_supported_vs_unsupported(y_true)
                y_pred = [
                    ft.predict_binary(r["claim"], r["evidence"]) for r in tqdm(rows, desc="finetuned")
                ]
            else:
                yt = y_true
                y_pred = [
                    ft.predict_three_way(r["claim"], r["evidence"])
                    for r in tqdm(rows, desc="finetuned")
                ]
            correct_by_model["finetuned"] = per_example_correct(yt, y_pred)
        else:
            logger.warning("Skipping finetuned (missing checkpoint or weights at %s)", ft_dir)

    if len(correct_by_model) < 2:
        raise SystemExit("Need at least two successful models for comparison; check --models and checkpoints.")

    out = {
        "meta": {
            "data": str(args.data.resolve()),
            "n_examples": len(rows),
            "models": list(correct_by_model.keys()),
        },
        "bootstrap": pairwise_bootstrap_accuracy(
            correct_by_model,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        ),
    }
    print(json.dumps(out, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
