#!/usr/bin/env python3
"""Evaluate embedding baseline, pretrained NLI, and optional fine-tuned model on processed JSONL."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from halludet.eval.metrics import classification_metrics, load_jsonl
from halludet.verify.embedding_baseline import EmbeddingBaseline
from halludet.verify.finetuned import FineTunedVerifier
from halludet.verify.nli_pretrained import NLIClassifier

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Evaluate verification methods")
    p.add_argument("--data", type=Path, required=True, help="Processed JSONL (claim, evidence, label)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--embedding-threshold", type=float, default=0.35)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--nli-model",
        type=str,
        default="typeform/distilbert-base-uncased-mnli",
    )
    p.add_argument(
        "--finetuned-dir",
        type=Path,
        default=Path("checkpoints/verify_roberta"),
        help="If present, run fine-tuned model",
    )
    p.add_argument("--skip-nli", action="store_true", help="Skip slow transformer NLI")
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    y_true = [r["label"] for r in rows]

    results: dict = {}

    if not args.skip_embedding:
        logger.info("Embedding baseline…")
        emb = EmbeddingBaseline(model_name=args.embedding_model)
        y_pred = [
            emb.predict_three_way(r["claim"], r["evidence"], threshold=args.embedding_threshold)
            for r in tqdm(rows, desc="embedding")
        ]
        results["embedding"] = classification_metrics(y_true, y_pred)
        print("=== Embedding baseline ===")
        print(
            json.dumps(
                {k: v for k, v in results["embedding"].items() if k != "report"},
                indent=2,
            )
        )
        print(results["embedding"]["report"])

    if not args.skip_nli:
        logger.info("Pretrained NLI…")
        nli = NLIClassifier(model_name=args.nli_model)
        y_pred = [nli.predict_three_way(r["claim"], r["evidence"]) for r in tqdm(rows, desc="nli")]
        results["nli"] = classification_metrics(y_true, y_pred)
        print("=== Pretrained NLI ===")
        print(json.dumps({k: v for k, v in results["nli"].items() if k != "report"}, indent=2))
        print(results["nli"]["report"])

    ft_dir = args.finetuned_dir
    if ft_dir.is_dir() and (ft_dir / "config.json").is_file():
        logger.info("Fine-tuned verifier…")
        ft = FineTunedVerifier(str(ft_dir))
        y_pred = [
            ft.predict_three_way(r["claim"], r["evidence"]) for r in tqdm(rows, desc="finetuned")
        ]
        results["finetuned"] = classification_metrics(y_true, y_pred)
        print("=== Fine-tuned ===")
        print(json.dumps({k: v for k, v in results["finetuned"].items() if k != "report"}, indent=2))
        print(results["finetuned"]["report"])
    else:
        logger.info("Skipping fine-tuned (no checkpoint at %s)", ft_dir)

    if args.output_json:
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in results.items()}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)


if __name__ == "__main__":
    main()
