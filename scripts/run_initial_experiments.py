#!/usr/bin/env python3
"""Initial experiments: embedding threshold sweep + NLI (+ optional fine-tuned)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from halludet.eval.metrics import classification_metrics, load_jsonl
from halludet.verify.embedding_baseline import EmbeddingBaseline
from halludet.verify.finetuned import FineTunedVerifier
from halludet.verify.nli_pretrained import NLIClassifier

logger = logging.getLogger(__name__)


def _slim(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "report"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Run initial verification experiments")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Cosine thresholds for embedding baseline (ablation)",
    )
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--nli-model", type=str, default="typeform/distilbert-base-uncased-mnli")
    p.add_argument("--finetuned-dir", type=Path, default=Path("checkpoints/verify_roberta"))
    p.add_argument("--skip-nli", action="store_true")
    p.add_argument("--output-json", type=Path, default=Path("results/initial_experiments.json"))
    args = p.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    y_true = [r["label"] for r in rows]
    n = len(rows)

    out: dict = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "data": str(args.data.resolve()),
            "num_examples": n,
            "embedding_model": args.embedding_model,
            "nli_model": args.nli_model,
        },
        "embedding_threshold_sweep": [],
    }

    logger.info("Loading embedding model (one load, many thresholds)…")
    emb = EmbeddingBaseline(model_name=args.embedding_model)
    sims = [emb.similarity(r["claim"], r["evidence"]) for r in tqdm(rows, desc="embedding_sims")]

    for t in args.thresholds:
        y_pred = [
            "not_supported" if s < t else "supported" for s in sims
        ]
        m = classification_metrics(y_true, y_pred)
        out["embedding_threshold_sweep"].append({"threshold": t, **_slim(m)})

    if not args.skip_nli:
        logger.info("Pretrained NLI…")
        nli = NLIClassifier(model_name=args.nli_model)
        y_pred = [nli.predict_three_way(r["claim"], r["evidence"]) for r in tqdm(rows, desc="nli")]
        out["nli_pretrained"] = _slim(classification_metrics(y_true, y_pred))

    ft_dir = args.finetuned_dir
    if ft_dir.is_dir() and (ft_dir / "config.json").is_file():
        logger.info("Fine-tuned verifier…")
        ft = FineTunedVerifier(str(ft_dir))
        y_pred = [
            ft.predict_three_way(r["claim"], r["evidence"]) for r in tqdm(rows, desc="finetuned")
        ]
        out["finetuned"] = _slim(classification_metrics(y_true, y_pred))
    else:
        out["finetuned"] = None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)

    best = max(out["embedding_threshold_sweep"], key=lambda x: x.get("f1_macro", 0))
    print("Best embedding sweep (by macro F1):", json.dumps(best, indent=2))
    if "nli_pretrained" in out:
        print("NLI:", json.dumps(out["nli_pretrained"], indent=2))


if __name__ == "__main__":
    main()
