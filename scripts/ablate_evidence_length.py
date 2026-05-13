#!/usr/bin/env python3
"""Ablation: effect of evidence length (prefix truncation) on embedding and/or NLI metrics."""

from __future__ import annotations

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
from halludet.verify.nli_pretrained import NLIClassifier

logger = logging.getLogger(__name__)


def truncate_evidence_prefix(evidence: str, max_chars: int | None) -> str:
    """Keep the first ``max_chars`` characters of evidence (prefix truncation)."""
    if max_chars is None or max_chars <= 0 or len(evidence) <= max_chars:
        return evidence
    return evidence[:max_chars]


def _slim(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "report"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Evidence length ablation (prefix truncation)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-evidence-chars",
        type=int,
        nargs="+",
        default=[0, 64, 128, 256, 512],
        help="0 = no truncation (full evidence); otherwise prefix length in characters",
    )
    p.add_argument("--embedding-threshold", type=float, default=0.35)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--nli-model", type=str, default="typeform/distilbert-base-uncased-mnli")
    p.add_argument("--skip-nli", action="store_true")
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--output-json", type=Path, default=Path("results/evidence_length_ablation.json"))
    args = p.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    y_true = [r["label"] for r in rows]
    out: dict = {"lengths": []}

    emb = None if args.skip_embedding else EmbeddingBaseline(model_name=args.embedding_model)
    nli = None if args.skip_nli else NLIClassifier(model_name=args.nli_model)

    for max_c in args.max_evidence_chars:
        trunc_label = "full" if max_c is None or max_c <= 0 else max_c
        block: dict = {"max_evidence_chars": trunc_label}
        evidences = [truncate_evidence_prefix(r["evidence"], max_c if max_c > 0 else None) for r in rows]

        if emb is not None:
            y_pred = [
                emb.predict_three_way(c, e, threshold=args.embedding_threshold)
                for c, e in tqdm(
                    zip([r["claim"] for r in rows], evidences),
                    desc=f"embedding_len_{trunc_label}",
                    total=len(rows),
                )
            ]
            block["embedding"] = _slim(classification_metrics(y_true, y_pred))

        if nli is not None:
            y_pred = [
                nli.predict_three_way(c, e)
                for c, e in tqdm(
                    zip([r["claim"] for r in rows], evidences),
                    desc=f"nli_len_{trunc_label}",
                    total=len(rows),
                )
            ]
            block["nli"] = _slim(classification_metrics(y_true, y_pred))

        out["lengths"].append(block)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
