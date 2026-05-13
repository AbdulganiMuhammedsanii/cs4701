#!/usr/bin/env python3
"""Ablation: claim-level vs response-level aggregation on multi-claim fixtures."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halludet.eval.metrics import classification_metrics, load_jsonl
from halludet.verify.embedding_baseline import EmbeddingBaseline
from halludet.verify.nli_pretrained import NLIClassifier

logger = logging.getLogger(__name__)

BINARY_LABELS = ("supported", "unsupported")


def load_multi_claim_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for r in rows:
        for k in ("claims", "evidence", "label_per_claim", "response_label_binary"):
            if k not in r:
                raise ValueError(f"Row missing {k!r}: {r.get('id')}")
        if len(r["claims"]) != len(r["label_per_claim"]):
            raise ValueError(f"claims/label_per_claim length mismatch for {r.get('id')}")
    return rows


def response_pred_embedding_strict(
    claims: list[str],
    evidence: str,
    emb: EmbeddingBaseline,
    *,
    threshold: float,
) -> str:
    """Response supported iff minimum claim–evidence cosine similarity is at or above threshold."""
    sims = [emb.similarity(c, evidence) for c in claims]
    return "supported" if sims and min(sims) >= threshold else "unsupported"


def response_pred_nli_strict(claims: list[str], evidence: str, nli: NLIClassifier) -> str:
    """Response unsupported if any claim is not three-way supported."""
    for c in claims:
        lab = nli.predict_three_way(c, evidence)
        if lab != "supported":
            return "unsupported"
    return "supported"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Compare claim-level metrics vs response-level binary aggregation (multi-claim JSONL)"
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/fixtures/multi_claim_responses.jsonl"),
        help="JSONL with response, claims, evidence, label_per_claim, response_label_binary",
    )
    p.add_argument("--embedding-threshold", type=float, default=0.35)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--nli-model", type=str, default="typeform/distilbert-base-uncased-mnli")
    p.add_argument("--skip-nli", action="store_true")
    p.add_argument("--output-json", type=Path, default=Path("results/claim_vs_response_ablation.json"))
    args = p.parse_args()

    examples = load_multi_claim_rows(args.data)

    claims_flat: list[str] = []
    evid_flat: list[str] = []
    y_claim: list[str] = []
    for ex in examples:
        ev = ex["evidence"]
        for c, lab in zip(ex["claims"], ex["label_per_claim"]):
            claims_flat.append(c)
            evid_flat.append(ev)
            y_claim.append(lab)

    out: dict[str, Any] = {
        "meta": {
            "data": str(args.data.resolve()),
            "num_examples": len(examples),
            "num_claim_rows": len(claims_flat),
            "embedding_threshold": args.embedding_threshold,
            "response_rules": {
                "embedding": "supported iff min cosine(claim_i, evidence) >= threshold",
                "nli": "supported iff every claim is three-way supported",
            },
        },
        "claim_level": {},
        "response_level": {},
    }

    emb = EmbeddingBaseline(model_name=args.embedding_model)
    y_pred_claim_emb = [
        emb.predict_three_way(c, e, threshold=args.embedding_threshold)
        for c, e in zip(claims_flat, evid_flat)
    ]
    out["claim_level"]["embedding"] = classification_metrics(y_claim, y_pred_claim_emb)

    y_gold_resp = [ex["response_label_binary"] for ex in examples]
    y_pred_resp_emb = [
        response_pred_embedding_strict(ex["claims"], ex["evidence"], emb, threshold=args.embedding_threshold)
        for ex in examples
    ]
    out["response_level"]["embedding"] = classification_metrics(
        y_gold_resp, y_pred_resp_emb, labels=BINARY_LABELS
    )

    if not args.skip_nli:
        nli = NLIClassifier(model_name=args.nli_model)
        y_pred_claim_nli = [nli.predict_three_way(c, e) for c, e in zip(claims_flat, evid_flat)]
        out["claim_level"]["nli"] = classification_metrics(y_claim, y_pred_claim_nli)

        y_pred_resp_nli = [
            response_pred_nli_strict(ex["claims"], ex["evidence"], nli) for ex in examples
        ]
        out["response_level"]["nli"] = classification_metrics(
            y_gold_resp, y_pred_resp_nli, labels=BINARY_LABELS
        )

    def _slim_section(d: dict) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in d.items()}

    slim = {"meta": out["meta"], "claim_level": _slim_section(out["claim_level"]), "response_level": _slim_section(out["response_level"])}
    print(json.dumps(slim, indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
