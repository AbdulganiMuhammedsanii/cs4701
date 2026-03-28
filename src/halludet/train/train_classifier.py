"""Fine-tune RoBERTa for 3-way claim–evidence verification (Approach 3)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

LABEL2ID = {"supported": 0, "contradicted": 1, "not_supported": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ClaimEvidenceDataset(Dataset):
    """Tokenizes one claim–evidence pair per item; padding is applied in the collator."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows[i]
        enc = self.tokenizer(
            r["evidence"],
            r["claim"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        enc["labels"] = LABEL2ID[r["label"]]
        return enc


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Fine-tune verification classifier on processed FEVER JSONL")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/verify_roberta"))
    p.add_argument("--base-model", type=str, default="roberta-base")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    train_rows = _read_jsonl(args.train_jsonl)
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]

    eval_rows: list[dict[str, Any]] | None = None
    if args.eval_jsonl and args.eval_jsonl.is_file():
        eval_rows = _read_jsonl(args.eval_jsonl)
        if args.max_samples:
            eval_rows = eval_rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_ds = ClaimEvidenceDataset(train_rows, tokenizer)
    eval_ds = ClaimEvidenceDataset(eval_rows, tokenizer) if eval_rows else None
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args_out = Path(args.out_dir)
    args_out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args_out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="accuracy" if eval_ds is not None else None,
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(args_out))
    tokenizer.save_pretrained(str(args_out))
    logger.info("Saved model to %s", args_out)


if __name__ == "__main__":
    main()
