"""Fine-tune RoBERTa for claim–evidence verification (Approach 3): three-way or binary."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

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

LABEL2ID_THREE: dict[str, int] = {"supported": 0, "contradicted": 1, "not_supported": 2}
ID2LABEL_THREE: dict[int, str] = {v: k for k, v in LABEL2ID_THREE.items()}

LABEL2ID_BINARY: dict[str, int] = {"supported": 0, "unsupported": 1}
ID2LABEL_BINARY: dict[int, str] = {0: "supported", 1: "unsupported"}

Task = Literal["three_way", "binary"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _training_label(raw: str, task: Task) -> str:
    if task == "binary":
        return "supported" if raw == "supported" else "unsupported"
    return raw


class ClaimEvidenceDataset(Dataset):
    """Tokenizes one claim–evidence pair per item; padding is applied in the collator."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, task: Task):
        self.rows = rows
        self.tokenizer = tokenizer
        self.task = task
        self.label2id = LABEL2ID_BINARY if task == "binary" else LABEL2ID_THREE

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
        key = _training_label(r["label"], self.task)
        enc["labels"] = self.label2id[key]
        return enc


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Fine-tune verification classifier on processed FEVER JSONL")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument(
        "--task",
        type=str,
        choices=["three_way", "binary"],
        default="three_way",
        help="three_way: 3-class; binary: supported vs unsupported (contradicted+not_supported→unsupported)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: checkpoints/verify_roberta (three_way) or checkpoints/verify_roberta_binary (binary)",
    )
    p.add_argument("--base-model", type=str, default="roberta-base")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    task: Task = args.task  # type: ignore[assignment]
    if args.out_dir is None:
        args_out = Path(
            "checkpoints/verify_roberta_binary" if task == "binary" else "checkpoints/verify_roberta"
        )
    else:
        args_out = Path(args.out_dir)

    train_rows = _read_jsonl(args.train_jsonl)
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]

    eval_rows: list[dict[str, Any]] | None = None
    if args.eval_jsonl and args.eval_jsonl.is_file():
        eval_rows = _read_jsonl(args.eval_jsonl)
        if args.max_samples:
            eval_rows = eval_rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_ds = ClaimEvidenceDataset(train_rows, tokenizer, task)
    eval_ds = ClaimEvidenceDataset(eval_rows, tokenizer, task) if eval_rows else None
    data_collator = DataCollatorWithPadding(tokenizer)

    if task == "binary":
        num_labels = 2
        id2label = ID2LABEL_BINARY
        label2id = LABEL2ID_BINARY
    else:
        num_labels = 3
        id2label = ID2LABEL_THREE
        label2id = LABEL2ID_THREE

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

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
    logger.info("Saved %s model to %s", task, args_out)


if __name__ == "__main__":
    main()
