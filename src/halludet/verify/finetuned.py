"""Approach 3: fine-tuned sequence classifier for claim–evidence verification."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

InternalLabel = Literal["supported", "contradicted", "not_supported"]

LABEL2ID = {"supported": 0, "contradicted": 1, "not_supported": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class FineTunedVerifier:
    def __init__(self, checkpoint_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_three_way(self, claim: str, evidence: str) -> InternalLabel:
        enc = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        pred = int(logits.argmax(dim=-1).item())
        config_id2label = getattr(self.model.config, "id2label", None) or {}
        if pred in config_id2label:
            name = str(config_id2label[pred]).lower()
            if "support" in name or name == "supported":
                return "supported"
            if "contrad" in name or name == "contradicted":
                return "contradicted"
            return "not_supported"
        return ID2LABEL.get(pred, "not_supported")
