"""Approach 3: fine-tuned sequence classifier for claim–evidence verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from halludet.verify.checkpoint_utils import resolve_classifier_checkpoint_dir

logger = logging.getLogger(__name__)

InternalLabel = Literal["supported", "contradicted", "not_supported"]

LABEL2ID = {"supported": 0, "contradicted": 1, "not_supported": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class FineTunedVerifier:
    def __init__(self, checkpoint_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        root = Path(checkpoint_dir)
        resolved = resolve_classifier_checkpoint_dir(root)
        if resolved is None:
            raise ValueError(
                f"No classifier weights (model.safetensors / pytorch_model.bin) under {checkpoint_dir}"
            )
        if resolved.resolve() != root.resolve():
            logger.info("Loading fine-tuned weights from %s", resolved)
        load_dir = str(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.model.to(self.device)
        self.model.eval()
        self.num_labels = int(getattr(self.model.config, "num_labels", 3))

    @torch.inference_mode()
    def predict_binary(self, claim: str, evidence: str) -> Literal["supported", "unsupported"]:
        if self.num_labels != 2:
            raise ValueError("predict_binary requires a 2-label checkpoint")
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
            if "support" in name and "unsupport" not in name:
                return "supported"
            if "unsupport" in name or "not_support" in name:
                return "unsupported"
        if pred == 0:
            return "supported"
        return "unsupported"

    @torch.inference_mode()
    def predict_three_way(self, claim: str, evidence: str) -> InternalLabel:
        if self.num_labels == 2:
            raise ValueError("Use predict_binary for 2-label checkpoints")
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
            # Order matters: "not_supported" contains the substring "support".
            if "contrad" in name or name == "contradicted":
                return "contradicted"
            if "not_support" in name or "unsupport" in name:
                return "not_supported"
            if name == "supported" or "entail" in name:
                return "supported"
            if "support" in name:
                return "supported"
            return "not_supported"
        return ID2LABEL.get(pred, "not_supported")
