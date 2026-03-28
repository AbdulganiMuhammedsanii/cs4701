"""Approach 2: pretrained transformer NLI (entailment / neutral / contradiction)."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

InternalLabel = Literal["supported", "contradicted", "not_supported"]


class NLIClassifier:
    """
    Maps MNLI-style labels to verification labels:
    - entailment → supported
    - contradiction → contradicted
    - neutral → not_supported (hallucinated / ungrounded per project spec)
    """

    def __init__(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.id2label = {int(k): str(v) for k, v in cfg.id2label.items()}

    @torch.inference_mode()
    def predict_nli_label(self, premise: str, hypothesis: str) -> str:
        """Raw NLI label string (model-dependent casing)."""
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        pred = int(logits.argmax(dim=-1).item())
        return str(self.id2label[pred])

    def predict_three_way(self, claim: str, evidence: str) -> InternalLabel:
        premise, hypothesis = evidence, claim
        raw = self.predict_nli_label(premise, hypothesis).lower()
        if "entail" in raw:
            return "supported"
        if "contrad" in raw:
            return "contradicted"
        return "not_supported"
