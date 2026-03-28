# cs4701 — Neural hallucination / fact verification

CS 4701 final project: **claim-level** verification of LLM outputs against retrieved evidence using (1) embedding similarity, (2) pretrained NLI, and (3) a fine-tuned RoBERTa classifier, evaluated on FEVER-style claim–evidence data.

## Layout

- [`src/halludet/data/`](src/halludet/data/) — FEVER loading (HuggingFace `datasets` 2.x), wiki-pages download/index, JSONL preprocessing
- [`src/halludet/claims/`](src/halludet/claims/) — sentence segmentation + lightweight rule filtering (`extract_claims`)
- [`src/halludet/verify/`](src/halludet/verify/) — embedding baseline, pretrained NLI, fine-tuned verifier
- [`src/halludet/train/`](src/halludet/train/) — `Trainer` fine-tuning for 3-way labels
- [`src/halludet/eval/`](src/halludet/eval/) — accuracy / precision / recall / F1 via `sklearn`
- [`scripts/`](scripts/) — CLI entrypoints
- [`data/fixtures/sample_processed.jsonl`](data/fixtures/sample_processed.jsonl) — tiny file for quick smoke tests without downloading FEVER wiki

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** The HuggingFace FEVER loader requires `datasets>=2.14,<3` (2.x still supports the FEVER dataset script).

## 1) Preprocess FEVER (wiki + claims + evidence text)

Downloads **wiki-pages.zip** (~1.7 GB) from [fever.ai/download/fever/wiki-pages.zip](https://fever.ai/download/fever/wiki-pages.zip), builds a SQLite sentence index, and writes JSONL with resolved evidence sentences.

Default indexing is **subset**: only Wikipedia pages needed for the chosen splits / `--max-samples` (much faster than indexing the full dump). Use `--full-wiki-index` for the complete corpus.

```bash
python scripts/preprocess_fever.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --splits paper_dev \
  --max-samples 2000 \
  --rebuild-wiki-index
```

Outputs: `data/processed/fever_paper_dev.jsonl`, `data/processed/wiki_sentences.sqlite`.

**Smoke test without the full wiki download:** point `--raw-dir` at [`data/fixtures/wiki_shard`](data/fixtures/wiki_shard) and pass `--skip-wiki-download` so only that shard is indexed (most rows will be skipped for missing pages; this still exercises the pipeline).

## 2) Fine-tune the hallucination / verification classifier (optional)

```bash
python scripts/train_detector.py \
  --train-jsonl data/processed/fever_paper_dev.jsonl \
  --eval-jsonl data/processed/fever_labelled_dev.jsonl \
  --out-dir checkpoints/verify_roberta \
  --epochs 1 \
  --batch-size 8 \
  --max-samples 5000
```

## 3) Evaluate baselines vs NLI vs fine-tuned

```bash
python scripts/evaluate_all.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --embedding-threshold 0.35 \
  --finetuned-dir checkpoints/verify_roberta \
  --output-json results/metrics.json
```

Quick test without downloads:

```bash
python scripts/evaluate_all.py --data data/fixtures/sample_processed.jsonl --max-samples 10
```

**Initial experiments** (embedding cosine threshold sweep + NLI + optional fine-tuned, JSON log):

```bash
python scripts/run_initial_experiments.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --output-json results/initial_experiments.json
```

Metrics include accuracy, macro/weighted precision, recall, and F1 (see `halludet.eval.metrics.classification_metrics`).

## Claim extraction (LLM output → atomic claims)

```python
from halludet.claims import extract_claims

claims = extract_claims("Paris is the capital of France. It is known for food. Yes.")
```

## Labels

- **supported** ← FEVER `SUPPORTS`
- **contradicted** ← FEVER `REFUTES`
- **not_supported** ← FEVER `NOT ENOUGH INFO`

Pretrained NLI maps **entailment → supported**, **contradiction → contradicted**, **neutral → not_supported** (per project spec). The default checkpoint is `typeform/distilbert-base-uncased-mnli`; you can pass `--nli-model FacebookAI/roberta-large-mnli` (or another MNLI model) to `evaluate_all.py` for a stronger baseline. The embedding baseline only uses cosine similarity, so it cannot detect contradiction; it labels low-similarity pairs as `not_supported` and higher similarity as `supported`.
