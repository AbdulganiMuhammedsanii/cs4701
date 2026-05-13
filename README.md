# cs4701 — Neural hallucination / fact verification

CS 4701 final project: **claim-level** verification of LLM outputs against retrieved evidence using (1) embedding similarity, (2) pretrained NLI, and (3) a fine-tuned RoBERTa classifier, evaluated on FEVER-style claim–evidence data.

## Layout

- `[src/halludet/data/](src/halludet/data/)` — FEVER loading (HuggingFace `datasets` 2.x), wiki-pages download/index, JSONL preprocessing
- `[src/halludet/claims/](src/halludet/claims/)` — sentence segmentation + lightweight rule filtering (`extract_claims`)
- `[src/halludet/verify/](src/halludet/verify/)` — embedding baseline, pretrained NLI, fine-tuned verifier
- `[src/halludet/train/](src/halludet/train/)` — `Trainer` fine-tuning for **three-way** or **binary** (supported vs unsupported) heads
- `[src/halludet/eval/](src/halludet/eval/)` — accuracy / precision / recall / F1 via `sklearn`, optional **paired bootstrap** comparisons
- `[scripts/](scripts/)` — CLI entrypoints (evaluate, train, ablations, significance)
- `[data/fixtures/sample_processed.jsonl](data/fixtures/sample_processed.jsonl)` — tiny file for quick smoke tests without downloading FEVER wiki
- `[data/fixtures/multi_claim_responses.jsonl](data/fixtures/multi_claim_responses.jsonl)` — toy multi-claim “responses” for claim-vs-response ablation

## Proposal mapping (course write-up ↔ code)


| Proposal item                                          | Where it lives                                                                                                                                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sentence segmentation + rule-based claim extraction    | `[src/halludet/claims/extract.py](src/halludet/claims/extract.py)`                                                                                                                                          |
| Approach 1: embedding + cosine + threshold             | `[src/halludet/verify/embedding_baseline.py](src/halludet/verify/embedding_baseline.py)`; threshold sweep `[scripts/run_initial_experiments.py](scripts/run_initial_experiments.py)`                        |
| Approach 2: pretrained NLI                             | `[src/halludet/verify/nli_pretrained.py](src/halludet/verify/nli_pretrained.py)`                                                                                                                            |
| Approach 3: fine-tuned verifier (3-way or binary)      | `[src/halludet/train/train_classifier.py](src/halludet/train/train_classifier.py)`, `[src/halludet/verify/finetuned.py](src/halludet/verify/finetuned.py)`                                                  |
| Metrics (accuracy, P/R/F1)                             | `[src/halludet/eval/metrics.py](src/halludet/eval/metrics.py)`                                                                                                                                              |
| Binary “hallucination” view (supported vs unsupported) | `collapse_to_supported_vs_unsupported` in metrics; `--report-binary` on `[scripts/evaluate_all.py](scripts/evaluate_all.py)` and `[scripts/run_initial_experiments.py](scripts/run_initial_experiments.py)` |
| Threshold sensitivity                                  | `[scripts/run_initial_experiments.py](scripts/run_initial_experiments.py)` `--thresholds`                                                                                                                   |
| Evidence length ablation                               | `[scripts/ablate_evidence_length.py](scripts/ablate_evidence_length.py)` (prefix truncation of evidence)                                                                                                    |
| Claim-level vs response-level ablation                 | `[scripts/ablate_claim_vs_response.py](scripts/ablate_claim_vs_response.py)` + multi-claim fixture                                                                                                          |
| Paired bootstrap significance (accuracy deltas)        | `[src/halludet/eval/significance.py](src/halludet/eval/significance.py)`, `[scripts/compare_models_significance.py](scripts/compare_models_significance.py)`                                                |


**Binary collapse rule (for reports):** `supported` stays supported; `contradicted` and `not_supported` map to **unsupported** (single “not adequately supported / hallucinated” bucket for aggregate metrics).

**Embedding limitation:** cosine similarity cannot distinguish **contradiction** from **neutral**; the baseline only splits low vs high similarity into `not_supported` vs `supported` (see embedding module docstring).

**Bootstrap note:** `compare_models_significance.py` resamples **paired** examples and reports a percentile **95% interval** for the difference in accuracy. It does **not** apply multiple-comparison correction; use Bonferroni (or similar) in the paper if you compare many pairs.

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

**Smoke test without the full wiki download:** point `--raw-dir` at `[data/fixtures/wiki_shard](data/fixtures/wiki_shard)` and pass `--skip-wiki-download` so only that shard is indexed (most rows will be skipped for missing pages; this still exercises the pipeline).

## 2) Fine-tune the hallucination / verification classifier (optional)

**Three-way** (supported / contradicted / not_supported) — default `--out-dir` is `checkpoints/verify_roberta`:

```bash
python scripts/train_detector.py \
  --train-jsonl data/processed/fever_paper_dev.jsonl \
  --eval-jsonl data/processed/fever_labelled_dev.jsonl \
  --task three_way \
  --out-dir checkpoints/verify_roberta \
  --epochs 1 \
  --batch-size 8 \
  --max-samples 5000
```

**Binary** (supported vs unsupported, with `contradicted` and `not_supported` merged at training time) — default `--out-dir` is `checkpoints/verify_roberta_binary` when `--out-dir` is omitted:

```bash
python scripts/train_detector.py \
  --train-jsonl data/processed/fever_paper_dev.jsonl \
  --eval-jsonl data/processed/fever_labelled_dev.jsonl \
  --task binary \
  --epochs 1 \
  --batch-size 8 \
  --max-samples 5000
```

## 3) Evaluate baselines vs NLI vs fine-tuned

Add `**--report-binary**` to also print **supported vs unsupported** metrics (three-way predictions collapsed with the same rule as training binary labels).

```bash
python scripts/evaluate_all.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --embedding-threshold 0.35 \
  --finetuned-dir checkpoints/verify_roberta \
  --report-binary \
  --output-json results/metrics.json
```

Fine-tuned checkpoints are loaded only when `**config.json` and model weights** (`model.safetensors` or `pytorch_model.bin`) are present; otherwise the fine-tuned block is skipped with a log message.

Quick test without downloads:

```bash
python scripts/evaluate_all.py --data data/fixtures/sample_processed.jsonl --max-samples 10
```

**Initial experiments** (embedding cosine threshold sweep + NLI + optional fine-tuned, JSON log):

```bash
python scripts/run_initial_experiments.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --report-binary \
  --output-json results/initial_experiments.json
```

**Evidence length ablation** (prefix truncation of evidence; reuses one embedding / one NLI load across lengths):

```bash
python scripts/ablate_evidence_length.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --max-evidence-chars 0 128 256 512 \
  --output-json results/evidence_length_ablation.json
```

**Claim vs response ablation** (multi-claim fixture; response rules are printed in the output `meta`):

```bash
python scripts/ablate_claim_vs_response.py \
  --data data/fixtures/multi_claim_responses.jsonl \
  --output-json results/claim_vs_response_ablation.json
```

**Paired bootstrap** on per-example accuracy (subset of models; requires at least two that succeed):

```bash
python scripts/compare_models_significance.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --models embedding,nli,finetuned \
  --n-bootstrap 10000 \
  --seed 0 \
  --output-json results/bootstrap_accuracy.json
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