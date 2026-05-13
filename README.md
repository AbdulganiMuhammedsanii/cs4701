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

**Paths:** run every `python scripts/...` command with the shell **current working directory** set to the **repository root** (the `cs4701` folder), so `data/...` and `results/...` resolve correctly.

## 1) Preprocess FEVER (wiki + claims + evidence text)

Downloads **wiki-pages.zip** (~1.7 GB) from [fever.ai/download/fever/wiki-pages.zip](https://fever.ai/download/fever/wiki-pages.zip), builds a SQLite sentence index, and writes JSONL with resolved evidence sentences.

Default indexing is **subset**: only Wikipedia pages needed for the chosen splits / `--max-samples` (much faster than indexing the full dump). Use `--full-wiki-index` for the complete corpus.

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/preprocess_fever.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --splits paper_dev \
  --max-samples 2000 \
  --rebuild-wiki-index
```

Outputs (for that command): `data/processed/fever_paper_dev.jsonl`, `data/processed/wiki_sentences.sqlite`.

**Also export `labelled_dev`** (for held-out–style eval vs training on `paper_dev` only). The wiki index must include evidence pages for **both** splits if you use subset indexing, so pass **both** split names when rebuilding the index:

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/preprocess_fever.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --splits paper_dev labelled_dev \
  --rebuild-wiki-index \
  --skip-wiki-download
```

That writes `data/processed/fever_labelled_dev.jsonl` (and refreshes `fever_paper_dev.jsonl`). Omit `--skip-wiki-download` the first time if wiki shards are not extracted yet.

**Smoke test without the full wiki download:** point `--raw-dir` at [`data/fixtures/wiki_shard`](data/fixtures/wiki_shard) and pass `--skip-wiki-download` so only that shard is indexed (most rows will be skipped for missing pages; this still exercises the pipeline).

## 2) Fine-tune the hallucination / verification classifier (optional)

**Three-way** (supported / contradicted / not_supported) — default `--out-dir` is `checkpoints/verify_roberta`:

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/train_detector.py \
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
source .venv/bin/activate && PYTHONPATH=src python scripts/train_detector.py \
  --train-jsonl data/processed/fever_paper_dev.jsonl \
  --eval-jsonl data/processed/fever_labelled_dev.jsonl \
  --task binary \
  --epochs 1 \
  --batch-size 8 \
  --max-samples 5000
```

## 3) Evaluate baselines vs NLI vs fine-tuned

Run commands with your shell **current directory set to this repo root** (`cs4701`), the venv active, and **`PYTHONPATH=src`** so `import halludet` works (all examples below use this).

Add **`--report-binary`** to also print **supported vs unsupported** metrics (three-way predictions collapsed with the same rule as the binary training labels).

**Typical split usage:** train on `fever_paper_dev.jsonl` (see §2); report **final** numbers on **`fever_labelled_dev.jsonl`** so `--data` is not the training file. If you pass `--eval-jsonl data/processed/fever_labelled_dev.jsonl` during `Trainer` fine-tuning, that split was still used for **early stopping / `load_best_model_at_end`**—state that in the write-up; it is not the same as training on those rows, but it is weaker than a third, untouched test split.

**Paper dev** (subset / debugging):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_all.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --embedding-threshold 0.35 \
  --finetuned-dir checkpoints/verify_roberta \
  --report-binary \
  --output-json results/metrics_paper_dev.json
```

**Labelled dev** (common choice for final tables when training on `paper_dev` only):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_all.py \
  --data data/processed/fever_labelled_dev.jsonl \
  --embedding-threshold 0.5 \
  --finetuned-dir checkpoints/verify_roberta \
  --report-binary \
  --output-json results/metrics_labelled_dev.json
```

Fine-tuned checkpoints are loaded only when **`config.json` and model weights** (`model.safetensors` or `pytorch_model.bin`, or sharded `model-*.safetensors`) are present; otherwise the fine-tuned block is skipped with a log message.

**What `--output-json` saves:** aggregate metrics per method only (no per-example predictions, no copy of the dataset). The `results/` directory is **gitignored**; JSON files still live on disk for your report—commit them elsewhere if your course requires artifacts in the repo.

Quick test without downloads:

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/evaluate_all.py \
  --data data/fixtures/sample_processed.jsonl \
  --max-samples 10
```

**Initial experiments** (embedding cosine threshold sweep + NLI + optional fine-tuned, JSON log):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/run_initial_experiments.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --report-binary \
  --output-json results/initial_experiments.json
```

**Choosing `--embedding-threshold`:** the script prints “Best embedding sweep (by **macro F1** on the **three-way** embedding predictions).” For a **binary** (supported vs unsupported) story, open `results/initial_experiments.json` and compare `embedding_threshold_sweep[].binary.f1_macro` (or accuracy) across `threshold` values; pick the threshold that maximizes the metric you report, then pass it as `--embedding-threshold` to `evaluate_all.py`. Re-run the sweep if you change `--data`, `--max-samples`, or the embedding model.

**Evidence length ablation** (prefix truncation of evidence; reuses one embedding / one NLI load across lengths):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/ablate_evidence_length.py \
  --data data/processed/fever_paper_dev.jsonl \
  --max-samples 500 \
  --max-evidence-chars 0 128 256 512 \
  --output-json results/evidence_length_ablation.json
```

**Claim vs response ablation** (multi-claim fixture; response rules are printed in the output `meta`):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/ablate_claim_vs_response.py \
  --data data/fixtures/multi_claim_responses.jsonl \
  --output-json results/claim_vs_response_ablation.json
```

**Paired bootstrap** on per-example accuracy (subset of models; requires at least two that succeed):

```bash
source .venv/bin/activate && PYTHONPATH=src python scripts/compare_models_significance.py \
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