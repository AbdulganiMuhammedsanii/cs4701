# Neural Hallucination Detection: Claim-Level Verification of LLM Outputs Against Retrieved Evidence

**Course:** CS 4701 — Practicum in Artificial Intelligence
**Semester:** Spring 2026
**Submission date:** May 13, 2026

---

## Team

- Abdulgani Muhammedsani — NetID: `[am-NETID]`
- `[Teammate 2 Name]` — NetID: `[NETID]`
- `[Teammate 3 Name]` — NetID: `[NETID]`

## AI Keywords

Natural language inference (NLI); fact verification; sentence embeddings; transformer fine-tuning; RoBERTa; FEVER; hallucination detection; claim extraction; classification metrics; paired bootstrap significance.

## Application Setting

Verifying the factual grounding of large-language-model (LLM) outputs against a retrieval corpus. Given a free-form model response and a body of trusted evidence (here, Wikipedia sentences indexed from the FEVER 1.0 wiki-pages dump), the system decomposes the response into atomic claims and, for each claim, predicts whether the retrieved evidence **supports** it, **contradicts** it, or provides **not enough information** (the “hallucination” bucket in user-facing terms). The application target is post-hoc auditing of LLM answers — for example, attaching a per-sentence support badge in a chat UI, or flagging suspect spans for human review in a documentation or customer-support pipeline.

---

# Part 1: Project Description

## 1.1 What we planned

The original proposal framed the project as a three-approach study of LLM-output hallucination detection at the **claim level**, not the response level. Concretely, we planned to (i) take a free-form LLM response, (ii) segment it into atomic factual claims with a lightweight sentence segmenter plus rule filter, (iii) pair each claim with retrieved evidence from a known-good corpus, and (iv) judge each claim with three families of verifiers of increasing capacity: a similarity-only embedding baseline, a pretrained NLI model used zero-shot, and a fine-tuned transformer trained on FEVER-style claim–evidence pairs. The headline questions were how much each approach contributes, whether the three-way label set (supported / contradicted / not_supported) gives meaningfully more information than a binary supported-vs-unsupported view, and how robust the methods are to short or truncated evidence.

We also committed in the proposal to a real evaluation rather than only qualitative demos: per-class precision, recall, F1, macro and weighted averages, a paired-bootstrap confidence interval on accuracy differences between approaches, and at least one ablation that varied an input the systems should plausibly be sensitive to — evidence length.

## 1.2 What the project actually became

The final system, `halludet`, is a small Python package organised around four sub-modules that closely mirror the proposal: `halludet.data` for FEVER loading, wiki indexing and JSONL preprocessing; `halludet.claims` for sentence segmentation and simple rule filtering; `halludet.verify` for the three verifiers; and `halludet.eval` for metrics and paired-bootstrap significance. A `scripts/` layer wraps each piece of functionality in a CLI entry point so the report's numbers can be regenerated end-to-end from the command line without touching Python.

We made three substantive changes relative to the proposal during implementation. First, the FEVER preprocessor exports rows in a flattened "one (claim, evidence sentence) per row" form rather than the nested per-claim multi-evidence form FEVER ships in. This makes the file directly trainable by Hugging Face `Trainer` without a custom collator and is closer to the contract the verifiers themselves expect. Second, we added an explicit binary head and an explicit binary collapse rule for reporting — `supported` stays supported, while both `contradicted` and `not_supported` map to a single `unsupported` bucket. The two heads share data preparation and metrics code, so the binary view costs essentially no engineering and gives a cleaner number for the "is this claim hallucinated?" question a deployment would actually ask. Third, the most important correction came late in the cycle: our first three-way run found that the fine-tuned classifier never predicted `not_supported` at all. Investigation showed that FEVER's `NOT ENOUGH INFO` rows ship with an empty `evidence_wiki_url` field (all 4,524 NEI rows in the `paper_dev` split, in our inspection), and the preprocessor was silently dropping any row for which the wiki lookup returned `None`. The training data therefore had two classes, not three, and the third logit was untrained. We fixed this by extending the wiki index with a `random_sentence` method and assigning each NEI row a uniformly sampled distractor sentence as its evidence. The result is a three-way dataset with 1,144 / 916 / 863 examples for `not_supported / supported / contradicted` on the 5,000-row prefix of `paper_dev`, and a fine-tuned head that now predicts all three classes. Section 2 below is explicit about how this change shapes our headline numbers and what they should and should not be read as evidence for.

## 1.3 The three core AI components

**Approach 1 — Embedding baseline (`halludet.verify.embedding_baseline.EmbeddingBaseline`).** The simplest verifier embeds the claim and evidence independently with `sentence-transformers/all-MiniLM-L6-v2`, computes cosine similarity between the L2-normalised embeddings, and thresholds: similarities below `τ` are labelled `not_supported`, similarities above are labelled `supported`. Crucially, cosine similarity is symmetric and undirected, so this approach **cannot** identify `contradicted` claims — a contradiction is topically related and will sit at high similarity. In the three-way report the contradiction class is therefore expected to look essentially random, and we view embedding alone as a strong "is this evidence even on-topic" signal rather than a hallucination classifier in the strict sense.

**Approach 2 — Pretrained NLI (`halludet.verify.nli_pretrained.NLIClassifier`).** The second verifier treats verification as natural language inference. Evidence is the premise, claim is the hypothesis, and we use a public MNLI-finetuned transformer (`typeform/distilbert-base-uncased-mnli` by default; the CLI accepts any MNLI-trained Hugging Face checkpoint, including `FacebookAI/roberta-large-mnli`). The model's three softmax outputs — entailment, neutral, contradiction — are mapped to `supported`, `not_supported`, and `contradicted` respectively. This is the cleanest off-the-shelf zero-shot baseline for the task, since MNLI's label space is structurally identical to ours; the cost is that MNLI premises are usually one or two sentences of similar style to the hypothesis, while FEVER evidence is wiki sentences that are often long and stylistically distinct.

**Approach 3 — Fine-tuned RoBERTa (`halludet.train.train_classifier`, `halludet.verify.finetuned.FineTunedVerifier`).** The third verifier is the heaviest: we initialise from `roberta-base` and add a 3-way (or 2-way, for the binary head) classification layer, then fine-tune on the processed FEVER JSONL using Hugging Face `Trainer` with cross-entropy loss, AdamW, learning rate 2e-5, batch size 8, and 2 epochs over the ~3,000 row training file. The tokeniser receives the (evidence, claim) pair as a sentence-pair input with `truncation=True, max_length=512`, padding handled by `DataCollatorWithPadding`. The checkpoint is saved to `checkpoints/verify_roberta` for the 3-way head and `checkpoints/verify_roberta_binary` for the binary head; the evaluation script loads either by directory and skips gracefully if no checkpoint is present, so the embedding and NLI baselines can always be run on a fresh clone without first having to train.

## 1.4 The data pipeline

`scripts/preprocess_fever.py` is the entry point that turns the raw FEVER release into the JSONL files the verifiers consume. It downloads the official `wiki-pages.zip` (~1.7 GB) on first run, extracts it into `data/raw/wiki-pages`, then builds a SQLite index (`data/processed/wiki_sentences.sqlite`) keyed by (normalised page title, sentence id) so that the FEVER `evidence_wiki_url` / `evidence_sentence_id` pairs can be resolved to actual sentence text in O(1). Indexing is "subset by default" — only the page titles that appear in the chosen FEVER split (and within `--max-samples`) are inserted, which compresses what would otherwise be a ~30-minute full-corpus index into a fast subset build. The preprocessor then iterates rows, looks up evidence text, and writes a JSONL with `id`, `claim`, `evidence`, `fever_label`, and the internal label (`supported` / `contradicted` / `not_supported`). The fix described in Section 1.2 — assigning random distractor sentences to NEI rows so the third class is actually represented — lives at this layer and is seeded for reproducibility.

## 1.5 Claim extraction

`halludet.claims.extract_claims` is intentionally light: NLTK's `punkt` sentence tokeniser (with a regex fallback if `punkt` cannot be downloaded) splits the response into sentences, and a short rule filter drops very short fragments (< 12 chars), boilerplate openers ("However,", "Therefore,", "In conclusion,", "Note:"), and pure acknowledgements ("yes", "no", "ok", "sure", "thanks"). This is deliberately not a full OpenIE or dependency-parse pipeline — the proposal scoped claim extraction to "sentence segmentation + light rule filter" so that the experimental focus stayed on the verification stage. The multi-claim fixture in `data/fixtures/multi_claim_responses.jsonl` exercises this stage in the response-vs-claim ablation, where we compare verifying each extracted claim independently against verifying the entire response as a single text.

---

# Part 2: Evaluation

## 2.1 Questions we asked

The evaluation was organised around five specific questions, all answered on the processed FEVER `paper_dev` split unless otherwise noted:

1. **Headline question.** How do the three approaches compare on the three-way verification task in terms of accuracy and per-class F1, and how does that comparison change when we collapse to the binary "supported vs unsupported" view that a deployment would actually surface?
2. **Threshold sensitivity.** The embedding baseline has one free parameter, the cosine threshold. How sharply does its three-way and binary performance change as we sweep that threshold from 0.25 to 0.50?
3. **Evidence-length ablation.** If we truncate evidence to a prefix of `k` characters, do the verifiers degrade gracefully (i.e. the longer the evidence the better) or do they have a sweet spot? This is a proxy for the realistic deployment case where retrieval returns truncated snippets.
4. **Granularity.** Does verifying at the claim level — using `extract_claims` to break a response into sentences and judging each independently — actually disagree with verifying the whole response as a single text, and if so in which direction?
5. **Significance.** Are the accuracy differences we report between approaches large enough to survive a paired bootstrap on the same evaluation set?

## 2.2 How we answered them

All numbers in this section come from the CLI scripts shipped in `scripts/`, run on a 500-example prefix of `data/processed/fever_paper_dev.jsonl` (≈ the first 500 rows of the regenerated three-class file produced by the fix in Section 1.2). `scripts/evaluate_all.py` produces the headline three-way and binary table by iterating each verifier over the same rows. `scripts/run_initial_experiments.py` sweeps the embedding threshold across `[0.25, 0.30, 0.35, 0.40, 0.45, 0.50]` and emits a single JSON log. `scripts/ablate_evidence_length.py` evaluates the embedding and NLI verifiers on `max_evidence_chars ∈ {full, 64, 128, 256, 512}` while reusing one model load across lengths. `scripts/ablate_claim_vs_response.py` evaluates against the multi-claim fixture. `scripts/compare_models_significance.py` resamples paired per-example correctness vectors `n_bootstrap = 10,000` times with seed 0 and reports a percentile 95 % interval for each pair of models. Metrics are computed via `sklearn` through `halludet.eval.metrics.classification_metrics`, which returns accuracy, macro and weighted precision/recall/F1, and an `sklearn.classification_report` table that we read directly off the console.

## 2.3 Q1 — Headline performance

Table 1 collects the headline numbers from `results/metrics.json`. Three-way accuracies on the same 500-row evaluation are 0.552 (embedding, `τ = 0.35`), 0.570 (pretrained NLI, DistilBERT-MNLI), and 0.964 (fine-tuned RoBERTa). In the binary collapse the spread tightens slightly but the ordering is preserved: 0.688, 0.766, 0.968. The embedding baseline behaves exactly as the design predicts — it scores reasonably on the binary task, where its job reduces to "are these two sentences on the same topic", but collapses on the three-way task because contradictions look identical to support under cosine similarity. NLI does much better on the binary task than on the three-way one, which is consistent with the well-known difficulty of the MNLI **neutral** class, especially when premise style (long encyclopaedia sentences) drifts from MNLI's training distribution. The fine-tuned RoBERTa head is the clear winner on both views, with macro-F1 of 0.964 (three-way) and 0.964 (binary), and balanced per-class behaviour: in the three-way confusion table from our run, the per-class F1 was 0.95 for `supported` (n=168), 0.95 for `contradicted` (n=165), and 0.99 for `not_supported` (n=167).

![Figure 1 — Headline accuracy & macro-F1 across verifiers](results/figures/fig1_headline.svg)

**Figure 1.** Accuracy and macro-F1 for the three verifiers on both the three-way and binary-collapse label sets, computed on a 500-row prefix of `data/processed/fever_paper_dev.jsonl`. The fine-tuned RoBERTa head dominates on every column; the embedding baseline only becomes competitive once the contradicted/not_supported classes are merged in the binary view.

## 2.4 Q2 — Threshold sensitivity for the embedding baseline

The threshold sweep in `results/initial_experiments.json` shows the expected pattern: the embedding baseline's binary accuracy is essentially flat in the 0.35–0.50 range (0.688, 0.688, 0.698, 0.694) and the best point is `τ = 0.45` at 0.698, while the three-way accuracy is monotonically decreasing across the same range (0.552 → 0.512 → 0.528 → 0.486). The three-way drop is mechanical: as `τ` rises, more examples cross from `supported` to `not_supported`, but since contradiction is invisible to cosine similarity, this only redistributes errors between the two non-contradicted classes; nothing the threshold can do recovers the contradicted class. The lesson we draw is that the binary collapse is the **right** evaluation for an embedding-only system; reporting embedding accuracy on the three-way task is a category error baked into the model's information access, not a tuning failure.

![Figure 2 — Embedding baseline accuracy vs cosine threshold τ](results/figures/fig2_threshold.svg)

**Figure 2.** Embedding-baseline accuracy as a function of the cosine-similarity threshold τ. The binary-collapse curve is flat-to-slightly-rising and peaks at τ = 0.45 (0.698 accuracy), while the three-way curve falls monotonically across the range — additional precision in the threshold cannot recover the contradicted class because cosine similarity does not encode polarity.

## 2.5 Q3 — Evidence-length ablation

`results/evidence_length_ablation.json` reports embedding and NLI performance under prefix truncation of evidence to 64, 128, 256, 512 characters and the untruncated "full" condition. For embeddings the three-way accuracy is essentially length-invariant in the 256-and-up regime (0.552, 0.552, 0.554 at 256, full, 512), drops modestly at 128 (0.540), and degrades to 0.512 at 64 characters. NLI is more brittle: three-way accuracy is 0.470 at 64 characters, 0.516 at 128, 0.560 at 256, 0.574 at 512, and 0.570 untruncated. The qualitative reading is that NLI is the more language-sensitive of the two baselines — it needs enough premise to form a real inference — while embeddings are dominated by topic-level overlap that is recoverable from short text. The practical implication for a retrieval-augmented deployment is that returning very short snippets (sub-128 character) primarily hurts the NLI stage, and that the fine-tuned head should be re-evaluated at the same truncations before claiming length-robustness; we did not have time to re-run the fine-tuned model under truncation, and we flag this as a follow-up.

![Figure 3 — Three-way accuracy vs max evidence chars](results/figures/fig3_length.svg)

**Figure 3.** Three-way accuracy under prefix truncation of evidence to 64, 128, 256, 512 characters and the untruncated `full` condition. NLI drops sharply at sub-128-character truncations while the embedding baseline degrades far more gracefully, consistent with embeddings relying on topic overlap that survives truncation and NLI needing enough premise to form a real inference.

## 2.6 Q4 — Claim vs response granularity

The `ablate_claim_vs_response.py` ablation runs each verifier in two modes on the toy multi-claim fixture: per-extracted-claim (using `extract_claims`) and whole-response (using the raw response as the "claim" input). The ablation's purpose is qualitative — the fixture is small, so we are not reporting a confidence interval here — but the observed pattern is consistent across approaches: a response that mixes a supported sentence with an unsupported one is labelled `supported` at the response level by both the NLI and fine-tuned classifiers (the supported sentence dominates the input), and only the claim-level decomposition surfaces the hallucinated span. This is exactly the failure mode the proposal predicted; it justifies the project's claim-level framing rather than a response-level one, and it is what we point at when arguing that decomposition is a load-bearing design choice and not just a preprocessing step.

![Figure 4 — Claim-level vs response-level verification](results/figures/fig4_claim_vs_response.svg)

**Figure 4.** Accuracy and macro-F1 for the embedding and NLI verifiers, computed at the claim level (per-extracted-claim decision) and at the response level (whole-response decision) on `data/fixtures/multi_claim_responses.jsonl`. Both verifiers gain when the response is decomposed before classification — claim-level accuracy is 0.667 vs 0.5 at the response level for each — which directly supports the project's claim-level framing.

## 2.7 Q5 — Paired bootstrap significance

`scripts/compare_models_significance.py` produces a paired-bootstrap 95 % interval for each pair of models, resampling the same per-example correctness vectors `n_bootstrap = 10,000` times with seed 0. With the fine-tuned classifier at 0.964 accuracy and the two baselines at 0.552 and 0.570, the differences `finetuned − embedding ≈ 0.412` and `finetuned − nli ≈ 0.394` sit well outside any plausible bootstrap interval; the embedding-vs-NLI difference of 0.018 is the only comparison that lands close enough to zero to be a serious question of significance, and we expect its interval to straddle zero on this slice. We do not apply a Bonferroni correction across the three pairwise comparisons; the README is explicit about this and we recommend it for any extension to more model variants. The bootstrap therefore confirms what the headline numbers already suggest — fine-tuning dominates both baselines at any reasonable significance level on this evaluation — and warns that the small embedding-vs-NLI gap should not be reported as a real ordering.

![Figure 5 — Paired bootstrap 95% CI for accuracy differences](results/figures/fig5_bootstrap.svg)

**Figure 5.** Paired-bootstrap mean accuracy difference and 95 % percentile interval for each pair of models (n=10,000 resamples, seed 0, 500 paired examples). Both `embedding − finetuned` (Δ ≈ −0.414, CI [−0.460, −0.370]) and `nli − finetuned` (Δ ≈ −0.396, CI [−0.442, −0.350]) sit far from zero, while `embedding − nli` (Δ ≈ −0.018, CI [−0.084, +0.046]) straddles zero — the two baselines are not separable on this slice.

## 2.8 Honest caveats

Three caveats are worth stating up front because they shape how the headline accuracy should be read. First, the 0.964 three-way accuracy for the fine-tuned head is on the same `paper_dev` slice that the model was trained on; we did not have a held-out preprocessed split at evaluation time, so this number should be read as an in-distribution upper bound rather than a generalisation estimate. A clean way to recover a proper held-out number would be to preprocess `labelled_dev` with the same NEI-distractor fix and re-score; this is a small change to the pipeline and a planned follow-up. Second, the NEI distractor for `not_supported` rows is sampled uniformly from the wiki sentence index, so the distractor is almost always topically far from the claim. The classifier therefore has an easy job on the `not_supported` class — discriminating "off-topic random sentence" from "real evidence" is much easier than discriminating "topically related but uninformative" from real evidence, which is what a deployed system would face. The 0.99 per-class F1 for `not_supported` should be interpreted accordingly; a stronger evaluation would replace random distractors with top-k retrieved-but-non-supporting sentences, and we expect that to materially compress the gap between fine-tuned and NLI on the third class. Third, the evidence-length ablation only covers the two non-fine-tuned approaches, so any claim that the fine-tuned head is "robust to truncation" would be unsupported on our data; we restrict our length claims to embedding and NLI.

## 2.9 What we conclude

On the questions we set ourselves the picture is straightforward. Fine-tuning a transformer end-to-end on FEVER-style claim–evidence pairs is by a large margin the strongest of the three approaches, both on the three-way task and on the binary "is this claim hallucinated" task that maps more directly to the application setting. The embedding baseline is a reasonable on-topic detector but structurally cannot identify contradictions and is best read through the binary collapse. Pretrained NLI is a credible zero-shot starting point but is sensitive to short evidence and conflates neutral with contradiction more often than the task allows. The evidence-length ablation shows that the baselines need at least ~128 characters of premise to behave, and the response-vs-claim ablation confirms that the claim-level framing of the proposal is doing real work — a response-level system would miss the very hallucinations a per-claim system catches. We treat our 0.964 in-distribution number as evidence that the architecture and data path are sound, not as a generalisation claim, and we name the held-out evaluation and the harder-distractor sampling as the two extensions that would let us defend a stronger statement.

---

## References

### Technical papers and primary references

- Thorne, J., Vlachos, A., Christodoulopoulos, C., and Mittal, A. (2018). *FEVER: A Large-scale Dataset for Fact Extraction and VERification.* NAACL. — The dataset and label scheme this project is built on.
- Williams, A., Nangia, N., and Bowman, S. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MNLI).* NAACL. — The training distribution for the pretrained NLI baseline.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692. — Base model for Approach 3.
- Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. — Methodology behind the embedding baseline.
- Manakul, P., Liusie, A., and Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP. — Reference for the broader hallucination-detection problem.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall. — Methodology for the paired-bootstrap significance analysis.

### Software resources

- Hugging Face `transformers`, `datasets`, and `tokenizers` libraries (used for `roberta-base`, `Trainer`, dataset loading).
- `sentence-transformers` (`all-MiniLM-L6-v2` checkpoint) for the embedding baseline.
- `typeform/distilbert-base-uncased-mnli` and `FacebookAI/roberta-large-mnli` for the pretrained NLI baseline.
- `scikit-learn` for `precision_score`, `recall_score`, `f1_score`, `classification_report`.
- `NumPy` for the bootstrap resampler.
- `NLTK` (`punkt`, `punkt_tab`) for sentence segmentation in `extract_claims`.
- `SQLite` (via the Python stdlib) for the wiki-sentence index.

### Data resources

- FEVER 1.0 (`paper_dev`, `labelled_dev`, `train` splits) loaded via Hugging Face `datasets` v2.x with `trust_remote_code=True`.
- Official FEVER `wiki-pages.zip` (~1.7 GB) from `fever.ai/download/fever/wiki-pages.zip`, used to resolve `evidence_wiki_url` + `evidence_sentence_id` to actual sentence text.
- Project fixtures under `data/fixtures/` (`sample_processed.jsonl`, `multi_claim_responses.jsonl`, `wiki_shard/`) for offline smoke tests and the response-vs-claim ablation.

---
