# Pre-registered analysis plan: expert sealed test set v1

> **English translation (unofficial).**
>
> - **Source:** the Chinese file `预注册分析计划.md` in the Stage 1 registration on OSF (https://osf.io/gj48z), registered on 2026-10-26 at 09:17 UTC.
> - **SHA-256 of the Chinese original:** `68b9e7353295cf9107a5ec07f89d257fcd0f8bcde62a3b35150c87c9d2b2c47c`. The original is the authoritative version.
> - **Purpose:** the translation was prepared on 2026-11-10, after all runs, for reviewers and readers. It adds nothing to the registered content.
> - **Conventions:** translator's notes are given in square brackets. "Gold labels" (金标) translates the original term for the frozen reference labels.
> - **Annotators:** this plan foresaw expert annotators. The labels were in fact produced and adjudicated by LLM agents, as recorded in the implementation protocol deposited with the same registration (`stage1_implementation_protocol_en.md`, Section 1 and deviation D1).

This plan must be finalised and time-stamped **before any model scores the sealed candidate pools**, for example by uploading it to OSF or committing it to a git repository with time records.
Any change after finalisation is entered in the "Deviation log" at the end and reported in the paper.

## 1. Data and freezing order

1. Phase 1 adjudication is completed and `final_questions.json` is generated; its SHA256 is recorded.
2. The candidate pools `sealed_pools.jsonl` are generated on the server and their SHA256 is recorded:
   - the BM25 top 20 for each question;
   - retrieval uses the question text only;
   - no reference answer is inserted.
3. Phase 2 adjudication is completed and `phase2_gold_labels.jsonl` is generated; its SHA256 is recorded.
4. Models may be run **only after these three files are frozen**. Each model and configuration is run once, and all results are reported.

## 2. Methods compared

All methods rerank **the same 20-candidate pool** and select the top 2 for each question. The evidence input is the title-plus-context version, identical to the text shown during annotation.

| Method | Description |
|---|---|
| Support-CE | Existing mDeBERTa model, seeds 42/2026/3407 |
| Relevance-CE | As above, but trained with relevance labels |
| BM25 original ranking | Baseline without learning |
| bge-reranker-v2-m3 | Zero-shot, strong public baseline |
| Qwen3-Reranker-0.6B | Zero-shot, strong public baseline |
| (Optional) LLM point-wise judgement | E.g., Qwen2.5-7B judging zero-shot "whether the passage fully supports a given point" |

Model checkpoints and the budget (Top-2) are all fixed. No parameter is tuned on the sealed set, and no threshold is used.

## 3. Primary hypothesis and primary metric

**H1:** On the sealed set, Support-CE has higher Top-2 coverage of required points than Relevance-CE.

**Metric definition:**

- Coverage: the union of full support (gold label = 2) of the question's required points by the 2 selected passages, divided by the number of points of the question.
- Average first over the 3 seeds within each question, then over all questions.

**Test:**

- Compute the mean difference.
- Paired bootstrap over questions, 10,000 resamples; report the 95% CI.
- If the lower bound is above 0, H1 is considered supported; otherwise it is reported as not supported.

## 4. Secondary analyses

All of the following comparisons are corrected for multiple comparisons with the Holm method.

- Coverage differences of Support-CE against BM25, bge-reranker-v2-m3 and Qwen3-Reranker-0.6B.
- Full coverage: the proportion of questions whose points are all fully supported.
- Selected-non-supporting rate: the proportion of selected passages that do not fully support any point.
- Ranking quality: in-pool nDCG@10. The gain is the maximum support score of the passage over the points: 2, 1 or 0.
- Pool upper bound: coverage when all 20 passages are selected, to show the recall bottleneck.
- Partial-support criterion: coverage when "partial support" is counted, as a sensitivity analysis.

## 5. Stratification (descriptive, no significance conclusions)

- Question type: diagnosis-explanation questions and treatment-principle questions.
- Source stratum: new cases never processed before (fresh_med) and held-out cases from the original test set (heldout_med).
- Evidence source: textbooks, classical texts, other case records.
- Variation between seeds: report each seed's result and the standard deviation.

## 6. Annotation quality report

- Phase 1:
  - κ of the point judgements;
  - proportions of programme-drafted points that were kept, modified or deleted;
  - number of excluded questions and the reasons.
- Phase 2:
  - Cohen's κ or Krippendorff's α of the primary-metric label (full support or not), with 95% CI;
  - linearly weighted κ or ordinal α of the three-level support scale;
  - κ of relevance;
  - proportion adjudicated.
- (Optional, after the gold labels are frozen) annotate the same candidate pools with AI and report the κ between AI and the expert gold labels, as an analysis of annotation cost.

## 7. Exclusions and missing data

- Only questions excluded in Phase 1 are left out of the analysis.
- After the gold labels are frozen, no question or candidate is removed.
- Candidates with missing evidence text count as not supporting, and their number is reported.

## 8. Deviation log

| Date | Change | Reason |
|---|---|---|
| | | |

[Translator's note: the log was left empty in the registered version. Deviations from this plan were recorded in the implementation protocol deposited with the same registration (Section 8 of `stage1_implementation_protocol_en.md`).]
