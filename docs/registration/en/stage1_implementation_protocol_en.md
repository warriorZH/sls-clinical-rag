# Protocol for the model comparison on the sealed set (eval_v1)

> **English translation (unofficial).**
>
> - **Source:** the Chinese file `实验对比方案.md` in the Stage 1 registration on OSF (https://osf.io/gj48z), registered on 2026-10-26 at 09:17 UTC together with the analysis plan and `FREEZE_MANIFEST.json`.
> - **SHA-256 of the Chinese original:** `1b8257ff78b70d34aaf42e81dec2a650e43939cddc7422a2c360315a342c5775`. The original is the authoritative version.
> - **Purpose:** the translation was prepared on 2026-11-10, after all runs, for reviewers and readers. It adds nothing to the registered content.
> - **Conventions:** translator's notes are given in square brackets. "Gold labels" (金标) translates the original term for the frozen reference labels. One local path contains the server user name; it is shown here as `<user>`.

Basis: `预注册分析计划.md` [the pre-registered analysis plan] (SHA256 `68b9e7353295cf9107a5ec07f89d257fcd0f8bcde62a3b35150c87c9d2b2c47c`).

This protocol only adds implementation details that the plan does not specify. It does not change the methods compared, the primary hypothesis or the metrics in the plan. Everything that differs from the plan or is not specified by it is listed in Section 8 and must be entered in the plan's "Deviation log".

This protocol must be finalised before the sealed set is scored and time-stamped together with the plan.

## 1. Frozen inputs

| File | SHA256 | Content |
|---|---|---|
| `final_questions.json` | `af0b2b4d34a84a4be57524c84788df938e295fd3e9afd263b25d7b7c70b5aeb3` | 180 questions, 176 included (ST016, ST100, ST107 and ST170 excluded); 547 required points: disease diagnosis 108, syndrome pattern 93, pathogenesis 181, treatment principle 165 |
| `sealed_pools.jsonl` | `6c228959456ad06765adc84f86811eb0abe410901769fe5013bd7d6559de4659` | 176 questions × 20 candidates, 3,520 candidates in total; sources: other patients' case records 2,860, textbooks 544, classical texts 116 |
| `phase2_gold_labels.jsonl` | `1311d616353f78fb843bfae58d5338877b5d4ec836d0ccd6ab47ec58334f4e7a` | Support score (2/1/0) of each of the 3,520 candidates for each point of its question, plus two further labels: relevance, and same disease with a different pattern or mismatched conditions |

Structural check (2026-10-26; structure only, no metric computed):

- every candidate has gold labels;
- ranks 1–20 are complete for every question, with no duplicate documents;
- no evidence text is missing;
- the point identifiers in the gold labels correspond one-to-one to the points of the questions.

**Source of the annotations.** According to the file records, the three-way A/B/C annotation of Phase 1 and Phase 2 and the adjudication of both phases were all carried out by AI agents following the *Annotation guidelines v1.1*, not by the TCM professionals foreseen in the README plan. The evidence is as follows:

- the Phase 2 annotation sheets were generated at 23:02 on 10-25, and annotator B returned all 3,520 rows at 23:19;
- A's 3,520 remarks contain only 27 distinct wordings;
- `phase2_仲裁说明_v2.md` [Phase 2 adjudication notes] contains the phrase "take B as specified by the user";
- the Phase 1 adjudication was filled in in batch by the `@oai/artifact-tool` script.

By decision of the principal investigator, this evaluation uses `phase2_gold_labels.jsonl` as the gold labels to validate the model approach. When the results are reported externally, the source of the annotations must be stated truthfully and may not be described as expert annotation; the κ among A/B/C may not be presented as expert agreement either.

## 2. Input text and selection rules

- **Query:** `question_final`. This is the question text used for retrieval and annotation, including the final sentence "Please explain …" or "Please state …".
- **Evidence:** `context_text`, i.e., the title plus context for textbook and classical-text entries; for case records, the original snippet. If `context_text` is empty, `text` is used instead. This is identical, character for character, to the "evidence text" in the annotation sheets.
- **Selection:** each method scores the 20 candidates of a question and takes the 2 with the highest scores. Ties are broken in favour of the better BM25 rank.
- **Ranking:** for nDCG@10, the 20 candidates are ranked with the same scores and the same tie rule.

## 3. Methods compared

| Method | Runs | Fixed settings |
|---|---|---|
| Support-CE | 3 (seeds 42/2026/3407) | The checkpoints used in E3, `experiments/stage4_retraining/support_<seed>/best_checkpoint.pt`. Loading, encoding and scoring exactly as in E3 (see below) |
| Relevance-CE | 3 (seeds 42/2026/3407) | As above, with checkpoints `relevance_<seed>/best_checkpoint.pt` |
| BM25 original ranking | 1 | No model is called; pool ranks 1 and 2 are taken |
| bge-reranker-v2-m3 | 1 | Zero-shot; fixed to revision `953dc6f6` on hf-mirror; transformers sequence classification; input in (query, evidence) order; maximum length 1024; fp32; the score is the logit |
| Qwen3-Reranker-0.6B | 1 | Zero-shot; fixed to revision `e61197ed`; official prompt template; the score is P(yes); maximum length 2048; fp32; the instruction is fixed as `Given a Traditional Chinese Medicine clinical case and a question about it, retrieve passages that help answer the question` |
| (Optional) LLM point-wise judgement | 0 | **Not run this time**; see D6 for the reason. Original design: Qwen2.5-7B-Instruct, zero-shot, judging candidate by candidate, given only the question and the evidence, without the gold points; the prompt is `LLM_JUDGE_PROMPT` in `server_score.py`; the score is P(yes)/[P(yes)+P(no)] |

- All checkpoints and settings are fixed before the sealed set is scored; no parameter is tuned on the sealed set, and no threshold is used.
- Maximum lengths are set from measurements so that the public models do not truncate: question plus evidence is at most 679 characters. The CE models keep their training lengths, and the number of truncated items is written to the run manifest.
- Precision: the server has Tesla P40 GPUs, which do not support bf16 and have low fp16 throughput, so all methods use fp32. The actual precision is written to the run manifest.

**Loading, encoding and scoring of Support-CE / Relevance-CE**

Based on `stage5_real_retrieval_v1_2/score_three_seed_review.py` of E3 on the server and `predict()` in `stage4_retraining/train_stage4.py`:

- A binary classification model is built from `/home/<user>/models/mdeberta-v3-base` (fp32), and the `model` weights of the checkpoint are then loaded.
- Input = `[CLS]` + the first 160 SentencePiece tokens of the question + `[SEP]` + the first (509 − question length) tokens of the evidence + `[SEP]`; only input_ids and attention_mask are given.
- Sorted by length, batches of 4; the score is the probability of class 1 after softmax.

**Reproduction check:** in the calibration stage, every CE checkpoint re-scores, in the way described above, 200 candidates already scored in E3 (rows of `candidates_unlabeled.jsonl` outside the training set, sampled at equal intervals), and the scores are compared with `parent_<target>_<seed>/expanded_context_scores.npy` saved by E3. The maximum difference must be below 0.001; where checkpoint hashes were recorded, they must also match the E3 records. Methods that fail are not used to score the sealed set.

## 4. Run rules

1. **Smoke test:** first run all methods on 5 calibration questions (development set, not part of the sealed set), i.e., `--target calib`. The CE checkpoints undergo the reproduction check at the same time.
2. **Time-stamping:** upload this protocol, the pre-registration plan and `FREEZE_MANIFEST.json` to OSF, or commit them to a git repository with time records.
3. **Single run:** run all methods on the sealed set, i.e., `--target sealed`.
   - The script requires every method to pass the calibration test with the same settings first, and it refuses to overwrite existing results.
   - If a rerun is truly necessary because of a failure, the reason must be stated. The old results are kept, and everything is reported truthfully.
4. **Isolation of the gold labels:** the gold-label file is not uploaded to the server; the scoring script reads only the question text and the candidate pools.
5. **Report everything:** all methods in the plan are reported; for methods that could not be run, the reason is stated. Every failed run is kept under `failures` in the run manifest.

## 5. Metric definitions

For each question q: P_q is the set of points, S_q the 2 selected passages, and s(c, p) the gold support score (2/1/0) of candidate c for point p.

| Metric | Value per question | Description |
|---|---|---|
| **Coverage (primary metric)** | \|{p ∈ P_q : max_{c∈S_q} s(c,p) = 2}\| / \|P_q\| | Union of full support over the 2 selected passages |
| Full coverage | 1[coverage = 1] | All points fully supported |
| Selected-non-supporting rate | \|{c ∈ S_q : max_p s(c,p) < 2}\| / 2 | Proportion of selected passages that fully support no point |
| nDCG@10 | DCG@10 / IDCG@10 | Gain g(c) = max_p s(c,p) ∈ {0,1,2}, linear gain, discount log2(i+1); questions whose pool contains no support at all (IDCG = 0) are not included, and their number is reported separately |
| Pool upper bound | Coverage when all 20 are selected | Independent of the method; shows the recall bottleneck |
| Coverage, partial-support criterion | Condition "= 2" replaced by "≥ 1" | Sensitivity analysis |

- **Aggregation for the CE models:** compute per seed, then average the 3 seeds within each question, then average over all questions.
- **Other methods:** one value per question, averaged directly over all questions.

## 6. Statistical tests

**Primary hypothesis H1**

- Let d_q be, for each question, the coverage of Support-CE (seed average) minus the coverage of Relevance-CE (seed average); the test statistic is mean_q d_q.
- Paired bootstrap over questions: 10,000 resamples, random seed 20261026, percentile 95% CI (linear interpolation).
- Decision rule: H1 is considered supported if the lower bound of the CI is above 0; otherwise it is reported as not supported.

**Secondary comparisons**

- All comparisons use the same set of bootstrap resampling indices.
- Two-sided p value: p = min(1, 2·min[(#{d* ≤ 0}+1)/(B+1), (#{d* ≥ 0}+1)/(B+1)]).
- Holm correction within the comparison family of each metric, α = 0.05:

| Family | Comparisons | Number of tests |
|---|---|---|
| F1 coverage | Support-CE vs. BM25, bge-reranker-v2-m3, Qwen3-Reranker-0.6B | 3 |
| F2 full coverage | Support-CE vs. Relevance-CE, BM25, bge, Qwen3 | 4 |
| F3 selected-non-supporting rate | As above | 4 |
| F4 nDCG@10 | As above | 4 |
| F5 coverage, partial-support criterion | As above | 4 |

- For every method and metric, the point estimate and the 95% bootstrap CI are reported.
- The optional LLM method is not placed in any family; only uncorrected CIs are reported.

## 7. Stratification, seeds and annotation quality

**Stratification** (descriptive, no significance conclusions):

- Question type: diagnosis-explanation and treatment-principle questions.
- Source stratum: fresh_med and heldout_med, crossed with question type.
- Evidence source: textbooks, classical texts, other case records. Two things are reported:
  - the source composition of the passages selected by each method, and the proportion of fully supporting passages in each source;
  - the yield of full support from each source in the candidate pools.

**Seeds:** for Support-CE and Relevance-CE, the result of each seed and the standard deviation across seeds are reported.

**Annotation quality:**

- Phase 1: κ of the point judgements; proportions of programme-drafted points kept, modified or deleted; number of excluded questions and the reasons.
- Phase 2:
  - full support or not: Cohen's κ and Krippendorff's α with 95% CI;
  - three-level support: linearly weighted κ and ordinal α;
  - κ of relevance;
  - proportion adjudicated.
- Additional reporting: number of positives, positive and negative agreement, PABAK and Gwet's AC1, to deal with the κ paradox caused by rare positives.
- The optional item in Section 6 of the plan, "κ between AI and the expert gold labels": not applicable, because the gold labels themselves were generated by AI.

**Exploratory analyses** (not pre-registered, listed separately, not used for conclusions):

- X1 graded coverage: full support counts 1, partial support 0.5.
- X2 conditional analysis: H1 repeated only on questions whose pool contains full support for at least one point.
- X3 misleading rate: proportion of selected passages labelled "same disease with a different pattern or mismatched conditions".
- X4 relevant-but-non-supporting rate: proportion of selected passages with relevance 1 that fully support no point; also the proportion with relevance 1 that support no point at all (all 0).
- X5 relevance rate of the selected passages.
- X6 results of the optional LLM point-wise judgement.
- X7 Spearman correlation between seeds.

## 8. Deviations and implementation notes (to be entered in the pre-registration "Deviation log")

| ID | Item | Description |
|---|---|---|
| D1 | Annotators | The plan foresaw annotation and adjudication by TCM professionals; they were actually carried out by AI agents (see Section 1). The principal investigator decided to use the labels as gold labels. |
| D2 | Questions and points | Phase 1 excluded 4 questions, and 176 questions enter the analysis, in line with Section 7 of the plan. The points were revised from 412 drafted points to 547. |
| D3 | Optional LLM method | Only the question and the evidence are given, and the model judges whether the evidence can fully support a point required to answer the question; the gold points are not given, so that no gold-label information enters the method. |
| D4 | Implementation details | The following items are not specified in the plan and are all fixed by this protocol before scoring: the Qwen3-Reranker instruction text, the maximum length and precision of each model, the tie rule, the handling of questions without support in nDCG, the bootstrap random seed and the definition of the p value, and the Holm families. |
| D5 | Exploratory analyses | X1–X7 are not pre-registered; they are labelled separately and not used for conclusions. |
| D6 | Optional LLM method not run | The server has Tesla P40 GPUs: no bf16 support, low fp16 throughput, a 7B model in fp32 needs two cards, and the model had not been downloaded. The method is optional in the plan; it is not run this time, and this is stated in the results. |
| D7 | Precision and model revisions | All methods use fp32; the two public rerankers are fixed to specific revisions on hf-mirror (bge `953dc6f6`, Qwen3 `e61197ed`). |

## 9. Output files

| Location | File | Content |
|---|---|---|
| Server `eval_v1/scores/<calib\|sealed>/` | `<method>.jsonl` | Score of every candidate |
| Same | `RUN_MANIFEST.json` | Checkpoint fingerprints, settings, number of truncated items, run times, hashes of the input files |
| Local `eval_v1/results/sealed/` | `评测结果报告.md` [evaluation report] | Results report |
| Same | `results.json` | All values |
| Same | `per_question.csv` | Metrics per question and method |
