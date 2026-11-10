# Supplementary analysis registration for eval_v2 (v2.4-sup): official objective evaluation, re-checking with a stronger judge, and fine-tuning strong rerankers with support labels

> **English translation (unofficial).**
>
> - **Source:** the Chinese file `补充登记_v2.4.md` in the Stage 3 registration on OSF (https://osf.io/e3gkh), registered on 2026-11-04 at 17:01 UTC together with `FREEZE_MANIFEST_sup.json`.
> - **SHA-256 of the Chinese original:** `e289464d96c0fd618a26cd41cb958b77bf8e82ef8cbf257eee4fe5fe42f89fbc`. The original is the authoritative version.
> - **Purpose:** the translation was prepared on 2026-11-10, after all runs, for reviewers and readers. It adds nothing to the registered content.
> - **Conventions:** translator's notes are given in square brackets. "Gold labels" (金标) translates the original term for reference labels.

- Registration date: 2026-11-04 (before the runs)
- Pre-run revision (2026-11-04, still before the runs): the hardware was fixed as 2 Blackwell workstation cards (RTX PRO 6000 96 GB or RTX 6000D 84 GB); the GPU allocation was changed accordingly, and pre-run checks and failure handling were added (Section 8). Hypotheses, conditions, metrics, statistical methods and decision criteria were not changed.
- Relation: this file is a supplementary registration to `实验设计方案_v2.md` (v2.3, registered on 2026-10-28) and does not modify any hypothesis, metric or decision criterion of the original protocol. The primary hypothesis H3 of the original protocol and its conclusion follow the original registration.
  [Translator's note: the Stage 2 registration on OSF contains v2.2 (2026-10-27). Version v2.3, which added the external validation, was hash-frozen on 2026-10-28 but not deposited on OSF; see `stage2_amendment_v2.3_external_validation_en.md`.]
- File check: the SHA256 of all code, configuration and data files in the one-click package are listed in `FREEZE_MANIFEST_sup.json`, uploaded to OSF together with this file. At run time the launcher checks these files and automatically records any change in `logs/变更记录_sup.txt` [change record].

---

## 1. Timing and nature of this registration (stated truthfully)

This supplementary registration was written after all eval_v2 results were known: the principal investigator and the analyst had seen all answer-level and retrieval-level results on the sealed set and the external set. Therefore this supplementary analysis:

- is a supplementary analysis **specified in advance after the results were known and before the runs**, not a confirmatory test;
- is written up in the paper as a "supplementary analysis", reported separately from the main analysis of the original registration;
- has its comparisons, metrics, statistical methods and interpretation rules below fixed before the runs and not changed afterwards.

## 2. Why

Checks after eval_v2 was completed found two problems (see Section 2, "On the scoring metrics", of `评测结果报告_v2.md` [eval_v2 results report]):

1. **The primary metric depends on a 9B judge, and the judge is lenient.** Of 36 spot-checked judgements "hit, although the point's text does not appear verbatim", about half should not have been judged hits; judge A never used the "contradicted" label. The comparisons between retrieval conditions (H3) are less affected, but the size of the finding "retrieval conditions far below closed-book" may be overestimated.
2. **Key-point recall only checks whether a point is answered.** Wrong or extra content is not penalised, and the other steps of syndrome differentiation and treatment are not evaluated.

This supplementary analysis re-checks with two independent methods and adds one new comparison (S4):

- **S1, S2:** scoring with the expert gold standard and the official evaluation code of the external dataset TCM-BEST4SDT. S1 is the objective multiple-choice score, which does not depend on any judge model; S2 is the full official score (including the official judge and reward model).
- **S3:** re-judging all existing eval_v2 answers with a larger open-source model, Qwen3-32B, following the judging protocol of the original registration.
- **S4:** fine-tuning Qwen3-Reranker-0.6B and 4B with the support labels and with the relevance labels of the same stage4 training data, and comparing them with the original weights (Section 6).

## 3. S1, S2: official TCM-BEST4SDT evaluation

### 3.1 Data

- **Source:** the syndrome-differentiation and treatment task of TCM-BEST4SDT, `TCM_SDT.json`, 300 cases; GitHub repository DYJG-research/TCM-BEST4SDT, commit 60407f2, file SHA256 d94711ac… (the same file as used in the eval_v2 external validation), CC BY 4.0.
- **Gold standard:** the multiple-choice options and answers of each case were annotated by TCM experts, cross-validated by the annotators and approved by a third-party review (as described by the dataset authors). This study does not modify them.
- **Case ID mapping:** case number n corresponds to eval_v2 external question EX{n:03d}; the case texts of the 300 cases are identical, character for character, to the eval_v2 external set (checked).

### 3.2 Conditions

Exactly corresponding to the 13 runs of the eval_v2 external validation. The evidence reuses the selections already frozen in eval_v2 (`data/selections_external/`); nothing is re-scored or re-selected:

| Group | Runs | Evidence |
|---|---|---|
| X00 | 1 | None (closed-book) |
| X01 | 1 | BM25 Top-2 |
| X02 | 3 (draws r1–r3) | 2 random passages |
| X04 | 3 (seeds 42, 2026, 3407) | Relevance-CE (B encoding) Top-2 |
| X05 | 3 (seeds 42, 2026, 3407) | Support-CE (B encoding) Top-2 |
| X08 | 1 | bge-reranker-v2-m3 Top-2 |
| X09 | 1 | Qwen3-Reranker-0.6B Top-2 |

Seeds and random draws are first averaged within each question; a case enters a group only if all runs of that group have a result.

### 3.3 Evaluation procedure

- **Code:** the official evaluation code is included unchanged in `official/` (Apache-2.0). For each "condition × case", the official `TCMBenchmark._evaluate_single_case` is called once, i.e., the official complete evaluation of a single case:
  - multiple-choice questions on 4 dimensions (syndrome pattern, nature of disease, location of disease, treatment principle and method): 3 rounds per question, round 1 in the original order, rounds 2 and 3 with shuffled options;
  - aetiology, pathogenesis: scored by the judge model against the standard answer;
  - prescription: the reward model assesses formula–syndrome fit, and the judge model assesses compatibility rules, incompatibilities, herb safety and pregnancy contraindications;
  - decoction and administration, precautions, modifications according to symptoms: scored by the judge model.
- **Model under test:** Qwen3-4B-Instruct-2507, i.e., eval_v2's QA model G1, served by vLLM in bfloat16.
  - Decoding parameters are those passed by the official code (temperature 0, at most 4096 tokens).
  - The model has no thinking mode, so the two CoT dimensions are skipped with the official option `skip_think`.
- **How the evidence is added (the only change):**
  - The prompt of X00 is exactly the official one.
  - For the other conditions, the following evidence section is put at the very beginning of **every** generation request to the model under test (`config/prompts_sup.json`). The text of the evidence items is the same as when eval_v2 generated its answers.

    ```
    [Reference evidence] The following evidence was selected automatically by a retrieval system from case records and TCM literature. It may match this case, or it may be irrelevant or have mismatched conditions. Please judge for yourself in the light of the case, and you may refer to the parts that match this case.

    [Evidence E1]
    ……
    [Evidence E2]
    ……

    [Task]
    (original official prompt)
    ```
    [Translator's note: the prompt was used in Chinese; the English is a translation.]
  - The official tasks require an answer (multiple-choice answers, writing a prescription), so the answering mode here is "evidence + own knowledge", equivalent to eval_v2's P-alt rather than P-main, which answers from the evidence only. The paper must state this.
- **Official judge and reward model:**
  - The judge is Qwen3-32B and the reward model FangZheng-RM (based on Qwen3-14B; ModelScope: DYJGresearch/FangZheng-RM), both served by vLLM in bfloat16.
  - Request content and parameters are determined by the official code and not changed in this study.
  - The official code does not specify whether the judge uses thinking mode; vLLM's default is used here (Qwen3 has thinking on by default), with the qwen3 reasoning-content parser enabled so that the returned text does not contain the thinking process.
- **Seed for randomising the multiple-choice options:** 42 (the default of the official evaluator).
- **Failure handling:**
  - The official code scores a dimension 0 when a call fails. Here, results containing "evaluation failed" are treated as unfinished, with at most 3 retries per case.
  - Those still failing keep the score of 0 under the official rule and are flagged; their number is reported in the summary, together with a sensitivity analysis excluding them.
- **Audit:**
  - All returns of the judge, the reward model and the model under test are recorded.
  - The summary reports: the number of times the output of the model under test reached the length limit, the number of judge or reward-model returns without JSON (the official code then scores a default of 50), and the number of multiple-choice rounds whose answer could not be parsed.

### 3.4 Metrics

- **S1 (primary metric of this supplementary analysis):** the official multiple-choice mean score, i.e., the mean of the official scores on the 4 dimensions syndrome pattern, nature of disease, location of disease, and treatment principle and method. It does not depend on any judge or reward model. Scoring follows the official rules:
  - single-answer questions (nature of disease) score only if all 3 rounds are correct;
  - multiple-answer questions (syndrome pattern, location of disease, treatment principle and method) score S = |A∩B| / (|A| + |Ā∩B|) averaged over the 3 rounds, so wrong choices are penalised.
- **S2 (secondary):** the official total SDT score, i.e., the mean of the official 14 dimensions (with skip_think), computed as in the official `_calculate_final_scores`.
- **Descriptive:** the mean score of each dimension.

## 4. S3: re-judging key-point recall with Qwen3-32B

- **Scope:** all answers already generated in eval_v2, 18,290 in total:
  - all conditions of G1 and G2 on the sealed set, 14,390;
  - external set, 3,900.
- **Judge:** Qwen3-32B, bfloat16, non-thinking mode (consistent with the non-thinking setting of eval_v2's judges A and B), temperature 0, at most 2048 tokens.
- **Protocol exactly as in the original registration:**
  - the judge prompts are `judge_prompts.json` v2.1, unchanged;
  - the order of points is shuffled deterministically by answer ID, as for the original judges;
  - output checking and rewriting at most 3 times;
  - "hit/partial" whose quote check fails counts as "not mentioned";
  - a full abstention scores 0.
  - These functions in the code are copied line by line from eval_v2's `server/judge.py`.
- **The evidence-support labels are neither used nor uploaded.** The sealed-set questions and reference points will be on the rented server; they have been on this study's GPU server since eval_v1 and do not contain the evidence-support labels.

## 5. Comparisons and decision criteria (fixed before the runs)

The statistical methods are the same as in Section 4.4 of the original protocol:

- paired bootstrap over cases (questions), 10,000 resamples, seed 20261026, 95% percentile CI;
- Holm correction within families;
- equivalence tested by comparing the 90% CI with ±0.05.

| ID | Metric | Comparison | Decision | Status |
|---|---|---|---|---|
| S1-H3 | S1 multiple-choice mean score | X05 − X04 | Lower bound of the 95% CI > 0 | Main comparison of this supplementary analysis |
| S1-H4 | Same | X05 − X01, X05 − X02, X05 − X00 | Holm-corrected p < 0.05, two-sided | Secondary |
| S1-H7 | Same | X05 − X08, X05 − X09 | 90% CI within ±0.05 | Secondary |
| S2 | Official total SDT score | The three sets of comparisons of S1 | As above | Secondary |
| S3 | Key-point recall judged by Qwen3-32B | Repeat H3, H4 (family B) and H7 of the original protocol, and HX3, HX4, HX7 of the external set | As in the original protocol | Sensitivity analysis |
| S3-κ | Judge agreement | Judge A vs. Qwen3-32B, judge B vs. Qwen3-32B | Report κ and agreement | Descriptive |

**Interpretation specified in advance:**

| Result | Interpretation |
|---|---|
| S1-H3 holds | The advantage of support labels also holds with the expert gold standard, official objective scoring and "evidence + own knowledge" answering; may be written into the abstract |
| S1-H3 does not hold, while the original H3 holds | The advantage of support labels appears only in the "answer from the evidence only" setting (consistent with the P-alt results in eval_v2); the paper restricts the conclusion to this setting |
| In S1, X00 significantly higher than X05 | "Closed-book beats retrieval" is confirmed under official objective scoring |
| In S1, X05 not lower than X00 | The large gap between closed-book and retrieval in eval_v2 comes mainly from the "answer from the evidence only" prompt or from the scoring of key-point recall; the paper corrects "Finding 3" accordingly |
| In S3, H3 and HX3 still hold | The main conclusion does not depend on judge A |
| The closed-book gap given by S3 is clearly smaller than with judge A | The paper reports the size of the gap with the 32B results and the values calibrated by the human spot check, with judge A's values reported alongside as the originally registered results |

## 6. S4: fine-tuning Qwen3-Reranker (0.6B, 4B) with support labels and comparing with the original weights

### 6.1 Why

1. **To answer "can support labels make the strongest reranker better".**
   - The strongest zero-shot reranker in eval_v2 is Qwen3-Reranker: 0.089 above Support-CE on the external set.
   - The existing evidence comes only from bge (E2, H2b): after fine-tuning with support labels, retrieval did not exceed zero-shot (−0.002), and the answer level gained +0.042 (p = 0.055); the conclusion is unclear.
   - This is the question reviewers are most likely to ask, and it also decides whether the paper can claim that "support labels are a supervision signal that can be added to existing strong models".
2. **To separate the effects of "labels" and "scale".** Two sizes of the same family (600 million and 4 billion parameters) are trained with the same data, the same procedure and the same labels. Everything other than the number of parameters is the same, so the question "does the difference in parameters affect the results" can be answered directly.
3. **To replicate the label effect (support vs. relevance) on a third backbone.** E1 and E2 have already observed it on mDeBERTa and bge.

### 6.2 Training (`sup/s4_train.py`)

- **Data:**
  - exactly the same `prepared.pt` as stage4 and E2: the balanced training set of 26,432 pairs and the same dev and test splits;
  - labels `y_support` (support) or `y_relevance` (relevance);
  - exported read-only from the P40 server by the local "导出S4训练数据.bat" [export S4 training data]; the export package contains a SHA256 list of its files.
- **Models:** the official weights of Qwen/Qwen3-Reranker-0.6B and Qwen/Qwen3-Reranker-4B (ModelScope), the same as eval_v2's G09 and G13.
- **Input format:** exactly as in the zero-shot scoring of eval_v1 and eval_v2 (the official prompt format and the same TCM task instruction). The training objective is binary cross-entropy on the two tokens "yes / no" at the last position, consistent with how the model scores.
- **Method:**
  - LoRA: r = 32, alpha = 64, dropout 0.05, applied to all linear layers of attention and MLP;
  - learning rate 1e-4, 10% warm-up then linear decay, weight decay 0.01;
  - effective batch size 16 (4 × 4 accumulation), at most 3 epochs;
  - gradient clipping 1.0, gradient checkpointing, bf16;
  - training inputs at most 1024 tokens.
- **Checkpoint selection as in E2:** after each epoch, selection on dev with stage4's `checkpoint_key`, stopping after 2 epochs without improvement; with the threshold set on dev, P, R and F1 of the support class are reported on test.
- **Runs:** 2 sizes × 2 labels × 3 seeds (42, 2026, 3407), 12 in total. No hyperparameter is tuned on any evaluation set.

### 6.3 Scoring and evidence selection

- **Scoring:** 14 models (2 original weights, 12 fine-tuned) score the candidate pools of the sealed set (3,520 pairs) and the external set (6,000 pairs).
  - The scoring method is the same as eval_v1's `score_qwen3`: log P(yes), at most 2048 tokens;
  - precision is uniformly bf16. The original weights are also re-scored here, so that they are comparable with the fine-tuned models.
- **Evidence selection:** Top-2 per question, by the same rule as in eval_v2: scores from high to low, ties by retrieval rank.

### 6.4 Answer generation and scoring

- **New conditions** (14 runs per dataset); short names Q06 = 0.6B, Q4B = 4B:

  | Short name | Meaning | Runs |
  |---|---|---|
  | Q06N, Q4BN | Original weights | 1 each |
  | Q06S, Q4BS | Fine-tuned with support labels | 3 seeds each |
  | Q06R, Q4BR | Fine-tuned with relevance labels | 3 seeds each |

- **Reference conditions:** closed-book (G00, X00), Relevance-CE (G04, X04), Support-CE (G05, X05). They reuse the evidence selections frozen in eval_v2, and their answers are regenerated in the cloud so that all conditions share the same generation environment.
- **Generation:** Qwen3-4B-Instruct-2507, vLLM, bf16. The prompt is eval_v2's main prompt P-main (answer from the evidence only), identical character for character to eval_v2's `generate.py`; greedy decoding, at most 768 tokens.
- **Scoring:**
  - Key-point recall: sealed set and external set, judged by Qwen3-32B following the original judging protocol, as in Section 4;
  - On the external set, additionally the official multiple-choice evaluation: same procedure and evidence section as S1, multiple-choice part only. The multiple-choice scores of the reference conditions are taken from S1;
  - The retrieval-level metrics on the sealed set (strict and graded coverage, etc.) are computed locally with the evidence gold labels, not on the server.

### 6.5 Comparisons and decisions (fixed before the runs)

Statistical methods as in Section 5. Seeds are first averaged within each question.

| ID | Comparison | Metric and data | Decision | Status |
|---|---|---|---|---|
| S4-H1 | Q06S − Q06N; Q4BS − Q4BN | External key-point recall (Qwen3-32B) | The two Holm-corrected together; p < 0.05 and difference > 0 | **Main comparison of S4** |
| S4-H1′ | As above | Sealed-set key-point recall; external official multiple-choice mean score; sealed-set retrieval-level strict and graded coverage | As above | Secondary (replication) |
| S4-H2 | Q06S − Q06R; Q4BS − Q4BR | The metrics of S4-H1 | Holm-corrected, p < 0.05 | Secondary |
| S4-H3 | Q4BS − Q06S; Q4BN − Q06N; Q4BR − Q06R; interaction (Q4BS − Q4BN) − (Q06S − Q06N) | As above | Report differences and 95% CIs | Descriptive (scale) |
| S4-H4 | Q06S − Support-CE; Q4BS − Support-CE | As above | Holm-corrected, two-sided | Secondary |
| — | P, R, F1 of the support class on the stage4 test set | 12 fine-tuned models | Report | Descriptive |

**Interpretation specified in advance:**

| Result | Interpretation |
|---|---|
| S4-H1 holds for one size | Support labels bring an additional gain to the strongest zero-shot reranker of that size; this can be one of the paper's main contributions |
| S4-H1 holds for neither size, but S4-H2 holds | Support labels beat relevance labels, but cannot exceed the strong model's original relevance training. The paper's contribution is limited to "a comparison of label definitions", without claiming a gain for strong models |
| In S4-H3 the size difference is small and the CI includes 0 | With this study's candidate pools and number of questions, the number of parameters is not a major factor |
| In S4-H4, Q4BS significantly better than Support-CE | The paper's EEE layer can switch to a Qwen3-Reranker fine-tuned with support labels, stating the computational cost truthfully |

### 6.6 Deviations and limitations specific to S4

- **The training method differs from E2:** LoRA is used here, whereas E2 fine-tuned all parameters of bge. The reason is that both sizes should use the same method, and full-parameter fine-tuning of 4 billion parameters needs too much GPU memory. The paper states this truthfully.
- **Hyperparameters follow common settings:** learning rate etc. use common LoRA settings and were not tuned. If the results are poor, it cannot be ruled out that tuning would improve them.
- **Generation environment:** the reference conditions are regenerated in the cloud (bf16) and are not compared directly with eval_v2's P40 fp32 answers. S4 compares only differences between conditions within this run.

## 7. Known deviations and limitations

- **Different generation precision:** in S1 and S2 the model under test runs on vLLM in bfloat16, whereas eval_v2 ran in fp32 on the P40. S1 and S2 compare only differences between conditions within this run, not directly with the eval_v2 answers.
- **Results not fully reproducible:** vLLM's batching can make greedy decoding differ very slightly under different batch compositions, so a rerun may give slightly different results; restarts after a service interruption, or switching to a fallback deployment or fallback launch mode (Section 8), belong to the same category. All outputs and the actual deployments are saved.
- **Thinking mode of the judge:** the official materials do not state whether the official judge uses thinking mode; the default (on) is used here. If this has an effect, the paper says so.
- **Default scores in the official code:** when the output of the judge or reward model cannot be parsed, the official code scores a default (mostly 50). The counts are reported under the audit of Section 3.3, without correction.
- **Evidence selection reused from eval_v2:** the candidate pools and evidence selections of the external set all come from eval_v2 and were frozen at registration.
- **Existing limitation unchanged:** the reference points of the sealed set were still annotated by AI agents. The human spot check of Section 4.3 of the protocol is carried out separately and is outside the scope of this supplementary analysis.
  [Translator's note: that human spot check was not carried out; see Supplementary Material S2, item "Human audit".]

## 8. Execution and resources

### 8.1 Hardware (pre-run revision: 2026-11-04)

- **Actually rented:** 2 Blackwell-architecture workstation cards, RTX PRO 6000 (96 GB) or RTX 6000D (84 GB), PCIe, no NVLink; data disk ≥ 300 GB; driver ≥ 570 (CUDA 12.8).
- **Why this is enough:** the bf16 weights of the three served models total about 97 GB (Qwen3-32B about 61 GiB, FangZheng-RM about 28 GiB, Qwen3-4B about 7.5 GiB), and the two cards have 168–192 GB together. S4 fine-tuning runs only one job per card at a time, and the peak memory of LoRA fine-tuning of the 4B model is far below 84 GB.
- **GPU allocation (serve section of `config/sup.json`, `sup/serve.py`):**

  | Phase | Card 0 | Card 1 |
  |---|---|---|
  | S1/S2 official evaluation, S4 answer generation and multiple choice (all mode) | Judge Qwen3-32B (memory fraction 0.92) | Reward model (0.50) + model under test (0.40) |
  | S3 re-judging, S4 answer judging (judge mode) | Judge, tensor-parallel over both cards (0.92); if that fails, one judge replica per card | Same |
  | S4 fine-tuning and scoring | One job | One job |

  The judge has a card to itself so that it has enough KV cache (about 10 GiB or more on an 84 GB card, enough for more than 40,000 tokens). In phases that serve only the judge, tensor parallelism over both cards or two replicas are used instead, to speed up S3.
- **Launch modes:** each service tries default → eager (`--enforce-eager`, no CUDA graph) → triton (additionally `VLLM_ATTENTION_BACKEND=TRITON_ATTN`) in turn, until it is ready and passes a question-answering check. Tensor parallelism over both cards always adds `--disable-custom-all-reduce` (PCIe cards use NCCL only). The deployment and mode actually used are recorded in `logs/servers_history.jsonl` and written into the "runtime environment" section of the summary.
- **Effect on the results:** whether tensor parallelism is used, the number of replicas, whether CUDA graphs are used and the attention implementation do not change the model, the prompts or the decoding parameters; they can only cause differences at the level of floating-point rounding. This belongs to the same category as "results not fully reproducible" in Section 7 and is explained together in the paper.
- **The 4-card plan of the original protocol (4 cards of ≥44 GB)** remains in the code but is no longer the recommended configuration.

### 8.2 Run order: verify first, then run long jobs

To avoid repeated failures on a rented machine, all checks and trial runs come before the long runs. If any item fails, the programme stops immediately and states the reason.

1. **Environment:**
   - check the driver and disk;
   - install a C compiler or Python headers automatically if missing (needed by Triton to compile kernels);
   - create an isolated Python environment and install fixed versions;
   - put all pip, compilation and kernel caches on the data disk.
2. **GPU pre-check (before downloading the large models, about 2 minutes, `sup/preflight.py gpu`):**
   - allocation plan;
   - PyTorch support for this card architecture (sm_120);
   - bf16 matrix multiplication and attention numerics on each card;
   - Triton kernel compilation, torch.compile;
   - NCCL communication between the two cards; if it fails, try `NCCL_P2P_DISABLE=1` and record it;
   - vLLM import;
   - disk, shared memory, RAM.
3. **vLLM pre-check (first download only the 8 GB model under test, `sup/preflight.py vllm`):** actually start vLLM with the model under test and check three deployments:
   - single card;
   - two services on the same card (memory fractions as in the real deployment);
   - tensor parallelism over two cards.

   At the same time determine the usable launch mode for each deployment and warm up the kernel compilation cache.
4. Download the remaining models, check that the shards are complete and record the SHA256; check the S4 training data.
5. **Trial runs:**
   - interface checks of the three services (`preflight.py api all`):
     - the judge returns JSON in thinking mode, with no `<think>` in the text; the same check for streaming;
     - the judge returns JSON in non-thinking mode;
     - the reward model returns a "match score" with the official formula–syndrome matching prompt;
     - the model under test answers normally.
   - Official evaluation trial (3 cases × 2 conditions). Acceptance criteria: all succeed; judge outputs missing JSON or truncated at most once in total; the same at most once for the reward model; truncation of the model under test at most one third.
   - S4 generation trial: 8 items, using the frozen evidence of the reference conditions.
   - Interface check of the judge served alone.
   - S3 and S4 re-judging trials, 6 items each; at least 5 valid.
   - S4 fine-tuning smoke test: train 0.6B and 4B for 15 steps each and trial-score, giving a time estimate for all fine-tuning.
6. **Main run:**
   - S4: 12 fine-tuning runs, scoring, evidence selection;
   - S1/S2;
   - S4 answer generation and multiple-choice evaluation;
   - S3 re-judging;
   - S4 answer judging;
   - summary and packaging.

### 8.3 Failure handling during the run

- **Service interruption:** when a task fails, first check the service health (3 checks, 20 seconds apart). If the service is down:
  - this failure does not count towards the retries of that case or answer, so a service failure is never recorded as a score of 0;
  - the programme ends with exit code 3, stops and restarts the service automatically and continues from the checkpoint, with at most 3 restarts per step.
- **Call failures swallowed by the official code:** the official code catches some call exceptions and continues scoring (e.g., with an empty answer). Therefore a case is treated as unfinished as soon as one call still fails after retries: if the service is down, it is handled as above; otherwise it counts as one retry. This way service failures do not silently turn into low scores.
- **Hung service:** if no task completes for 30 consecutive minutes, the service is considered hung and handled as an interruption.
- **Resuming:** results are appended item by item, and completed items are not rerun. After restarting the machine, running `bash run_all.sh` again continues from the checkpoint.
- **Remembering launch modes:** once a service has started successfully in a certain mode, that mode is tried first on later restarts.

### 8.4 Software and time

- **Software versions:** vllm 0.11.0 (torch 2.8.0, CUDA 12.8), transformers 4.56.1, openai 2.7.1, numpy 2.2.6, modelscope 1.31.0, peft 0.17.1, accelerate 1.10.1; Python 3.10–3.13.
- **Planned time (2 cards; estimated from the card specifications, not measured):**

  | Phase | RTX PRO 6000 96 GB | RTX 6000D 84 GB |
  |---|---|---|
  | Installation, pre-checks, downloads, trial runs | About 1.5–2.5 hours | Same |
  | S4 fine-tuning and scoring | About 6–8 hours | About 8–11 hours |
  | S1/S2, S4 generation and multiple choice | About 9–13 hours | About 14–20 hours |
  | S3 re-judging, S4 answer judging | About 2–3.5 hours | About 2.5–4 hours |
  | Total | About 19–27 hours (about 38–54 card-hours) | About 26–38 hours (about 52–76 card-hours) |

  The bottleneck of S1/S2 is the judge (thinking mode, 3 judge calls per case); on the 84 GB card the judge's KV cache is about half that on the 96 GB card, so this phase is clearly slower. The estimate given after the trial runs and the "time remaining" during the run take precedence.
