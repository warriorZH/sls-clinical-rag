# Experimental design protocol v2: input-matched retrieval re-evaluation and answer-level evaluation

> **English translation (unofficial).**
>
> - **Source:** the Chinese file `实验设计方案_v2.md` (version v2.2) in the Stage 2 registration on OSF (https://osf.io/e37na), registered on 2026-10-27 at 14:46 UTC together with `FREEZE_MANIFEST_v2.json`.
> - **SHA-256 of the Chinese original:** `93d66880961665a2611184948d9ea813bc18b21112cfad1d3093bf0483113119`. The original is the authoritative version.
> - **Purpose:** the translation was prepared on 2026-11-10, after all runs, for reviewers and readers. It adds nothing to the registered content.
> - **External validation:** this registered version does not contain the external validation on TCM-BEST4SDT. That was added by amendment v2.3, which was hash-frozen on 2026-10-28 at 07:50 (UTC+8), before any external-set run, but was not deposited on OSF. See `stage2_amendment_v2.3_external_validation_en.md`.
> - **Conventions:** translator's notes are given in square brackets. The GPU host name is shown as `<gpu-host>`.
>   - "Gold labels" (金标) translates the original term for the frozen reference labels, which were produced by AI agents (Section 0, item 6).
>   - "The thesis" refers to the first author's unpublished thesis, which the protocol used as a design reference. Equation and section numbers prefixed "thesis" point to it.
>   - Prompts and the judging protocol in Appendices A and B are given in English translation. The Chinese wording in the original is authoritative.

- Data: expert_testset_v1 (gold labels, questions and candidate pools all unchanged)
- Date written: 2026-10-26
- Version: v2.2 (registered version). Changes relative to v2.0 are in Section 14; v2.2 changes the judges to local open-source models on the server, see Section 15. All experiments are executed by an automatic pipeline on the server.
- Methodological basis: the experiment-design skill (hypotheses → variables → metrics → baselines → ablations → compute budget)

---

## Summary

**Why redo the experiments.** The existing experiments have two gaps:

1. **Unequal inputs.** The CE models see only the first 160 tokens of the question, while the public models see the full question.
2. **No answer-level evaluation.** eval_v1 only evaluated the coverage of points by the Top-2 evidence; it did not give the evidence to a question-answering model and compare the generated answers.

**Structure of the new protocol:**

- **Retrieval level: re-compare rerankers with matched inputs (E1, E2).** E1 corrects the truncation; E2 lets a strong public model learn "support" and "relevance" labels separately on the same data, separating the effect of the label definition from that of the model and data.
- **Answer level: same QA model, different evidence sources, compare the answers (E3).** The QA model is fixed to Qwen3-4B-Instruct-2507 (the base of the thesis's QA layer, not fine-tuned).
- **Mechanism: build evidence directly from the gold labels (E4).** Give "supporting evidence", "relevant but non-supporting evidence" and "irrelevant evidence" separately, to test directly whether "support ≠ relevance" holds at the answer level, independently of how good any reranker is.

**Primary hypothesis (H3):** With the same QA model, answers based on the Top-2 evidence selected by Support-CE have a higher strict key-point recall than answers based on evidence selected by Relevance-CE.

**Constraints:**

- The gold labels are not changed. The evidence-support labels (phase2_gold_labels.jsonl) are not uploaded to the server; the question and reference-point files have been on the server since eval_v1.
- Prompts, the judging protocol, etc. are all fixed on non-sealed data, and the protocol is run once on the sealed set only after OSF registration.
- The sealed set was already used once by eval_v1; v2 is a new pre-registered analysis on the same test set. The final conclusions still need to be confirmed on a new sealed set annotated by TCM experts.

**Resources:**

- GPU: about 95 GPU-hours for generation, scoring and fine-tuning, and about 100 GPU-hours for local judging, about 195 GPU-hours in total (two P40s running continuously for about 4.5–5 days).
- Judges: both judges are open-source models running on the server; no paid API is called.
- Human spot check: about 6–8 hours (after the results are available).
- Duration: about 5 days of automatic running on the server (main results in about 2 days), followed by about 1 week of analysis and reporting.

---

## 0. Diagnosis of the current state

| Problem | Facts | Design response |
|---|---|---|
| 1. Question truncation (to CE's disadvantage) | The E3 scoring script (`score_three_seed_review.py`, etc.) uses `sp.encode(question)[:160]`; the training script `prepare_stage4.py` uses the full question (each pair at most 512 tokens, not truncated). So 160 is an inference-time convention inconsistent with training. Of the 176 sealed questions, 68 (1,360 inputs) are truncated. The last sentence of every question is the question-type instruction (99 "Please explain the diagnosis, syndrome pattern and aetiology and pathogenesis of this case", 77 "Please state the treatment principle applicable to this case …"), and truncation cuts it off first; the median first appearance of tongue and pulse findings is 76 characters before the end of the background description. In other words, for these 68 questions CE does not know whether the question asks about diagnosis or treatment principle, and often does not see the tongue and pulse either | E1: re-score CE with the full question; at the same time a reverse control in which the public models see exactly the same truncated text as CE |
| 2. Unequal training data (to CE's advantage) | CE was trained on 26,432 in-domain pairs; bge and Qwen3-Reranker are zero-shot | E2: fine-tune bge-reranker-v2-m3 with the same data and procedure, once with support labels and once with relevance labels |
| 3. No answer-level evaluation | eval_v1 only evaluated point coverage by the Top-2 evidence; no answers have been generated on the sealed set. The evaluate.py of `multimodel_qa_v1/v2` and `single_gpu_fp16_v1` on the server contains condition, reranker_seed and necessary_points fields. Judging from these fields, condition-wise QA comparisons were run on old data before; details to be confirmed by inspection | E3, E4; collect the old QA code first and reuse what can be reused |
| 4. Narrow measurable range | The pool upper bound under the strict criterion is 0.084. Under the partial-support criterion it is 0.470: 0.703 for diagnosis questions and 0.171 for treatment-principle questions. Only 31 questions have fully supporting evidence in the pool, and 114 have at least partially supporting evidence | Answer-level key-point recall is not bounded by the pool upper bound (the model also uses its own knowledge); pre-registered analysis of conditional subsets; E6 enlarges the candidate pool (optional) |
| 5. Weak control | The rank correlation among the three Relevance-CE seeds is only 0.36–0.59 | E2 adds relevance-fine-tuned models with the same backbone and data as a strong control |
| 6. Source of the gold labels | Annotated and adjudicated by AI agents; Krippendorff's α of the "full support" label is 0.220 | Gold labels not changed; a separate human spot check of the answer-level judging; all conclusions state the source of the gold labels |

---

## 1. Research hypotheses

### 1.1 Overall hypothesis

Selecting evidence by "whether it can support the required answer points" (support labels), rather than by topical relevance, makes the same QA model write more correct points and fewer unsupported claims; and with matched inputs this advantage is not smaller than that of existing strong rerankers.

### 1.2 Sub-hypotheses

| ID | Hypothesis | Experiment | Decision rule | Status |
|---|---|---|---|---|
| H1 | Support-CE has higher strict Top-2 coverage than Relevance-CE | eval_v1 | Already supported: difference 0.033, 95% CI [0.014, 0.055] | Reported |
| H1a | With the full question, Support-CE's graded coverage on the 68 truncated questions increases | E1 | Lower bound of the 95% CI of S-B − S-A on the 68 questions > 0 | Secondary |
| H1b | When CE also sees the full question (S-B), Support-CE is non-inferior to bge and Qwen3-Reranker-0.6B (N) | E1 | S-B − BGE-N and S-B − QR-N: lower bound of the 90% CI of the strict-coverage difference > −0.02; lower bound of the 90% CI of the graded-coverage difference > −0.03 | Secondary |
| H1c | If the public models see only the same truncated text as CE, their coverage drops | E1 | Report point estimates and CIs of BGE-A − BGE-N and QR-A − QR-N to quantify the cost of truncation | Descriptive |
| H1d | H1 still holds with the full question | E1 | Strict-coverage difference S-B − R-B, lower bound of the 95% CI > 0 | Secondary |
| H2a | With the same backbone (bge), fine-tuning with support labels beats fine-tuning with relevance labels | E2 | Lower bound of the 95% CI of the strict-coverage difference > 0 | Secondary |
| H2b | Fine-tuning with support labels makes a strong reranker better | E2 | Lower bound of the 95% CI of BGE-S − BGE-N > 0 | Secondary |
| H2c | The label effect is the same on the two backbones (interaction) | E2 | Two-sided test of the interaction term | Exploratory |
| **H3** | **With the same QA model, the strict key-point recall (KPR) with Support-CE evidence is higher than with Relevance-CE evidence** | E3 | Lower bound of the 95% CI of KPR(G05) − KPR(G04) > 0 | **Primary hypothesis** |
| H4 | KPR with Support-CE evidence is higher than with BM25 Top-2 and with 2 random passages; two-sided test against closed-book | E3 | Holm-corrected p < 0.05 | Secondary |
| H5 | With Support-CE evidence, the supported-claim rate (SCR) is higher and the unsupported-claim rate (GHR) lower, compared with Relevance-CE and BM25 | E3 | Holm-corrected p < 0.05 | Secondary |
| H6a | Gold supporting evidence (O1) yields higher KPR than "relevant but non-supporting" evidence (O2) | E4 | Lower bound of the 95% CI of O1 − O2 on the 70-question subset > 0 | Secondary (mechanism) |
| H6b | "Relevant but non-supporting" evidence cannot raise KPR, and may even lower it (relative to closed-book) | E4 | Two-sided test of O2 − G00 | Secondary (mechanism) |
| H7 | At the answer level, Support-CE is equivalent to bge and Qwen3-Reranker | E3 | 90% CI of the difference within ±δ (TOST; δ defaults to 0.05, confirmed in E0) | Secondary |
| H8 | The higher the support level of the evidence for a point, the higher the probability that the answer states the point (controlling for whether the closed-book answer stated it) | E4 | In the point-level GEE, lower bound of the 95% CI of the OR for s=2 vs. s=0 > 1 | Exploratory |

---

## 2. Variable definitions

### 2.1 Independent variables

| Variable | Values | Experiment |
|---|---|---|
| Evidence selection method | Closed-book; BM25; random; full pool; Relevance-CE; Support-CE; bge-reranker-v2-m3; Qwen3-Reranker-0.6B; (optional) Qwen3-Reranker-4B, LLM zero-shot judgement; bge fine-tuned with support/relevance labels; evidence built from the gold labels, O1–O4 | E1–E4 |
| Reranker input | A: first 160 tokens of the question, evidence filling up to 512 (original E3)<br>B: full question, evidence filling up to 512<br>C: no truncation at all (at most 1024)<br>N: full input of the public models (eval_v1 setting, no truncation)<br>Two further settings for the public models: text identical to A or to B | E1 |
| Training labels | Support, relevance, (optional) shuffled labels | E2 |
| Backbone | mDeBERTa-v3-base, bge-reranker-v2-m3 | E2 |
| Number of passages k | 2 (main); 1, 3, 5; full pool of 20 | E3, E5 |
| Prompt | P-main: answer from the evidence only; say so when the evidence is insufficient<br>P-alt: may also use own knowledge; cite the evidence numbers used; no abstention instruction | E3, E5 |
| QA model | G1: Qwen3-4B-Instruct-2507 (main)<br>G2: GLM-4-9B-0414 (robustness; if the conditions are not met, glm-edge-4b-chat is used instead, see Section 14) | E3, E5 |

### 2.2 Dependent variables

| Variable | Definition | Status |
|---|---|---|
| Strict key-point recall (KPR) | For each question, the number of points judged "hit" by the judge ÷ the number of points of the question; a full abstention counts as 0; then averaged over questions | Primary answer-level metric |
| Graded key-point recall | Hit counts 1, partial 0.5 | Secondary |
| Point contradiction rate | Proportion of points judged "contradicted" | Secondary |
| Key-claim precision, F1 | Among the diagnosis, syndrome-pattern, pathogenesis and treatment-principle claims in the answer, the proportion that match a gold point | Secondary |
| All-correct rate | Proportion of questions whose points are all hit | Secondary |
| Supported-claim rate (SCR), unsupported-claim rate (GHR) | Computed as in thesis Eqs. (3.59) and (3.60), relative to the given evidence; "contradicted" and "unsupported" reported separately | Secondary (only conditions with evidence) |
| Answer coverage, selective accuracy | Computed as in thesis Eq. (3.61) | Secondary |
| Literal recall | Proportion of points whose original text appears verbatim in the answer; does not depend on a judge model | Robustness |
| ROUGE-L, BERTScore | Against the syndrome-differentiation analysis or treatment text of the original case record | Descriptive |
| Retrieval-level metrics | Strict, graded and partial coverage, nDCG@10, selected-non-supporting rate, misleading rate, full coverage; definitions as in eval_v1 | E1, E2 |
| Mechanism metrics | Evidence-use rates U0, U1, U2; points gained and lost relative to closed-book | E4 |
| Efficiency | Input and output tokens, reranking and generation time, peak GPU memory, number of parameters | Descriptive |

### 2.3 Control variables

| Variable | Fixed value | Note |
|---|---|---|
| Questions, candidate pools, gold labels | SHA256 af0b2b4d…, 6c228959… and 1311d616… respectively | Same as eval_v1's FREEZE_MANIFEST |
| Evidence text | Title plus context, as in annotation and scoring | No source information in the prompt |
| Evidence order | Rank given by the method; random conditions and gold-built conditions shuffled with fixed seeds | |
| QA model | Qwen3-4B-Instruct-2507, fixed snapshot hash; fp32; greedy decoding; max_new_tokens = 768; batched by a fixed token budget (independent of the GPU used) | No LoRA, reason in 3.4 |
| Prompts | Final text of Appendix A, hash recorded | |
| Judges | Two open-source models from families different from the QA model G1 (Qwen3), run locally on the server: A is GLM-4-9B-0414 (all answers), B is Hunyuan-7B-Instruct (about 25% of questions, thinking mode off); fixed weight hashes; fp32, greedy decoding; following the protocol in Appendix B | Judges do not see condition names |
| Reranker scoring | fp32; public models fixed to snapshots (bge 953dc6f6…, Qwen3-Reranker e61197ed…) | |
| Randomness | CE seeds 42, 2026, 3407; 3 sampling seeds for the random condition (fixed at registration); bootstrap seed 20261026 | |
| Hardware, software | 2 Tesla P40; Python 3.10.12; torch 2.7.1+cu118; transformers 5.19.0 | Written to the run manifest for every run |

---

## 3. Experimental modules

### 3.0 Overview

| Module | Question answered | What is run on the sealed set | Priority |
|---|---|---|---|
| E0 technical check | Does the pipeline run, are the results reproducible, are the judge interfaces usable | Sealed set not touched | Required (automatic) |
| E1 input-matched retrieval re-evaluation | After correcting the truncation, is CE or the public models better | Reranker scoring only | Required |
| E2 labels × backbone | Does the advantage come from the "support labels", or from the model and data | Training on stage4 data; sealed set scoring only | Should do |
| E3 main answer-level experiment | How much better do different kinds of evidence make the same QA model's answers | Generate answers and judge them | Required |
| E4 mechanism experiment | Does the difference between "support" and "relevance" affect the answers | Generate answers and judge them | Required |
| E5 robustness and sensitivity | Do the conclusions depend on k, the prompt, the QA model or the decoding | Generate answers and judge them | Should do |
| E6 open-pool end-to-end | Do the conclusions change when recall improves | New retrieval, generation and judging | Optional |
| E7 qualitative analysis | At which step do the errors occur | Sample-based analysis | Should do |

### 3.1 E0 technical check (automatic, sealed set not touched)

From v2.1 on, the prompts, the judging protocol and the equivalence margin δ = 0.05 are all finalised before registration; E0 is no longer used to adjust them and only checks whether the technical conditions are met. E0 is run automatically by the pipeline; if it fails, the dependent jobs are not started.

1. **Environment check.** Number and compute capability of the GPUs, Python package versions, input file hashes, eval_v1 score files and checkpoints, stage4 training data, disk and network.
2. **Generation check.** Run three conditions (closed-book, BM25 Top-2, full pool of 20) on the 5 calibration questions, twice each:
   - the two outputs must be identical character for character (greedy decoding is reproducible);
   - `<think>` must not appear;
   - record peak GPU memory, speed, and the proportions of outputs reaching the length limit or repeating.
   - GPU memory stress test: first tokenise (tokenise only, no generation) the "all 20 passages" prompt of every calibration and sealed question to get the token count of the longest prompt; then build a prompt of the same length from a calibration question and cyclically concatenated calibration evidence, force generation of the full 768 tokens, and record peak memory, speed and whether chunked prefill is needed. The answer is not saved and does not enter any analysis.
3. **Judge check.** Load judges A and B in turn on the two cards (each occupying both cards) and record the weight hashes; first compare the outputs of the two attention implementations on the actual model (see 2.3), then trial-judge key-point recall and claim faithfulness on the answers from the previous step; the valid rate of key-point judging must be ≥ 90%. If judge A fails, the pipeline stops; if judge B fails, only the reason is recorded, the main experiment proceeds and the agreement analysis is missing.
4. **Collect information on earlier QA experiments (read-only).** Code and prompts of directories such as `multimodel_qa_v1/v2` and `single_gpu_fp16_v1`, and the generation models and LoRA adapters on the server, for comparison in the thesis; they do not affect this experiment.

Agreement between judges and humans is instead spot-checked on the sealed-set answers (Section 4.3), as reported content, not as a gate for running.

### 3.2 E1 input-matched retrieval re-evaluation

**Design matrix** (rows are models, columns are input settings):

| Model | A: first 160 tokens of the question (original E3) | B: full question, 512 window | C: no truncation (≤1024) | N: full input (eval_v1 setting) |
|---|---|---|---|---|
| Support-CE × 3 | Available (eval_v1) | **New scoring (main setting)** | New scoring | Same as B |
| Relevance-CE × 3 | Available | New scoring | New scoring | Same as B |
| bge-reranker-v2-m3 | New scoring (text identical to A) | New scoring (text identical to B) | Same as N | Available |
| Qwen3-Reranker-0.6B | New scoring (text identical to A) | New scoring (text identical to B) | Same as N | Available |
| (Optional) Qwen3-Reranker-4B | — | — | — | New scoring |
| (Optional) LLM zero-shot judgement | — | — | — | New scoring (Qwen3-4B computes the probability of "supports") |
| BM25 | Not affected by the input setting | | | Available |

**Notes:**

- **B is the main setting.** It matches CE's training format: full question, whole pair at most 512 tokens.
- **C tests whether going beyond 512 still helps.** mDeBERTa-v3 uses relative positions and can in theory accept longer inputs, but it never saw such lengths in training.
- **Meaning of N.** In eval_v1 the maximum length was 1024 for bge and 2048 for Qwen3-Reranker; nothing was truncated on the sealed set.
- **How "identical text" is done.** Truncate first by CE's SentencePiece rule, then decode the truncated tokens back into a string and give it to the public model. Both sides see the same text, only with different tokenisers.
- **Validity check.** For inputs where neither the question nor the evidence is truncated, the scores of the A, B and C encodings should be identical (maximum difference < 1e-5).
- For every input, "question truncated" and "evidence truncated" are recorded separately.
- Metrics and statistics as in eval_v1. Additionally reported by stratum: the 68 truncated and the 108 untruncated questions.
- About 3 GPU-hours.

### 3.3 E2 supervision signal × backbone

| Backbone \ training | Zero-shot | Relevance labels | Support labels | Shuffled labels (optional) |
|---|---|---|---|---|
| mDeBERTa-v3-base | — | Available × 3 | Available × 3 | New × 1 |
| bge-reranker-v2-m3 | Available | New × 3 (BGE-R) | New × 3 (BGE-S) | — |

**Training settings:** data and procedure exactly as in stage4.

- **Data:** the same `prepared.pt`, including the balanced training set of 26,432 pairs and the same dev and test splits.
- **Hyperparameters:** learning rate 1e-5, effective batch size 16, 3 epochs, 10% warm-up, weight decay 0.01, fp32, gradient checkpointing.
- **Checkpoint selection:** same dev selection rule and early stopping as stage4.
- **Output head:** bge keeps its original single-logit output and is trained with binary cross-entropy (equivalent to a two-class softmax); the classification head is not re-initialised.

**Outputs:**

- Precision, Recall and F1 of the support class on the stage4 test set, corresponding to thesis Section 3.5.2;
- retrieval metrics on the sealed set;
- conditions G10 and G11 for E3.

**Analyses:**

- the label effect (S − R) within each backbone;
- the backbone effect;
- the label × backbone interaction;
- BGE-S compared with zero-shot bge.

**Time:**

- In stage4, each training run took 3.5–3.8 hours. bge is about 3 times that, estimated at 10–13 hours per run.
- 6 runs, about 70 GPU-hours, about 1.5 days with two cards in parallel.
- A smoke test is run before the actual training to confirm memory and speed.

### 3.4 E3 main answer-level experiment

**QA model.**

- G1 is the Qwen3-4B-Instruct-2507 base model, without LoRA.
- Reason: the second-stage fine-tuning of the thesis's QA layer used evidence selected by Support-CE (thesis Eqs. 3.39–3.40). Comparing rerankers with the fine-tuned model would favour Support-CE.
- If the fine-tuned model exists on the server, it is reported separately as a supplementary "thesis pipeline" result.

**Prompts.**

- P-main follows the prompt structure of thesis Section 3.4.3.1, translated into a final Chinese version (Appendix A): answer from the evidence only, and say so explicitly when the evidence is insufficient.
- The closed-book condition uses the same structure without evidence, and likewise allows the answer "cannot answer reliably".

**Decoding.**

- Greedy decoding (do_sample=False; temperature, top_p and top_k unset), max_new_tokens = 768, fp32; prompts sorted by length and batched by a fixed token budget (G1 12,000; G2 9,000), at most 8 per batch; batching is identical on the two cards.
- GPU memory handling (fixed in advance): the usable memory of every generation process is limited to "the capacity of the smallest card on the machine − 1.5 GiB", so that out-of-memory conditions do not depend on the GPU used; on out-of-memory the batch is split in half and retried; if a single item still does not fit, chunked prefill (1,024 tokens per chunk) is used. Splitting and chunking are recorded per item in the answer file (oom_split, prefill_chunk).
- The P40 has no FlashAttention. For single inputs, transformers by default sends SDPA through the grouped-query attention (GQA) path, for which PyTorch can only fall back to the math kernel, whose memory grows quadratically with length. On GPUs with compute capability below 8.0 the pipeline instead expands K/V before calling SDPA, so that prefill can use a memory-efficient kernel. This changes only the computation path, not the model or the decoding settings.
- During token-by-token decoding, transformers expands the K/V cache at every step to the number of query heads (4× for Qwen3 and Hunyuan, 16× for GLM-4), and at larger batch sizes the memory traffic of this step exceeds that of the model weights. The pipeline instead treats the query heads of a group as several query positions and computes directly with the unexpanded K/V, which is mathematically equivalent; after each model load, the two implementations are first compared on the actual model (two left-padded inputs decoded for 6 steps each; the greedy results must be identical and the relative logit difference < 10⁻³), and if they differ the new path is not enabled. The judges additionally use chunked prefill (192–256 positions per chunk) to accommodate larger batches.
- Input and output token counts, time, and whether the length limit was reached are recorded.

**Evidence presentation.** Each passage is numbered E1, E2, …; the body is title plus context, without the source, in the rank order given by the method.

**Conditions** (k = 2, P-main, G1):

| ID | Condition | Evidence | Runs | Role |
|---|---|---|---|---|
| G00 | Closed-book | None | 1 | No-evidence reference (thesis Table 3.5) |
| G01 | BM25 Top-2 | Original pool ranks | 1 | Standard RAG (thesis Table 3.5) |
| G02 | 2 random passages | Drawn at random from the pool | 3 (fixed seeds) | Simple baseline: effect of arbitrary evidence |
| G03 | Full pool of 20 | BM25 order | 1 | No selection, long context |
| G04 | Relevance-CE (B) | Top-2 | 3 (seeds) | Relevance-label control |
| **G05** | **Support-CE (B)** | Top-2 | 3 (seeds) | **Main method (RAG with EEE in the thesis)** |
| G06 | Relevance-CE (A) | Top-2 | 3 | Effect of truncation on answers |
| G07 | Support-CE (A) | Top-2 | 3 | Effect of truncation on answers |
| G08 | bge-reranker-v2-m3 | Top-2 | 1 | Strong zero-shot baseline |
| G09 | Qwen3-Reranker-0.6B | Top-2 | 1 | Strong zero-shot baseline |
| G10 | BGE-S (fine-tuned with support labels) | Top-2 | 3 | Added after E2 is completed |
| G11 | BGE-R (fine-tuned with relevance labels) | Top-2 | 3 | Added after E2 is completed |
| G12 | (Optional) LLM zero-shot judgement | Top-2 | 1 | The QA model selects its own evidence |
| G13 | (Optional) Qwen3-Reranker-4B | Top-2 | 1 | Stronger zero-shot baseline |
| G03b | Full pool of 20, ordered by the mean score of the 3 Support-CE (B) seeds | — | 1 | Exploratory: position effects in long context |

**Aggregation:** for CE conditions, first average the 3 seeds within each question (consistent with H1); for G02, average the 3 draws.

### 3.5 E4 mechanism experiment: evidence built from the gold labels

| ID | Evidence | Construction rule | Usable questions |
|---|---|---|---|
| O1 | Supporting evidence | Select by the gold labels the 2 passages that cover the most points: first by full-support coverage, then by graded coverage, then by BM25 rank. If fewer than 2 exist, fill up with irrelevant candidates | 114 |
| O2 | Relevant but non-supporting | Candidates with relevance 1 and support 0 for all points; the 2 with the best BM25 ranks | 119 |
| O3 | Irrelevant | Relevance 0 and support 0 for all points | 172 |
| O4 | Same disease with a different pattern, or mismatched conditions | Candidates with "same disease with a different pattern or mismatched conditions" = 1 in the gold labels, at most 2 | 28 (exploratory) |

**Comparisons:**

- O1 corresponds to the gold-evidence condition in thesis Section 3.5.3.
- O1 − O2 tests directly whether "support ≠ relevance" holds at the answer level, independently of how good any reranker is.
- The main comparison is paired on the 70-question subset in which O1, O2 and O3 are all available. The subset is defined only by the availability of evidence, independently of the answer results.

**Evidence-use rate:**

- For each "question × point × condition", let s be the highest gold support level (0, 1, 2) of the given evidence for the point, and h the judgement of the answer. Define U_s = P(h = hit | s).
- The effect of s is estimated with a point-level logistic GEE (clustered by question). Covariates: whether the closed-book answer already stated the point, point type, question type.

**Gain–loss analysis:**

- Relative to closed-book, count for each condition the "points newly stated correctly" and the "points stated correctly before but wrong or missing after evidence is added".
- Separately by evidence category: supporting, relevant but non-supporting, mismatched conditions, irrelevant.

**Mediation analysis (exploratory):**

- Path: method → graded coverage of the selected evidence → KPR.
- Estimated on within-question differences, with bootstrap confidence intervals.
- Among the 176 questions only a few have different coverage under the two methods, so power is limited.

**Run order:** the evidence lists for O1–O4 are derived from the gold labels and are uploaded to the server for generation only after the answers of the other E3 conditions are frozen (see Section 8).

### 3.6 E5 robustness and sensitivity

| Item | Values | Conditions involved |
|---|---|---|
| Number of passages k | 1, 3, 5 (main analysis 2) | G01, G04, G05, G08, G09 |
| Prompt | P-alt: may use own knowledge, must cite the evidence numbers used, no abstention instruction | G00, G01, G02 (1 draw), G04, G05, G08, G09 |
| QA model | G2: GLM-4-9B-0414 (fp32, split over two cards; glm-edge-4b-chat if accelerate is missing or only one card is available) | G00, G01, G04, G05, G08, G09, O1–O3 |
| Decoding | Temperature 0.7, top-p 0.8, 3 samples per condition | G00, G01, G04, G05, G08 (optional; CE seed 42 only) |
| Threshold gating (thesis τ strategy) | Select only if the Support-CE score ≥ τ, at most 2; τ set on the stage4 dev set by the thesis rule. Public models calibrated on the same dev set to the same selection rate | Exploratory |
| Stratification | Question type (diagnosis, treatment principle); source stratum (fresh, heldout); truncated or not (68, 108); supporting evidence in the pool or not (114, 62) | All conditions, descriptive |
| Cross-dataset | The non-sealed question set used in E0 (labels from AI judges) | Main conditions repeated once, exploratory |
| Fine-tuned QA model | Thesis θ*QA (if available on the server) | G00, G01, G04, G05, G08, reported separately with a note that it favours Support-CE |

### 3.7 E6 open-pool end-to-end (optional)

**Purpose.** eval_v1 showed that the bottleneck is recall. The answer-level metrics need only the gold points, not the support labels of the candidate evidence, so the candidate pool can be enlarged without changing the gold labels.

**Retrieval:**

- BM25 top-100, same settings as the candidate pools;
- bge-m3 dense retrieval top-100;
- fused with RRF.
- The case's own original record is excluded (following the rule of `server_build_pools.py`); passages with character similarity ≥ 0.85 to the case are also excluded (the threshold used for question selection), to prevent near-duplicate passages from leaking the answer.

**Pipeline:** retrieval → each reranker selects the Top-2 → G1 generates the answer → KPR and SCR are computed. The new candidates have no gold labels, so at the retrieval level only the "silver" support rate given by the judge models is reported, clearly labelled as such.

This part corresponds to the end-to-end comparison in thesis Table 3.5.

### 3.8 E7 qualitative and error analysis

Errors are divided into six categories:

1. **Recall miss:** no supporting evidence in the pool.
2. **Selection miss:** supporting evidence in the pool, but not selected.
3. **Use failure:** supporting evidence selected, but the answer still misses the point.
4. **Misleading evidence:** after relevant-but-non-supporting or mismatched evidence is selected, the answer states a wrong point.
5. **Generation hallucination:** a wrong claim unrelated to both the evidence and the question.
6. **Over-abstention.**

For each category, 15–20 cases are judged blind, with illustrative case figures.

---

## 4. Evaluation metrics and statistical methods

### 4.1 Key-point recall judging (KPR)

**Input:** the question, the answer, the gold points (type and text). The evidence and the condition name are not provided, and the order of the conditions is shuffled.

**Judgement:** each point is judged as one of four categories:

| Label | Meaning |
|---|---|
| 2 Hit | The point is stated explicitly, or with a recognised synonym. Synonym rules follow Section 4.1 of the annotation guidelines v1.1 |
| 1 Partial | Only part of a compound point is stated, or a broader or narrower related concept |
| 0 Not mentioned | |
| X Contradicted | A conclusion incompatible with the point is stated |

- When judging 2 or 1, the judge must quote a span of the answer. The programme checks that the span really occurs in the answer; otherwise the item is judged again.
- The judge also decides whether the answer abstains as a whole.

**Procedure:**

- Judge A (main judge): `GLM-4-9B-0414`, judges all answers. Judge B (second judge): `tencent/Hunyuan-7B-Instruct`, judges only the answers, under all conditions, to about 25% of the questions selected deterministically by a hash of the question ID (sha256("B:"+ID) mod 4 = 0; 45 sealed questions). Both are from families different from the main QA model Qwen3, with fixed weights (SHA256 recorded), and run on the server in fp32 with greedy decoding. B is a hybrid reasoning model; following the official instructions, `/no_think` is prepended to the user message to turn off thinking mode. Output stops at the first complete JSON object.
- Judge A is the same model as G2, so A's labels on G2 answers are self-judgements. G2 is used only as a robustness check (comparing differences between conditions within G2), cross-checked on judge B's subset.
- Non-compliant outputs (invalid JSON, missing points, quotes that are not verbatim) are rewritten after the problem is pointed out to the judge, at most 3 times. All outputs are cached.
- A "hit" or "partial" whose quote check fails (the quote is not verbatim from the answer, still after retries) counts as "not mentioned".
- The primary metric uses judge A's labels. Judge B's subset is used for: agreement between A and B (κ); and, as a sensitivity analysis, recomputing the main comparisons on the same questions with A's labels, B's labels, and the lower of the two (X < 0 < 1 < 2).

### 4.2 Claim-faithfulness judging (SCR, GHR)

1. Judge 1 splits the answer into independent claims. The split is done once and shared by both judges. Claims that only restate the case findings in the question are marked separately and not counted in the denominator.
2. Each claim is judged against the given evidence as "supported", "contradicted" or "unsupported". The criterion for "supported" matches "full support" in the annotation guidelines and requires comparable applicability conditions.
3. Judge B re-judges its subset of questions (about 25%), and agreement is reported.

### 4.3 Human spot check

- **Sealed set:** stratified by condition and question type, about 120 answers are sampled (about 370 point judgements and 600 claims). The principal investigator and a colleague judge them blind, and their κ and agreement with the judge models are reported.
- **Note:** the checkers have no TCM background. What they check is "whether the answer expresses the gold point" and "whether the evidence states the claim", not whether it is correct in TCM terms. If possible, a TCM collaborator judges 30–50 of these answers blind.

[Translator's note: this spot check was not carried out; see Supplementary Material S2, item "Human audit".]

### 4.4 Statistical methods

**Unit of analysis and aggregation:** the unit of analysis is the question (176). For CE conditions, the 3 seeds are first averaged within each question; for random conditions, the 3 draws are first averaged.

**Tests:**

- **Paired bootstrap:** 10,000 resamples over questions, seed 20261026, 95% percentile CI; p values computed as in eval_v1.
- **Equivalence and non-inferiority:** TOST, comparing the 90% CI with the preset margins.
- **Binary outcome (all-correct rate):** McNemar test.
- **Point level:** logistic GEE, clustered by question, exchangeable correlation structure.
- **Robustness supplement:** Wilcoxon signed-rank test (thesis Section 3.5.5).

**Multiple comparisons:** Holm correction within each hypothesis family.

| Family | Content |
|---|---|
| Primary | H3 tested alone, α = 0.05 |
| B: answer-level comparisons | G05 vs. each of G01, G00, G02, G08, G09 |
| C: faithfulness | SCR, GHR; G05 vs. each of G04, G01 |
| D: mechanism | O1 − O2, O2 − G00, O1 − G00, O3 − G00 |
| E: retrieval fairness | H1a, H1b, H1d, corrected separately by metric |
| F: labels × backbone | H2a, H2b |
| Others | All exploratory; only effect sizes and CIs are reported |

**Missing data:**

- Failed generations are retried once; those still failing count as abstentions (KPR = 0).
- Non-compliant judge outputs are retried 3 times; those still failing count as misses.
- Both counts are reported, together with a sensitivity analysis excluding these questions.

### 4.5 Statistical power

| Comparison | Metric | SD of the paired difference | Minimum detectable difference (80% power, two-sided 0.05) | Source |
|---|---|---|---|---|
| Support-CE vs Relevance-CE | Strict coverage | 0.139 | 0.029 | Measured in eval_v1 |
| Support-CE vs bge | Strict coverage | 0.075 | 0.016 | Measured in eval_v1 |
| Support-CE vs bge | Graded coverage | 0.123 | 0.026 | Measured in eval_v1 |
| Answer level, 176 questions | KPR | Assumed 0.20–0.25 | 0.042–0.053 | Measured value given in the results report |
| O1 vs O2, 70 questions | KPR | Assumed 0.25 | About 0.084 | Measured value given in the results report |

**Conclusions:**

- The non-inferiority margins of H1b (0.02 and 0.03) have adequate power.
- At the answer level, only differences of 4–5 percentage points or more can be detected reliably. The equivalence margin of H7 is therefore preset at 0.05; the results report gives the measured SD of the paired differences and the corresponding minimum detectable difference.
- Smaller differences cannot be distinguished with this dataset, and the paper must say so.

---

## 5. Baselines and fair comparison

### 5.1 Baselines

| Type | Method | Source | Note |
|---|---|---|---|
| No-evidence reference | Closed-book | Qwen3-4B-Instruct-2507 | Thesis Table 3.5 |
| Simple baselines | 2 random passages; full pool of 20 | — | The former tests the effect of "any evidence", the latter whether "selection is needed" |
| Classical method | BM25 Top-2 (character bigrams, k1 = 1.2, b = 0.75) | Original pool ranks | Standard RAG |
| Strong zero-shot baselines | bge-reranker-v2-m3; Qwen3-Reranker-0.6B; (optional) Qwen3-Reranker-4B, LLM zero-shot judgement | Official weights, fixed snapshots | Maximum lengths as in eval_v1 (1024, 2048) |
| Strong baselines fine-tuned on the same data | BGE-R, BGE-S | E2 | Answers the objection of "comparing only with weak controls" |
| Learned control | Relevance-CE | Existing stage4 checkpoints | |
| Upper and lower bounds | Gold supporting evidence (upper bound); irrelevant evidence (lower bound) | E4 | |

### 5.2 Principles of fair comparison

- **Same evaluation conditions:** all methods use the same questions, the same candidate pools, the same evidence text and the same evaluation scripts, with the same Top-k budget.
- **Input:**
  - Main comparison: every model sees the full question, each with its own maximum length: 512 for CE (its training length), 1024 for bge, 2048 for Qwen3-Reranker. The latter two need no truncation at these lengths.
  - A further text-matched control: the public models see exactly the same truncated text as CE.
- **Training:**
  - All fine-tuned models use the same training data, the same split and the same checkpoint-selection rule.
  - No hyperparameter is tuned on the sealed set.
- **Generation and judging:** all conditions share the same QA model, the same prompts, decoding settings and judging protocol.
- **Report parameters and reranking time:** mDeBERTa-v3-base has about 280 million parameters, bge-reranker-v2-m3 about 570 million, Qwen3-Reranker-0.6B about 600 million.

---

## 6. Ablation design

### 6.1 Component ablation

| Variant | Condition | External evidence | Top-2 selection | Learned reranking | Support labels | Full question | Expectation |
|---|---|---|---|---|---|---|---|
| Full approach | G05 | ✓ | ✓ | ✓ | ✓ | ✓ | Best, or on a par with the strong baselines |
| Without support labels | G04 | ✓ | ✓ | ✓ | ✗ | ✓ | KPR drops (H3) |
| Without the full question | G07 | ✓ | ✓ | ✓ | ✓ | ✗ | Drops, mainly on the 68 truncated questions |
| Without learned reranking | G01 | ✓ | ✓ | ✗ | ✗ | — | Drops (H4) |
| Without selection | G03 | ✓ | ✗ | ✗ | ✗ | — | Longer context; KPR uncertain, SCR expected to drop |
| Without external evidence | G00 | ✗ | — | — | — | — | Reference |
| Stronger backbone | G10 | ✓ | ✓ | ✓ | ✓ | ✓ | If higher than G05, the backbone is the bottleneck |

### 6.2 Hyperparameter sensitivity

| Hyperparameter | Values | Purpose |
|---|---|---|
| Number of passages k | 1, 2, 3, 5, 20 | Relation between the selection budget and answer quality and context length |
| Maximum question tokens | 160, full | Cost of truncation |
| Maximum input length | 512, 1024 | Whether going beyond the training length helps |
| Decoding temperature | 0, 0.7 | Whether the conclusions depend on the decoding |
| Selection threshold τ | The value set on dev and one step above and below | Gating strategy (exploratory) |

### 6.3 Cross-dataset

The main conditions are repeated on the non-sealed question set used in E0 (labels from AI judges), as an exploratory analysis. TCMBench in the thesis is used to evaluate the general ability of the QA model and is outside the scope of this protocol.

### 6.4 Qualitative analysis

See E7.

---

## 7. Decision table for interpreting the results

Written in advance: each pattern of results corresponds to one conclusion, to avoid choosing the wording after the fact.

| Pattern of results | Conclusion | What the paper may say |
|---|---|---|
| O1 > O2; G05 > G04; G05 equivalent to or better than G08 and G09 | "Support" matters at the answer level, and Support-CE turns it into answer gains | Support-CE has a positive effect; also state that the size is bounded by the pool upper bound |
| O1 > O2; G05 > G04; no difference between G05 and G08, G09 | Support labels beat relevance labels, but existing strong rerankers achieve the same | The contribution is the operational definition of "support" and the evaluation framework. If BGE-S > zero-shot bge, support labels also help strong models |
| O1 > O2; G05 ≈ G04 | Supporting evidence itself is useful, but Support-CE can select too little of it (pool upper bound) | The bottleneck is recall; the paper's focus shifts to recall (E6) |
| O1 ≈ O2, or both ≈ closed-book | This QA model hardly depends on evidence, or "relevant" and "supporting" evidence act on it in the same way | No answer-level claim about Support-CE can be made; look again with G2 or P-alt |
| O2 < G00 | Relevant but non-supporting evidence misleads the answers | Strong support for "relevance ≠ support"; shows the value of filtering out non-supporting evidence |
| In E1, S-B clearly better than S-A | eval_v1 underestimated CE | Use the full question from now on |
| In E1, S-B still below the public models (non-inferiority not met) | Support-CE cannot be promoted as a strong reranker | Look at BGE-S in E2 |

---

## 8. Pre-registration and leakage-prevention procedure

1. **Finalisation and freezing.** Prompts, the judging protocol, δ and all scripts are finalised before registration; the local launcher computes the SHA256 of all files and writes them to FREEZE_MANIFEST_v2.json.
2. **OSF registration.** Register eval_v2 on OSF: this protocol, the prompts, the judging protocol, the list of conditions and the script hashes. The registration states that the results of eval_v1 are known and that some hypotheses were derived from them. Only after registration is any new scoring or generation done on the sealed set.
3. **E1 reranker scoring.** Run once, keeping eval_v1's safeguard of "pass the calibration set first, then run the sealed set".
4. **Conditions that do not need the gold labels.** After these conditions of E3 and E5 are generated, their hashes are computed and they are frozen.
5. **O conditions.** In the analysis environment, the candidate-ID lists for O1–O4 are derived from the gold labels (IDs only, no labels), frozen and uploaded to the server; after the answers are generated they are frozen in the same way.
6. **Judging.** Judging is done on the server with local open-source models; no data are sent outside. The judges use only the questions, the answers, the reference points (final_questions.json, on the server since eval_v1) and the given evidence; the evidence-support labels do not go to the server. All judge outputs are cached.
7. **Statistics.** The server first produces a summary without the evidence gold labels (summarize.py); the retrieval-level and mechanism analyses that use the evidence gold labels are done after the results are downloaded. If code changes after registration, the launcher automatically records the changed files in 变更记录_v2.txt [change record], and the paper reports them truthfully.
8. **E2.** Training uses only stage4 data; after the checkpoints are frozen, the sealed set is scored and G10 and G11 are generated.

> The sealed set was already used in eval_v1. eval_v2 is a new pre-registered analysis on the same test set, not an entirely new confirmatory test. Before formal submission, it is still advisable to re-check the main conclusions on a new sealed set annotated by TCM experts.

---

## 9. Compute resource estimate

### 9.1 Unit costs

| Item | Estimate |
|---|---|
| GPU | One Tesla P40 24 GB and one 22 GB |
| CE re-scoring (3,520 pairs) | About 6 minutes per checkpoint (372 seconds measured in eval_v1) |
| bge fine-tuning | About 10–13 hours per run (estimated from the measured mDeBERTa time) |
| Generation (Qwen3-4B, fp32, token budget 12,000) | About 2–5 seconds per item, about 20 GB of GPU memory; measured values recorded in E0 |
| Local judging (fp32, two cards) | Judge A about 4–11 seconds per request (key-point judging fastest, claim splitting slowest); judge B similar; measured values recorded in E0 |

### 9.2 Total workload

| Category | Runs | Answers or scored items | GPU-hours |
|---|---|---|---|
| E1 re-scoring | 6 CEs × 2 encodings, public models × 2 settings | About 56,000 pairs | About 3 |
| E2 training and scoring | 6 runs (plus 1 optional) | — | About 75 |
| E3 main conditions (G00–G09, G03b) | 21 | 3,696 | About 4 |
| E3 supplementary (G10, G11) | 6 | 1,056 | About 1 |
| E4 (O1–O4) | 4 | 433 | About 0.5 |
| E5: k sensitivity | 27 | 4,752 | About 4 |
| E5: P-alt | 11 | 1,936 | About 2 |
| E5: G2 QA model | 10 + O1–O3 | 2,165 | About 5 |
| Local judge A (key-point recall, claim splitting and checking) | About 25,000 requests | — | About 85 |
| Local judge B (about 25% of questions) | About 5,000 requests | — | About 16 |
| **Total (excluding optional items and E6)** | | **About 14,000 answers** | **About 195** |
| E6 (optional) | | | About 20 |

### 9.3 Resource planning

- About 195 GPU-hours in total, about 4.5–5 days of continuous running on two cards; E2 fine-tuning and local judging each take about 40%. The judges call no paid API, so there is no extra cost.
- Human effort: about 6–8 hours for the spot check of sealed-set answers (after the results are available).

---

## 10. Reproducibility

### 10.1 Hardware

```
- GPU: Tesla P40 24 GB × 1, Tesla P40 22 GB × 1 (host <gpu-host>)
- Local machine: Windows, PowerShell 5.1 (only uploads, starts, checks progress and downloads results)
```

### 10.2 Software

```
- Server: Linux (glibc 2.35), Python 3.10.12, torch 2.7.1+cu118, transformers 5.19.0, sentencepiece
- Statistical analysis: summarize.py on the server (without the evidence gold labels); analyses that use the evidence gold labels are done after the results are downloaded
```

### 10.3 Random seeds

- CE, BGE-S, BGE-R: 42, 2026, 3407.
- Random-evidence condition: 3 fixed seeds, fixed at registration.
- Bootstrap: 20261026.
- Generation uses greedy decoding with deterministic algorithms enabled; E0 verifies that two runs on the same input give identical outputs, character for character.

### 10.4 Code and records

| Script | Where it runs | Purpose |
|---|---|---|
| server_score.py v2 | Server | Adds the A, B and C encodings and the "identical text" mode; records question truncation and evidence truncation separately |
| train_bge_stage4.py | Server | E2 fine-tuning, procedure aligned with train_stage4.py |
| server_generate.py | Server | Generates answers, records token counts and time |
| analysis/build_oracle_lists.py | Analysis environment (not via the server) | Derives the candidate-ID lists for O1–O4 from the gold labels (IDs only) |
| server/judge.py | Server | Local judging: batching by length, rewriting when non-compliant, caching, resuming (the OpenAI-compatible API code path is kept but not used by default) |
| server/p40.py | Server | Attention and memory settings on the P40 (see 2.3) |
| server/summarize.py | Server | Quick summary and pre-registered comparisons without the evidence gold labels |
| server/pipeline.py, run_v2.sh | Server | Automatic pipeline: dependency scheduling, allocation over two cards, retry on failure, resuming |
| tools/launcher.ps1 and three .bat files | Local machine | One-click upload and start, check progress and download results, stop |
| Retrieval-level and mechanism analysis scripts | Analysis environment | All analyses that use the evidence gold labels, done after the results are downloaded |

The local machine has no Python, so only PowerShell scripts run locally; steps that use the evidence-support labels are done in the analysis environment (where the eval_v1 evaluation was also done).

Every run writes a run manifest: settings hash, weight hashes, input hashes, runtime environment. Hashes of all outputs are computed and collected in FREEZE_MANIFEST_v2.

---

## 11. Timeline

| Phase | Content | Planned time | Status |
|---|---|---|---|
| 1 | Finalise, freeze, register on OSF; start the pipeline (E0 completes automatically) | Day 1 | [ ] |
| 2 | E1 scoring; E3 and E4 generation and judge A (main results) | Days 1–2 | [ ] |
| 3 | E5 judging, G2 generation and judging, judge B subset; E2 fine-tuning, generation and judging of G10 and G11 | Days 3–5 | [ ] |
| 4 | Retrieval-level and mechanism analyses using the evidence gold labels, human spot check, results report | About 1 week after the results are downloaded | [ ] |
| 5 | E6 (optional), E7 qualitative analysis, figures for the paper | Afterwards | [ ] |

---

## 12. Risks and responses

| Risk | Likelihood | Impact | Response |
|---|---|---|---|
| Answer-level differences too small: low pool upper bound, the model relies mainly on its own knowledge | High | H3 not significant | Pre-registered equivalence tests and conditional subsets; E4 gives an upper bound; report truthfully; E6 enlarges the candidate pool |
| Judge models unreliable on TCM synonyms (local judges are only 7–9B) | Medium | Primary metric biased | Protocol follows the synonym rules of the annotation guidelines; second-judge subset and sensitivity analysis; report literal recall; human spot check (κ between judges and humans is the main evidence of judge reliability); E0 gate |
| The instruction "say so when the evidence is insufficient" leads to many abstentions | Medium | KPR dominated by abstentions | Report answer coverage and selective accuracy; P-alt without an abstention instruction as a control |
| The QA model ignores the evidence | Medium | All RAG conditions close to closed-book | The mechanism experiment can detect this; switch to G2; use P-alt |
| P40 does not support bf16, fp16 is slow | Certain | Slow | fp32 throughout; 7–9B models split over two cards; no K/V expansion during decoding; E0 records memory and speed |
| Gold-label information leaks to the server | Low | Violates a design principle | Evidence-support labels do not go to the server; O lists contain IDs only and are read by the programme only after the non-O conditions are frozen |
| Relevance-CE unstable | Known | Main comparison questioned | Average over seeds and report each seed; E2 provides a strong relevance control |
| A fine-tuned QA model favours Support-CE | Known | Unfair comparison | Main analysis uses the base model; the fine-tuned model is reported separately |
| Judge model versions | Low | Hard to reproduce | Local weights fixed and hashes recorded; greedy decoding, so results can be reproduced by rerunning; all outputs cached |
| Pre-training corpora may contain similar case records | Medium | Closed-book scores inflated | Affects all conditions equally; closed-book results reported; stratified by fresh and heldout |
| bge fine-tuning fails or runs out of memory | Low | E2 delayed | Smoke test first; if necessary reduce to a single seed and state this in the paper |

---

## 13. Remarks

**Relation to eval_v1.** The frozen files and results of eval_v1 remain unchanged and are reported as they are. Everything new in v2 is registered separately.

**Wording about the gold labels.** All conclusions state that the gold labels were annotated and adjudicated by AI agents; they are not called expert gold labels.

**Correspondence with thesis Section 3.5.**

| Thesis item | Condition in this protocol |
|---|---|
| Table 3.5: Closed-book | G00 |
| Table 3.5: Standard RAG | G01 (same budget), G03 (all candidates) |
| Table 3.5: RAG with EEE | G05 |
| Table 3.5: Full cascaded (with HDV) | Outside the scope of this protocol; the answers and evidence of this protocol can be passed directly to HDV for evaluation |
| Table 3.6: Without EEE | G01, G03 |
| Table 3.6: Without abstention instruction | P-alt |
| Table 3.6: Gold-evidence training only | Needs a fine-tuned QA model; optional item in E5 |
| Table 3.6: Without hard negatives | Needs retraining of CE; can be added to the E2 training queue (optional) |
| 3.5.3: gold-evidence condition | O1 |

---

## 14. v2.1 revision record (before registration)

To make the experiments run fully automatically with fewer manual steps, and based on checks of the models and interfaces, the following changes were made to v2.0 before registration:

| Item | v2.0 | v2.1 | Reason |
|---|---|---|---|
| Where judging runs | Local PowerShell | Python on the server (standard library only) | No Python on the local machine; the server has internet access. The reference-point file has been on the server since eval_v1; the evidence-support labels still do not go to the server |
| O1–O4 lists | Uploaded after the non-O conditions are frozen | Uploaded with the code, but read by the programme only after the non-O conditions are frozen (e3_freeze) | Avoids a manual upload midway; the lists contain candidate IDs only |
| E0 | Pilot, may be used to adjust prompts, protocol and δ | Technical check only; prompts, protocol and δ = 0.05 finalised before registration | Avoids manual confirmation midway, and avoids changing the protocol after seeing data |
| Judge models | Not specified | A: DeepSeek deepseek-flash; B: Volcano Ark doubao-seed-2-1-pro-260915; C (optional): Qianfan ernie-5.1 | Different families, thinking mode can be turned off; B has a dated fixed version. DeepSeek has no fixed version number, so its judging must be completed in a short period and the model name in the response recorded |
| Handling of disagreements | Decided by a third model | Decided by C if available, otherwise the lower label | The third key becomes optional |
| Quote check | Re-judge on failure | "Hit/partial" still failing after re-judging counts as "not mentioned" | Conservative and can be specified in advance |
| max_new_tokens | 512 | 768 | Fewer truncated answers |
| Batching | Batch size 8 | Fixed token budget (G1 12,000, G2 9,000), at most 8 items | Identical batching on the two cards (24 GB, 22.5 GB), so results are reproducible |
| GPU memory | Not specified | The usable memory of every generation process is uniformly "smallest card capacity − 1.5 GiB"; split the batch when short of memory, chunked prefill (1,024 tokens) when a single item still does not fit; on the P40, expand K/V before calling SDPA; E0 runs a memory stress test with the longest prompt | Long prompts with 20 passages easily run out of memory on the P40; a uniform cap makes the splitting behaviour identical on the two cards |
| G2 | Another family, 7–9B | GLM-4-9B-0414 (fp32, two cards, about 18 GiB of weights per card); glm-edge-4b-chat if accelerate is missing or only one card is available | Strong in Chinese, natively supported, no licence application needed; fp16 too slow on the P40, so fp32 only |
| Optional baselines | Optional | Qwen3-Reranker-4B (G13) and LLM zero-shot judgement (G12) run by default | Low cost (about 0.5 GPU-hours each), completes the strong baselines |
| Scope of claim faithfulness | E3 main conditions | Conditions with evidence under G1, the main prompt and k = 2, plus O1–O4, G10, G11; judge B re-checks a fixed 20% selected by aid hash | Controls judging costs |
| Mode of running | Manual, in stages | Automatic pipeline on the server: schedules the two cards by dependency, main line (E0, E1, E3, E4) first, E2 uses only one card before the main line is completed; failures retried automatically once; resumes after interruption | No manual supervision needed |

Fixed random seeds: random-evidence condition 101, 202, 303; presentation order of the evidence in the O conditions 7; order of judge requests 20261026, 20261027; bootstrap 20261026.

---

## 15. v2.2 revision record (before registration): judges changed to local open-source models

To avoid paid APIs, the two judges were changed before registration to open-source models running on the two P40s of the server. Prompts, judging rules, quote-check rules and δ are unchanged.

| Item | v2.1 | v2.2 | Reason |
|---|---|---|---|
| Judge A | DeepSeek API, judges all answers | `GLM-4-9B-0414`, local fp32, judges all answers | Free; a family different from the main QA model Qwen3; stable instruction following and JSON output, vocabulary and chat template natively supported; only 2 K/V heads, so large batches and fast; fixed weights, reproducible by rerunning. Same model as G2, so judging of G2 answers is self-judging |
| Judge B | Doubao API, judges all answers | `tencent/Hunyuan-7B-Instruct`, local fp32, thinking mode turned off with `/no_think`, judges only the answers, under all conditions, to about 25% of the questions (by question-ID hash; 45 sealed questions) | A third family (Tencent), different from G1, G2 and judge A; local judging is slow (fp32), and double judging of everything would take about 2 more days; sampling by question keeps the conditions comparable in pairs |
| Judge C | Qianfan API, optional, decides disagreements | Dropped | No API is used any more |
| Labels for the primary metric | A and B combined (disagreements decided by C or the lower label) | Judge A's labels | B covers only a subset |
| Agreement and sensitivity | A–B κ on everything; sensitivity with A only | A–B κ on B's subset; main comparisons recomputed on the same questions with A, B and the lower of the two | The judges are smaller, so it must be shown that the conclusions do not depend on one judge |
| Re-judging of claim checks | B re-checks 20% (by aid) | B re-checks its subset of questions (about 25%) | Consistent with the key-point judging subset |
| Output control | JSON mode of the API | JSON requested only through the prompt; output stops at the first complete JSON object; when non-compliant, the problem is pointed out to the judge and it rewrites, at most 3 times | Local models have no JSON mode |
| Attention implementation | — | No K/V expansion during decoding; the judges use chunked prefill (see 2.3); compared on the actual model after loading | Faster decoding with large batches on the P40 |
| Scheduling | E2 may use one card before the main line is completed | Judges occupy both cards; cards allocated strictly by priority; E2 starts only after the main line (including the judging of the main conditions) is completed, with fine-tuning on both cards at once | Avoids long waits for cards by two-card jobs |
| Single-card fallback | G2 switches to glm-edge-4b-chat | Judge A switches to glm-edge-4b-chat and judge B to Hunyuan-4B-Instruct (only if accelerate is missing or fewer than two GPUs are available) | Loading on two cards needs accelerate |
| Duration and cost | About 2–3 days; API about 400–600 yuan | About 5 days; no API cost | Local judging takes about 100 GPU-hours |

With smaller judge models, the reliability of judging depends more on the human spot check (Section 4.3): the paper uses the κ between judge A and human judgements as the main evidence of judge reliability, and reports A–B agreement and the sensitivity analysis.

[Translator's note: the human spot check was not carried out; see Supplementary Material S2, item "Human audit".]

---

## Appendix A: Generation prompts (final; identical to server/config/prompts.json)

[Translator's note: the prompts were used in Chinese. The English below is a translation; the Chinese text is authoritative.]

**P-main, with evidence:**

```
[system]
You are a TCM clinical question-answering assistant. Answer the question using only the evidence provided below; do not add information that has no basis in the evidence.
If the evidence is not sufficient to answer, state clearly "The available evidence is not sufficient to give a well-founded answer", and state what kind of evidence is missing.

[user]
[Question]
{question}

[Evidence E1]
{evidence_1}

[Evidence E2]
{evidence_2}

[Answer requirements]
Please give a concise, evidence-based answer.
```

**P-main, closed-book:**

```
[system]
You are a TCM clinical question-answering assistant. Answer the question based on TCM knowledge, and do not make up information you are not sure about; if you cannot answer reliably, say so clearly.

[user]
[Question]
{question}

[Answer requirements]
Please give a concise answer.
```

**P-alt, with evidence:**

```
[system]
You are a TCM clinical question-answering assistant. Answer the question by combining the evidence provided below with your TCM knowledge. When you use a piece of evidence, mark its number at the end of the sentence, e.g., [E1].

[user]
(Question and evidence as in P-main)

[Answer requirements]
Please give a concise answer.
```

**P-alt, closed-book:** the system message becomes "You are a TCM clinical question-answering assistant. Answer the question based on TCM knowledge."; the user message is the same as in P-main closed-book.

---

## Appendix B: Judging protocol (skeleton; the full final text is in server/config/judge_prompts.json)

**Key-point recall judging:**

```
[system]
You are a strict grader. Only judge "whether the answer expresses each reference point"; do not assess who is medically right,
and do not complete the answer with knowledge beyond the reference points.

[user]
[Question] {question}
[Reference points]
P1 ({type}): {text}
P2 ({type}): {text}
……
[Answer to be judged] {answer}

[Criteria]
2 = The answer states the point explicitly, or uses a recognised synonym (e.g., liver–kidney insufficiency = liver–kidney deficiency); for a compound point, all parts must be stated.
1 = Only part of a compound point is stated; or a broader or narrower related concept is stated; or there is a shared pathological component without equivalence (e.g., "blood deficiency" relative to "liver depression with blood deficiency").
0 = Not addressed.
X = A conclusion incompatible with the point is stated (e.g., a different syndrome pattern, an opposite treatment principle).
If the answer as a whole refuses to answer or only states that the evidence is insufficient, without giving any point, abstain = true.

[Output] Output JSON only:
{"P1": {"label": "2", "quote": "span of the answer"}, "P2": {...}, "abstain": false}
When label is 1 or 2, quote must be copied verbatim from the answer.
```

**Claim-faithfulness judging:**

1. **Splitting (judge 1):** split the answer into independent claims, each containing only one checkable judgement. Claims that only restate the case findings in the question are marked "restatement of the question".
2. **Judgement (judges 1 and 2):** each claim that is not a restatement of the question is judged against the [Evidence]:
   - S (supported): the evidence states it explicitly, or with a recognised synonym, and the applicability conditions are comparable with this case;
   - C (contradicted);
   - N (the evidence does not address it, or is not sufficient to support it).
3. **Output:** a JSON list.

---

## Appendix C: Quality checklist

- [x] Every hypothesis has a corresponding experiment (Section 1)
- [x] The values of all independent variables are clearly defined (2.1)
- [x] The control variables are fully listed (2.3)
- [x] The metrics cover answer quality, faithfulness, abstention and efficiency (Sections 2 and 4)
- [x] The baselines include simple baselines, a classical method, strong zero-shot models, models fine-tuned on the same data, and upper and lower bounds (Section 5)
- [x] The ablations cover each component of the approach (6.1)
- [x] The compute estimate includes a safety margin (Section 9)
- [x] The reproducibility information is complete (Section 10)
- [x] The statistical tests and multiple-comparison corrections are fixed (4.4)
- [x] Prompts, judging protocol and δ finalised before registration (v2.1); judge models changed to local open-source models before registration (v2.2)
