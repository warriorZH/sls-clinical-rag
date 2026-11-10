# Data files

All files are UTF-8 JSON Lines (one JSON object per line) unless noted otherwise. Chinese is the language of all question and passage content.

## `data/sealed/`

| File | Fields |
|---|---|
| `questions.jsonl` (176) | `qid` (ST001–ST180); `question_type` (`diagnosis` or `principle`); `stratum` (`fresh_med` = cases never used before; `heldout_med` = untouched cases of the held-out split); `origin_task_id` (record ID in the source collection); `required_points`: a list of `{pid, type, text}` where `type` is 疾病诊断 (disease), 证候 (syndrome pattern), 病机 (mechanism) or 治疗原则 (treatment principle) |
| `excluded_questions.jsonl` | The 4 questions excluded during verification |
| `candidate_pools.jsonl` (3,520) | `qid`, `rank` (BM25 rank 1–20), `document_id`, `source` (corpus file and field), `start_char`, `end_char`, `sha256` (of the passage text), `bm25_score` |
| `SELECTION_MANIFEST.json` | Score-blind selection rule, strata, exclusion counts and the selection hash |
| `POOL_MANIFEST.json` | Retrieval settings (character-bigram BM25, k1 = 1.2, b = 0.75, top 20, own record excluded) |

## `data/external/`

These files are derived from TCM-BEST4SDT (Li et al., CC BY 4.0).

| File | Fields |
|---|---|
| `questions.jsonl` (300) | `qid` (EX001–EX300), `source_id` (case ID in `TCM_SDT.json`), `question_type`, `stratum`, `background`, `question_final` and `required_points` (with `source_field`) |
| `reference_answers.jsonl` | The benchmark's expert reference fields (disease, syndrome, cause, mechanism, nature, location, principle, prescription and herbs) |
| `candidate_pools.jsonl` | As for the sealed set; queries retrieve from the same corpus |
| `EXT_MANIFEST.json`, `BUILD_MANIFEST.json`, `pool_filter_log.jsonl` | Source commit and hash, conversion rules and pool filtering log |

## `data/training/`

`pairs_train.jsonl` (26,432; class-balanced for support labels, used for all selector training), `pairs_train_full.jsonl` (30,487; before balancing), `pairs_dev.jsonl` (4,978), `pairs_test.jsonl` (5,030).

| Field | Meaning |
|---|---|
| `task_id`, `candidate_id`, `group` | Question, passage and case-group IDs. Splits are disjoint by group |
| `source` | `med` (case records) or `hallu` |
| `y_relevance`, `y_support` | Binary labels. `y_support = 1` implies `y_relevance = 1`. The label flip changes exactly the pairs with `y_relevance = 1, y_support = 0` |
| `covers` | IDs of the answer points the passage fully supports |
| `constructed` | Whether the passage was synthetically constructed (always false in these files) |

## Results files (`results/`)

- **Scores:** `qid`, `document_id`, `rank` and `score` (with truncation flags where relevant).
- **Selections:** `cond`, `qid` and `docs` (selected `document_id`s in presentation order).
- **Answers:** `aid` (answer ID), `cond`, `qid`, `gen` (G1 = Qwen3-4B-Instruct-2507; G2 = GLM-4-9B-0414), model revision, `prompt` (`main` = evidence-only; `alt` = knowledge-permitted), `k`, `docs` and `answer`.
- **KPR judgements:** `aid`, `qid`, `cond`, `judge`, `labels` (per point: 2 stated, 1 partial, 0 not stated, X contradicted), `quotes` (supporting span from the answer), `quote_ok` and `abstain`.

Condition codes follow the Stage 2 protocol (`docs/registration/osf/stage2_e37na/实验设计方案_v2.md`; external-set codes X00–X09 in the amendment `docs/registration/stage2_amendment_v2.3_protocol_zh.md`; English translations in `docs/registration/en/`):

- G00 closed book; G01 BM25; G02 random; G03 all 20 candidates.
- G04 and G05: relevance-CE and support-CE, the main comparison. G06 and G07: their stage 1 versions.
- G08 bge-reranker-v2-m3 zero-shot; G09 Qwen3-Reranker-0.6B zero-shot.
- G10 and G11: bge support and bge relevance. G12 LLM selector; G13 Qwen3-Reranker-4B zero-shot.
- Prefixes: `K1`, `K3`, `K5` vary the number of passages; `PA_` uses the knowledge-permitted prompt; `G2_` uses the second generator.
- X-codes are the external-set counterparts.
- Stage 3: Q06 and Q4B are Qwen3-Reranker-0.6B and 4B, with suffix N (zero-shot), S (support) or R (relevance).
- Oracle conditions (O1–O4) are withheld with the support labels.
