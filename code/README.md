# Code

The scripts are released exactly as they were run. Server addresses and user names have been replaced by placeholders. Most comments and log messages are in Chinese.

| Folder | Stage | Main entry points |
|---|---|---|
| `01_sealed_set_construction/` | Building the sealed set | `build_sealed_testset.py` (score-blind selection), `draft_points.py` (rule-based point drafts), `server_build_pools.py` (BM25 pools), `make_phase2_csv.py`, `compute_agreement.py`, `adjudicate.py` |
| `02_stage1_retrieval/` | Stage 1 | `selector_training/stage4_retraining__*.py` (CE training under both label sets), `server_score.py` (scoring the sealed pools), `evaluate.py` (registered retrieval metrics) |
| `03_stage2_pipeline/` | Stage 2 | `run_v2.sh` → `pipeline.py` (scheduler); `train_bge.py`, `select_evidence.py`, `generate.py`, `judge.py`, `score_v2.py`, `summarize.py`; `data_prep/build_external_set.py`; settings in `config/` |
| `04_stage3_supplementary/` | Stage 3 | `run_all.sh` → `s4_orchestrate.py` (Qwen3-Reranker label flip: `s4_prepare_data.py`, `s4_train.py`, `s4_score.py`, `s4_select.py`, `s4_generate.py`); `task_rejudge.py` (Qwen3-32B judge); `task_official.py` and `s4_mcq.py` (expert protocol; needs the TCM-BEST4SDT evaluation code) |
| `05_analysis/` | Paper numbers | `stage2/retrieval.py`, `answers.py`, `mechanism.py`, `registered.py`, `scr.py`, `extra.py`, `scale.py`, `metric_check.py`; `stage3/s4_retrieval.py`, `s3_extra.py`, `cov_vs_kpr.py` |

## Path map for the analysis scripts

The analysis scripts use the paths of the original working copy. Point them at this release as follows.

| Path in the scripts | In this release |
|---|---|
| `expert_testset_v1/_sealed_负责人保存_勿发给标注员/final_questions.json` | `data/sealed/questions.jsonl` (points); question text is withheld |
| `…/sealed_pools.jsonl` | `data/sealed/candidate_pools.jsonl` |
| `…/phase2_gold_labels.jsonl` | withheld (see the main README) |
| `eval_v2/server/data/oracle_lists.jsonl` | withheld |
| results pack `runs/scores`, `runs/selections`, `runs/gen`, `runs/judge` | `results/stage2/scores`, `selections`, `answers`, `judgements` |
| results pack `runs/external/*` | `data/external/*` |
| stage 3 `results/s4/*`, `results/rejudge/*`, `results/official/*` | `results/stage3/*` |
