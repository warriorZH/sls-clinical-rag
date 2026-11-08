# Relevant is not enough: support-label supervision for evidence selection in clinical RAG

Data, code, protocols and outputs for the paper

> Zhang P, Hu J, Li X, Hao B, Abu Bakar J. Relevant is not enough: Support-label supervision for evidence selection in retrieval-augmented clinical question answering. Manuscript submitted to *Artificial Intelligence in Medicine*, 2026.

**Status:** v1.0, accompanying the manuscript submitted to *Artificial Intelligence in Medicine* · **Registrations (OSF):** [Stage 1](https://osf.io/gj48z) · [Stage 2](https://osf.io/e37na) · [Stage 3](https://osf.io/e3gkh)

This repository supports a pre-registered label-flip study. Evidence selectors were trained on the same 26,432 question–passage pairs under two label definitions, *relevance* and *support*. Only the labels of relevant-but-non-supporting pairs differ. Four selectors (mDeBERTa cross-encoder, bge-reranker-v2-m3, Qwen3-Reranker-0.6B and Qwen3-Reranker-4B) were compared on a 176-question sealed test set and on 300 cases of the expert-annotated TCM-BEST4SDT benchmark, at the retrieval level, the answer level and under the benchmark's expert multiple-choice protocol.

> **Not for clinical use.** The data and models are for research on evidence selection only. Generated answers are model outputs and may be wrong.

## Contents

| Folder | What it holds |
|---|---|
| `data/sealed/` | Sealed test set: question IDs, type, stratum, source IDs and **required answer points**; candidate pools (passage IDs, offsets, SHA-256 and BM25 scores); selection and pool manifests |
| `data/external/` | The 300 TCM-BEST4SDT cases as used here: questions, required points, reference answers and candidate pools (derived from TCM-BEST4SDT, CC BY 4.0) |
| `data/training/` | Training, development and test pairs: IDs, groups, relevance label, support label and covered points |
| `results/stage1_retrieval/` | Stage 1 (registered first): selector scores on the sealed pools |
| `results/stage2/` | Stage 2 (main registration): selector scores, selected evidence, generated answers, judge labels (KPR and claim support), bge label-flip training logs, run configuration and summary |
| `results/stage3/` | Stage 3 (supplementary registration): Qwen3-Reranker label flip, Qwen3-32B re-judging and the expert multiple-choice protocol |
| `results/analysis/` | Aggregate outputs of the analysis scripts (all numbers reported in the paper) |
| `results/annotation_agreement/` | Agreement reports for the sealed-set annotation |
| `code/` | Sealed-set construction, selector training, generation and judging pipelines and analysis scripts, ordered by stage |
| `docs/registration/` | The three registrations and the post-registration change log, as registered (Chinese) |
| `docs/annotation/` | Annotation guidelines and adjudication notes (Chinese) |
| `docs/label_review/` | Clinician review of a 25% sample of the sealed-set labels: protocol and aggregate results |
| `DATASHEET.md` | Provenance, composition, labelling process and limitations of the data |
| `SHA256SUMS.txt` | Checksums of every file in this release |

Field descriptions are in `data/README.md`.

## How the labels were produced (please read before reuse)

- **Sealed set: required points and support labels.** Required points were drafted by rules from each record's own analysis. Three LLM annotation agents then verified the points and pre-annotated every candidate–point pair, and disagreements were adjudicated against the annotation guidelines. A TCM expert (Bin Hao) reviewed, item by item, all 161 candidates on which the annotators disagreed about full support, the label behind the primary retrieval metric, and confirmed the adjudicated labels. The remaining labels were not re-annotated by experts. Agreement among the three annotators on full support was low (Krippendorff's α = 0.220); see `results/annotation_agreement/`.
- **External set: reference answers.** TCM experts wrote and reviewed these for TCM-BEST4SDT (Li et al.). We converted them into required points by rule; see `docs/external_set_construction_zh.md`.
- **Training pairs.** Relevance and support labels come from the project's earlier evidence-construction pipeline. A support label of 1 always implies relevance 1 (`data/README.md`).
- **Clinician review.** A clinician co-author reviewed a stratified 25% sample (44 questions, 880 candidates) in a confirmatory format. 80.6% of the reviewed points were judged correct, and support labels agreed 94.8% (κ = 0.79). The labels never over-called full support but missed about half of it, so the coverage metrics are conservative; see `docs/label_review/`.
- **Answer judgements.** Judge A (GLM-4-9B-0414), judge B (Hunyuan-7B-Instruct) and Qwen3-32B are open-weight models. The judges were not calibrated against human raters, because the registered audit of judged answers was not carried out. Literal recall is reported as a judge-free check.

## What is withheld and why

| Item | Reason | Access |
|---|---|---|
| Text of sealed-set questions, candidate passages and training pairs | Redistribution rights of the source corpora have not been confirmed | Rebuild from the source collections using the IDs, offsets and checksums provided |
| Passage-level support and relevance labels of the sealed set, and everything derived from them (oracle evidence lists, oracle-condition selections, answers and judgements) | Keeps the sealed set usable as a held-out benchmark | On reasonable request to the corresponding author |
| Fine-tuned selector weights | Size | On reasonable request to the corresponding author |
| TCM-BEST4SDT evaluation code (used by `code/04_stage3_supplementary/task_official.py`) | Third-party code (Apache-2.0) | Obtain from the benchmark's repository |

All passages are identified by `document_id`, character offsets and SHA-256 hashes. Anyone with the source corpora can therefore rebuild the texts and verify them.

## Reproducing the results

The analysis scripts in `code/05_analysis/` read the files in `results/` and write the JSON files in `results/analysis/`. `retrieval.py`, `mechanism.py`, `registered.py`, `extra.py`, `scale.py` (stage 2) and `s4_retrieval.py` (stage 3) also read the withheld support labels, so retrieval-level coverage and the point-level mechanism analysis cannot be recomputed from the public files alone. The scripts are released exactly as run, with their original input paths; `code/README.md` maps those paths to this release. Answer-level results (key-point recall, literal recall and the expert protocol) need only the public files.

| Stage | Entry point | Hardware used |
|---|---|---|
| Sealed-set construction | `code/01_sealed_set_construction/` (Python ≥ 3.8; `build_sealed_testset.py` needs scikit-learn) | CPU |
| Stage 1: CE training and retrieval scoring | `code/02_stage1_retrieval/` | 2 × NVIDIA P40 |
| Stage 2: bge label flip, generation, judging | `code/03_stage2_pipeline/run_v2.sh` (config in `config/pipeline.json`) | 2 × NVIDIA P40 |
| Stage 3: Qwen3-Reranker label flip, 32B judge, expert protocol | `code/04_stage3_supplementary/run_all.sh` (vLLM; `requirements_cloud.txt`) | rented cloud GPUs (AutoDL, 2 × RTX 6000D 84 GB) |
| Analysis | `code/05_analysis/` | CPU |

Model repositories and revisions are pinned in `results/stage2/model_registry.json` and `results/stage3/models_registry.json`. Random seeds are 42, 2026 and 3407 for training, and 20261026 for the bootstrap. Frozen input manifests (`FREEZE_MANIFEST*.json`) list the SHA-256 of every input at registration time.

Server addresses, user names and local paths were replaced by placeholders (`<SERVER>`, `<user>`). Nothing else in the code or registered documents was changed.

## Licence

- Code: MIT (see `LICENSE`).
- Our annotations, required points, labels and outputs: CC BY 4.0 (see `DATA_LICENSE.md`).
- Files in `data/external/` are derived from TCM-BEST4SDT (CC BY 4.0). Please also cite Li et al. (DOI 10.6084/m9.figshare.30615956).

## Citation

See `CITATION.cff`.

## Contact

Peihua Zhang (maintainer, zhangpeihua@hbfu.edu.cn); corresponding author: Juhaida Abu Bakar (juhaida.ab@uum.edu.my)
