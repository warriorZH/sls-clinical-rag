# Datasheet

This datasheet follows the structure of *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation

The data were created to test whether evidence selectors for retrieval-augmented clinical question answering should be trained on *support* labels (the passage states a required conclusion) rather than *relevance* labels, with everything else held constant. The work was supported by the Youth Science and Technology Research Fund of Hebei University Affiliated Hospital (No. 2024QA05).

## Composition

- **Sealed test set:** 176 syndrome-differentiation questions (99 diagnosis, 77 treatment principle) with 547 required points, and 20 BM25 candidates per question.
- **External set:** 300 cases from TCM-BEST4SDT with points converted from the expert reference answers.
- **Training pairs:** 26,432 question–passage pairs (class-balanced), with development (4,978) and test (5,030) splits.
- **Outputs:** selector scores, selected evidence, generated answers and judge labels for every registered condition.

## Collection and sources

- **Case records and retrieval corpus:** a de-identified TCM case-record collection and a corpus of TCM textbooks and classical texts, assembled in an earlier stage of the project; their individual sources and licences will be documented in a later release. The corpus has 1.82 million passages from textbooks, classical texts and case records. The case records are publicly available and de-identified; no new patient data were collected.
- **TCM-BEST4SDT:** Li et al., Figshare DOI 10.6084/m9.figshare.30615956, CC BY 4.0.

## Labelling

- **Sealed set:** points were drafted by rules and verified by three LLM annotation agents (three independent GPT-6 LUNA sessions), which pre-annotated every candidate–point pair with a support level (2 full, 1 partial, 0 none) and every candidate with a relevance label; disagreements were adjudicated against the annotation guidelines. A TCM expert (Bin Hao) reviewed and confirmed all 161 adjudicated full-support disagreements; a clinician reviewed a stratified 25% sample (`docs/label_review/`). Agreement: α = 0.220 for full support (AC1 = 0.982), 0.487 for the three-level scale and 0.363 for relevance.
- **Answer judging:** done by open-weight LLM judges. The judges were not calibrated against human raters, because the registered audit of judged answers was not carried out. Literal recall is reported as a judge-free check.

## Uses

**Intended:** research on evidence selection, retrieval supervision and the evaluation of evidence-grounded generation.

**Not intended:** clinical decision-making, patient care, or training systems presented as medical advice.

## Known limitations

- One language and one medical tradition.
- Sealed-set labels were pre-annotated by LLMs; experts confirmed the full-support disagreements and reviewed a 25% sample, but did not re-annotate the full set.
- The relevance labels of the training data are a weak relevance signal.
- The label flip also changes the share of positives (50.0% vs. 78.6%).
- Key-point recall does not penalise added errors.
- The sealed set was evaluated once before the main registration (stage 1).

## Distribution and maintenance

- Distributed via https://github.com/warriorZH/sls-clinical-rag.
- Maintainer: Peihua Zhang (maintainer, zhangpeihua@hbfu.edu.cn); corresponding author: Juhaida Abu Bakar (juhaida.ab@uum.edu.my).
- Errors will be corrected in tagged releases; the paper refers to release v1.0.
