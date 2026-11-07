# Datasheet

This datasheet follows the structure of *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation

The data were created to test whether evidence selectors for retrieval-augmented clinical question answering should be trained on *support* labels (the passage states a required conclusion) rather than *relevance* labels, with everything else held constant. {{FUNDING_SHORT}}

## Composition

- **Sealed test set:** 176 syndrome-differentiation questions (99 diagnosis, 77 treatment principle) with 547 required points, and 20 BM25 candidates per question.
- **External set:** 300 cases from TCM-BEST4SDT with points converted from the expert reference answers.
- **Training pairs:** 26,432 question–passage pairs (class-balanced), with development (4,978) and test (5,030) splits.
- **Outputs:** selector scores, selected evidence, generated answers and judge labels for every registered condition.

## Collection and sources

- **Case records and retrieval corpus:** {{SOURCE_CORPORA}}. The corpus has 1.82 million passages from textbooks, classical texts and case records. The case records are publicly available and de-identified; no new patient data were collected.
- **TCM-BEST4SDT:** Li et al., Figshare DOI 10.6084/m9.figshare.30615956, CC BY 4.0.

## Labelling

- **Sealed set:** points were drafted by rules and verified by three LLM annotation agents ({{ANNOTATION_AGENTS}}), and disagreements were adjudicated. Every candidate–point pair received a support level (2 full, 1 partial, 0 none) and every candidate a relevance label. These are LLM-agent labels. Agreement: α = 0.220 for full support (AC1 = 0.982), 0.487 for the three-level scale and 0.363 for relevance.
- **Answer judging:** done by open-weight LLM judges. {{HUMAN_AUDIT_SUMMARY}}

## Uses

**Intended:** research on evidence selection, retrieval supervision and the evaluation of evidence-grounded generation.

**Not intended:** clinical decision-making, patient care, or training systems presented as medical advice.

## Known limitations

- One language and one medical tradition.
- LLM-agent labels on the sealed set.
- The relevance labels of the training data are a weak relevance signal.
- The label flip also changes the share of positives (50.0% vs. 78.6%).
- Key-point recall does not penalise added errors.
- The sealed set was evaluated once before the main registration (stage 1).

## Distribution and maintenance

- Distributed via {{REPO_URL}}, with an archived copy at {{ZENODO_DOI}}.
- Maintainer: {{CONTACT}}.
- Errors will be corrected in tagged releases; the paper refers to release {{RELEASE_TAG}}.
