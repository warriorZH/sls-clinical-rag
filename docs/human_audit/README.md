# Human audit of the LLM judges

**Purpose:** to calibrate the key-point recall (KPR) judgements of the LLM judges against human raters with TCM training.

| Item | Value |
|---|---|
| Raters | {{AUDIT_RATERS}} |
| Sample | {{AUDIT_SAMPLE}} |
| Blinding | {{AUDIT_BLINDING}} |
| Rubric | Same four-level rubric as the judges (2 stated, 1 partial, 0 not stated, X contradicted), with the reference points shown and the condition hidden |
| Results | {{AUDIT_RESULTS}} |

The registered plan (stage 2 protocol, section 4.3) specified two raters without TCM background. Raters with TCM training were used instead; this deviation is listed in the paper's Supplementary Material S2.

Per-judgement human labels: `human_audit_labels.csv` ({{AUDIT_FILE_STATUS}}).
