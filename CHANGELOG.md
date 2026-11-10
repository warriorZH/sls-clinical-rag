# Changelog

## v1.1 (2026-11-10)

- `docs/registration/`:
  - now holds byte-identical copies of all files in the three OSF registrations;
  - adds the Stage 1 implementation protocol (`实验对比方案.md`), which was missing in v1.0;
  - adds English translations (`en/`) and an index (`README.md`).
- Corrects a labelling error in v1.0. The Stage 2 protocol file in v1.0 was version v2.3, presented as "registered". The OSF Stage 2 registration contains v2.2.
  - v2.3 is now kept separately as `stage2_amendment_v2.3_protocol_zh.md`.
  - It added the external validation; it was hash-frozen before the runs but not deposited on OSF.
- README, README_zh and DATASHEET:
  - state that disagreements were adjudicated by an LLM agent;
  - state that the labels were frozen before any selector was scored;
  - state that the TCM expert's confirmation took place after the Stage 1 and Stage 2 registrations.
- Adds `.gitattributes` so that Git stores every file byte for byte and `SHA256SUMS.txt` stays valid on Windows.
- Data, code and results are unchanged.

## v1.0 (2026-11-07)

- First public release.
