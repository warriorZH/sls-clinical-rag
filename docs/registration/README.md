# Registrations and protocol documents

The analyses were registered on the Open Science Framework (OSF) in three stages. The Chinese documents are the authoritative versions. English translations, prepared on 2026-11-10 after all runs, are in `en/`.

## What is where

| Stage | OSF registration | Registered (UTC) | Byte-identical copies of the OSF files | English translation |
|---|---|---|---|---|
| 1 | https://osf.io/gj48z | 2026-10-26 09:17 | `osf/stage1_gj48z/`: `预注册分析计划.md` (analysis plan), `实验对比方案.md` (implementation protocol), `FREEZE_MANIFEST.json` | `en/stage1_preregistration_plan_en.md`, `en/stage1_implementation_protocol_en.md` |
| 2 | https://osf.io/e37na | 2026-10-27 14:46 | `osf/stage2_e37na/`: `实验设计方案_v2.md` (protocol v2.2), `FREEZE_MANIFEST_v2.json` | `en/stage2_protocol_v2.2_registered_en.md` |
| 2, amendment | **Not on OSF** | — | `stage2_amendment_v2.3_protocol_zh.md` (protocol v2.3; GPU host name replaced by a placeholder) | `en/stage2_amendment_v2.3_external_validation_en.md` |
| 3 | https://osf.io/e3gkh | 2026-11-04 17:01 | `osf/stage3_e3gkh/`: `补充登记_v2.4.md` (supplementary registration), `FREEZE_MANIFEST_sup.json` | `en/stage3_supplementary_registration_v2.4_en.md` |

`stage2_post_registration_changes_zh.md` is the change log of the infrastructure changes made after the Stage 2 runs had started. It was not deposited on OSF.

## SHA-256 of the registered files

These values match the hashes that OSF reports for the registered files.

| File | SHA-256 |
|---|---|
| `osf/stage1_gj48z/预注册分析计划.md` | `68b9e7353295cf9107a5ec07f89d257fcd0f8bcde62a3b35150c87c9d2b2c47c` |
| `osf/stage1_gj48z/实验对比方案.md` | `1b8257ff78b70d34aaf42e81dec2a650e43939cddc7422a2c360315a342c5775` |
| `osf/stage1_gj48z/FREEZE_MANIFEST.json` | `fc4c80ebe2c4db3e70ee46ca8490f4221f8938426b7bd93f146b2192a80fde5f` |
| `osf/stage2_e37na/实验设计方案_v2.md` | `93d66880961665a2611184948d9ea813bc18b21112cfad1d3093bf0483113119` |
| `osf/stage2_e37na/FREEZE_MANIFEST_v2.json` | `2fb3b7dd0789bc0313dfa2bcdb3f69bd48efc225ad837d71b14f89f3c4817692` |
| `osf/stage3_e3gkh/补充登记_v2.4.md` | `e289464d96c0fd618a26cd41cb958b77bf8e82ef8cbf257eee4fe5fe42f89fbc` |
| `osf/stage3_e3gkh/FREEZE_MANIFEST_sup.json` | `ec11a0120e0e0b9c0d6646ddc09e3a420c8f0881a2d2bae110c6ef6cac4256ba` |

Because these are exact copies, they keep one server user name and one GPU host name that are replaced by placeholders elsewhere in this release.

## The Stage 2 amendment (v2.3)

The external validation on TCM-BEST4SDT was added to the Stage 2 protocol after the Stage 2 registration, as version v2.3.

- **What it changed.** The amendment added the external-set analyses (HX3, HX4, HX7) and four fixes to the launcher and scheduler (its Section 16.8). It did not change any sealed-set hypothesis, condition, metric, prompt, judging rule or statistical method; `en/stage2_amendment_v2.3_external_validation_en.md` lists every difference from v2.2.
- **When it was frozen.** The v2.3 protocol, the external-set data files and the code were hash-frozen on 2026-10-28 at 07:50:45 (UTC+8). The frozen manifest is `code/03_stage2_pipeline/FREEZE_MANIFEST_v2.json` (placeholders as elsewhere); in its original form it lists the v2.3 protocol with SHA-256 `2fdb2c1ee08ac1f194426961b493ae250665ba0ad5c30e6ea123e5cd821617da`. All Stage 2 runs, including every external-set run, started after this freeze.
- **What was not done.** The amendment was not deposited on OSF. The Stage 2 external-set analyses are therefore pre-specified, not publicly registered, as stated in the manuscript (Section 2.7) and its Supplementary Material (S2).
- **Earlier launch.** An earlier launch under v2.2 on 2026-10-27 performed no scoring or generation, because the server's GPUs were unavailable.

## Note on the annotators

The Stage 1 analysis plan foresaw annotation and adjudication of the sealed set by TCM professionals. Both were in fact carried out by LLM agents. This was recorded as deviation D1 in the implementation protocol deposited with the Stage 1 registration, before any selector was scored.

After the Stage 1 and Stage 2 registrations, a TCM expert reviewed the 161 adjudicated disagreements on full support and confirmed them without changes.

---

# 登记文件与方案文件（中文说明）

- **`osf/`：** 三次 OSF 登记中全部文件的逐字节副本，与 OSF 上的校验值一致。中文原件为准。
- **`en/`：** 英文译本，2026-11-10 在全部运行结束后翻译，不增加任何登记内容。
- **第 2 阶段修订 v2.3（外部验证）：**
  - 文件为 `stage2_amendment_v2.3_protocol_zh.md`。
  - 已于 2026-10-28 07:50:45（北京时间）冻结哈希，早于外部集的任何运行，但**没有提交到 OSF**。
  - 因此，第 2 阶段的外部集分析属于事先设定、但未公开登记的分析。论文第 2.7 节和补充材料 S2 已如实说明。
- **`stage2_post_registration_changes_zh.md`：** 第 2 阶段运行开始后的环境变更记录，未提交到 OSF。
