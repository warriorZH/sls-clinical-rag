# 相关还不够：面向检索增强临床问答证据选择的支持标签监督

本仓库包含下面这篇论文的数据、代码、登记文件和实验输出：

> Zhang P, Hu J, Li X, Hao B, Abu Bakar J. Relevant is not enough: Support-label supervision for evidence selection in retrieval-augmented clinical question answering. Manuscript submitted to *Artificial Intelligence in Medicine*, 2026.

**状态：** v1.1，配合投稿至 *Artificial Intelligence in Medicine* 的稿件 · **预注册（OSF）：** [第 1 阶段](https://osf.io/gj48z) · [第 2 阶段](https://osf.io/e37na) · [第 3 阶段](https://osf.io/e3gkh)

本研究是一项预注册的标签翻转研究：

- **训练：** 在同一批 26,432 个“问题–段落”对上，分别用“相关”和“支持”两种标签训练证据选择器。两套标签只在“相关但不支持”的段落上不同。
- **选择器：** mDeBERTa 交叉编码器、bge-reranker-v2-m3、Qwen3-Reranker-0.6B 与 4B，共四个。
- **测试集：** 176 题封存集，以及专家标注的 TCM-BEST4SDT 中的 300 个病例。
- **比较层次：** 检索层、答案层，以及该基准的专家选择题方案。

> **不得用于临床。** 本资源只用于证据选择方法的研究。生成的答案是模型输出，可能有错。

## 目录

| 文件夹 | 内容 |
|---|---|
| `data/sealed/` | 封存集：题号、题型、分层、原始病例编号、**必需要点**；候选池（段落编号、字符位置、SHA-256、BM25 分数）；选题与候选池清单 |
| `data/external/` | 本研究使用的 TCM-BEST4SDT 300 例：问题、必需要点、参考答案、候选池（派生自 TCM-BEST4SDT，CC BY 4.0） |
| `data/training/` | 训练、开发、测试对：编号、分组、相关标签、支持标签、覆盖的要点 |
| `results/stage1_retrieval/` | 第一阶段：封存候选池上的选择器打分 |
| `results/stage2/` | 第二阶段（主登记）：打分、选中证据、生成答案、评审标签、bge 标签翻转训练记录、运行配置与汇总 |
| `results/stage3/` | 第三阶段（补充登记）：Qwen3-Reranker 标签翻转、Qwen3-32B 复评、专家选择题方案 |
| `results/analysis/` | 分析脚本的汇总输出（论文中的全部数字） |
| `results/annotation_agreement/` | 封存集标注一致性报告 |
| `code/` | 按阶段排列的全部代码 |
| `docs/registration/` | 三次 OSF 登记文件的逐字节副本（中文原文）、第 2 阶段修订 v2.3（未提交到 OSF）、登记后变更记录，以及 `docs/registration/en/` 中的英文译本；说明见 `docs/registration/README.md` |
| `docs/annotation/` | 标注规范与仲裁说明 |
| `docs/label_review/` | 临床医师对封存集 25% 标签的抽检复核：方案与汇总结果 |
| `DATASHEET.md` | 数据来源、构成、标注过程与局限 |
| `SHA256SUMS.txt` | 全部文件的校验值 |

## 标签来源（复用前请阅读）

- **封存集要点与支持标签：** 要点先由规则程序草拟，再由三个 LLM 标注代理核验并预标注，分歧再由 LLM 代理按标注规范仲裁。题目、候选池和标签随后连同校验值一起冻结，冻结后才对任何选择器打分。在第 1、2 阶段登记之后，标注者在“完整支持”上存在分歧的全部 161 个候选（主要检索指标依据的正是这一标签），由中医专家郝斌逐条复核，确认了仲裁结果，未作改动；其余标签未经专家重新标注。三位标注者在“完整支持”上的一致性较低（α = 0.220）。
- **外部集参考答案：** 由 TCM-BEST4SDT 的中医专家撰写和审定，本研究按规则转为要点。
- **训练对标签：** 来自本项目早先的证据构建流程。支持必然相关。
- **临床医师复核：** 一位临床医师合著者以确认式复核了 25% 的分层样本（44 题、880 条候选）。已复核要点的正确率为 80.6%，支持度一致率为 94.8%（κ = 0.79）。标签从未把“完整支持”判高，但漏判约一半，因此覆盖率指标偏保守；详见 `docs/label_review/`。
- **答案评审：** 由开源模型完成。评审未经人工校准（登记时计划的答案抽查没有进行）；另报告不依赖评审的逐字召回率。

## 未公开的内容及原因

| 内容 | 原因 | 获取方式 |
|---|---|---|
| 封存题目、候选段落、训练对的原文 | 原始语料的再分发许可尚未确认 | 可根据所给编号、字符位置和校验值从原始语料还原 |
| 封存集的段落级支持与相关标签，以及由其派生的内容（oracle 证据列表、oracle 条件的证据、答案和评审） | 保持封存集可继续作为留出测试集 | 可向通讯作者合理申请 |
| 微调后的选择器权重 | 体积大 | 可向通讯作者合理申请 |
| TCM-BEST4SDT 评测代码 | 第三方代码（Apache-2.0） | 请从该基准的仓库获取 |

每个段落都给出了编号、字符位置和 SHA-256。拥有原始语料的研究者可以据此还原段落原文，并逐一核对。

## 复现

- **不需要支持标签的结果：** 答案层结果（要点召回、逐字召回、专家方案）只需公开文件即可复现。
- **需要支持标签的结果：** 检索层覆盖率和要点层机制分析还需要读取未公开的支持标签。
- **脚本路径：** 脚本按原样发布，路径对照见 `code/README.md`。
- **模型版本与随机种子：** 模型版本固定在 `model_registry.json` 中；训练种子为 42、2026、3407，bootstrap 种子为 20261026。

服务器地址、用户名、主机名和本地路径已替换为占位符，其余内容均未改动。`docs/registration/osf/` 中是 OSF 登记文件的原样副本，没有任何改动。外部集的第 2 阶段运行依据修订 v2.3：该修订在运行前已冻结哈希，但没有提交到 OSF（见 `docs/registration/README.md`）。

## 许可

- **代码：** MIT。
- **本研究产生的标注、标签与输出：** CC BY 4.0。
- **`data/external/`：** 派生自 TCM-BEST4SDT（CC BY 4.0），使用时请同时引用 Li 等人的工作。

## 联系方式

Peihua Zhang (maintainer, zhangpeihua@hbfu.edu.cn); corresponding author: Juhaida Abu Bakar (juhaida.ab@uum.edu.my)
