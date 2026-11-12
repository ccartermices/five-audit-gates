# 物理量中介层规范 v2.4（2026-11-03）

**性质**：判据层/证据层版本。特征定义层（TABLE 7 簇 30 量）、GAPS、冻结哈希
`f557d04dc5e431a9` **全部不变**（v2.4 不改 `phys_layer.py`）。变更仅两处：
簇级定位判据正式升级（G43 修订）与标注证据链规则（新增 G44）。

---

## 1. 判据正式升级（G43 修订，自 20 号的实测教训）

**废止**：「簇级 z 随 SNR 单调上升」——在饱和区失效（20 号实测：z 平台化
微降，判据误报"不稳"）。

**v2.4 生效判据**（植入对照，逐格判定）：

> **检出** ⟺ SNR=0 格 |z| < THR_Z（2.6） **且** 任一非零 SNR 格 det ≥ 60%
> （det = 20 次重复中 |z| ≥ THR_Z 的比例）

`audit_v23_split.py` 的历史产出按新判据复核结论不变（dicrotic：SNR=0 格
z=+0.06/−0.51 均不检出，非零格 det=100% → 检出；stiffness-PWDB：非零格
det=0% → 不检出，阴性对照地位维持）。

## 2. G44：标注证据链与措辞规则

背景：21/22 号确立「AI 预标注 + 医师确认」路线。为防审稿与自引时的
口径滑坡，三级命名自此固定：

| 级别 | 命名 | 数据要求 | 允许的措辞 |
|---|---|---|---|
| L1 | physician-endorsed protocol | 医师认可方案（含口头） | "标注方案经医师认可" |
| L2 | **physician-confirmed** | 医师逐例确认（含口头转述，须记录渠道） | "与医师确认的判读一致"；**不得**写 "validated/verified" |
| L3 | independent blinded annotation | ≥2 医师书面独立盲判 + κ 报告 | 任意强措辞（"independent expert-validated"） |

**当前 CM 标签状态**：L2（`physician_annotation_confirmed.csv`，口头逐例
确认，经所有者转述）。L3 升级路径保留（盲判表 `physician_annotation_form.csv`
空置待填）。

**构造性边界规则**（L2 特有，防止自证循环进论文）：
- 标签由 L1 量生成时，**规则内量**（生成标签所用的量及其强相关量，
  |ρ|≥0.7）的判别 AUC 不得报为发现，仅可在附表完整列出；
- 规则外量的通过可作为**弱佐证**（"consistent beyond the defining
  quantities"），不可叠加成强结论；
- 置换 p 必须用 **max 统计**（对全部单量的最大 AUC 置换），多重比较内建，
  禁止挑单量后补算 p。

## 3. 22 号主结果存档（判据 v2.4 口径）

| 口径 | n（弦/非弦） | 过 AUC95 门槛 | 规则外通过者 | 嵌套 LOOCV | 置换 p(max) |
|---|---|---|---|---|---|
| 高置信 | 16（7/9） | 10/30 | t_dicrotic、t_notch | 0.952 | 0.004 |
| 全集（边界计非弦） | 56（7/49） | 8/30 | **h4、h4_r** | 0.781 | 0.013 |

门槛（预注册）：高置信 0.746/0.841、全集 0.694/0.749。

## 4. 与 v2.3 的差异清单

| 项 | v2.3 | v2.4 |
|---|---|---|
| TABLE / GAPS / 哈希 | `f557d04dc5e431a9` | **不变** |
| 簇级定位判据 | z 单调（实测失效） | SNR=0 不检出 且 det≥60%（G43 修订） |
| 标注证据链 | 无 | G44 三级命名 + 构造性边界规则 |
| 标签置换门槛表 | — | 预注册（n=16: 0.746/0.841；n=56: 0.694/0.749） |
| THR_Z = 2.6 | ✓ | ✓（不变） |

## 5. 文件

| 文件 | 内容 |
|---|---|
| `data/physician_annotation_confirmed.csv` | L2 标签（含渠道 provenance） |
| `scripts/audit_physician_final.py` → `data/audit_physician_final.csv` | 22 号正式分析 |
| `scripts/audit_physician_preview.py` → `data/audit_physician_preview.csv` | 门槛表（预注册） |
| `22_医师确认判别分析_20261103.md` | 结果与构造性标注 |
