# Protocol

The documents that fixed the criteria before the analyses that use them.

| File | Original name | What it fixes |
|---|---|---|
| `analysis_freeze_v2.5.md` | 分析冻结_v2.5.md | the increment of 2026-10-31: which earlier clauses are withdrawn, the calibration reporting obligation, the join-key requirement, and why a criterion may not be rewritten after the fact |
| `physician_protocol_v1.0.md` | 医生标注协议_v1.0_20261103.md | the blind annotation protocol: what the physician sees, in what order, and the decision rule for each case. Section 5 is what `04_gates/analyze_physician.py` implements |
| `preprocessing_spec_v2.0.md` | 统一处理规范_v2.0.md | the single preprocessing specification shared by the four sources |
| `mediator_layer_v2.4.md` | 物理量中介层_v2.4.md | the mediator layer, including the five dimensions it declares unmeasurable |
| `applied_cohort_annotation_v1.0.md` | 真实队列标注规程_v1.0.md | the applied cohort's quality gate and annotation procedure. Two example subject codes are replaced by `SUBJ-A` and `SUBJ-B` |

The freeze documents are incremental by design: each version states what it withdraws
from the one before, and the text of a withdrawn clause is kept with the reason rather
than deleted. That is the point of shipping them. A criterion that was changed after the
measurement it judges is visible as a dated entry, not as silence.

## The thresholds

Which threshold each gate used, how it was classified, and the date it was fixed are in
`../tables/threshold_timing.csv`. The `protocol_file` column in that table reproduces the
file name as it stood in the working tree at freeze time; four of those names are `.yaml`
files that are not in this repository. The criteria themselves are restated in the
docstrings of the gate scripts, which is where the reader can check them against the code
that applies them.
