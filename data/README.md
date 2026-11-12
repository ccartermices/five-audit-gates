# Data

Derived tables only. No waveforms, no subject-level records, no identifiers.

Every file here is the output of one script in `scripts/`, and every figure in `figures/`
is drawn from these files. Column names are the names the analyses wrote; the encoding is
UTF-8 with a byte-order mark, because the audit log's CSV writer has always added one, so
read them with `encoding="utf-8-sig"` or let Pandas sniff it.

## By gate

| Gate | Files | Produced by |
|---|---|---|
| G1 | `phys_mediator_v1.json` (the frozen layer definition, hash `f557d04dc5e431a9`) | `01_features/phys_layer.py` |
| G2 | `audit_physician_preview.csv`, `audit_physician_scene_summary.csv` | `04_gates/audit_physician_preview.py` |
| G2 | `audit_physician_single_full.csv`, `audit_physician_final.csv` | `04_gates/analyze_physician.py` |
| G2 | `e_label_validity.csv`, `e_label_single_auc.csv` | `04_gates/e_label_validity.py` |
| G2 | `vdb_ep2_validity.csv`, `vdb_ep2_age_strata.csv` | `04_gates/ep_vitaldb.py` |
| G2 | `audit_external_validation.csv` | `04_gates/audit_external_validation.py` |
| G2 | `fusion_T1_sbp.csv` | `02_models/fusion.py` |
| G3 | `audit_vdb_icc.csv` | `04_gates/audit_vdb_icc.py` |
| G3 | `ep9_reliability.csv` | `04_gates/ep9_reliability.py` |
| G3 | `audit_cm_person_level.csv` | `04_gates/audit_cm_person_level.py` |
| G3 | `l1_rho_pairwise.csv` | `04_gates/audit_missing.py` |
| G3, G5 | `audit_threshold_sensitivity.csv` | `04_gates/audit_threshold_sensitivity.py` |
| G4 | `audit_null_family.csv`, `audit_null_share.csv` | `04_gates/audit_null_family.py` |
| G4 | `audit_weighting.csv` | `04_gates/audit_weighting.py` |
| G4 | `audit_cluster_alias.csv` | `04_gates/audit_cluster_alias.py` |
| G4 | `audit_v22_zcal.csv`, `audit_v22_power.csv`, `audit_v22_estimability.csv` | `04_gates/audit_v22_zcal.py`, `audit_v22_power.py`, `audit_v22_estimability.py` |
| G4, G5 | `audit_v23_split.csv` | `04_gates/audit_v23_split.py` |
| G5 | `probe_what.csv`, `probe_what_meta.csv`, `probe_what_seed_stability.csv`, `probe_what_fam_by_seed.csv`, `probe_what_k_scan.csv`, `probe_what_top12_jaccard.csv`, `probe_what_top12_family.csv` | `03_attribution/probe_what.py`, `04_gates/audit_probe_what_seeds.py` |
| G4, G5 | `probe_mediator.csv`, `probe_mediator_l1.csv`, `probe_mediator_meta.csv`, `probe_mediator_pooled.csv`, `probe_mediator_pooled_meta.csv`, `probe_mediator.npz` | `03_attribution/probe_mediator.py` |
| discussion | `xinterp_*.csv` | `03_attribution/x_interp_v2.py` |
| background | `m1_*.csv`, `m2_*.csv`, `tiny_*.csv`, `e1_pwdb_readout.csv`, `audit_clinical_metrics.csv` | the scripts of the same name |

`probe_mediator.npz` holds sparse-autoencoder activations and the mediator matrix per
source and architecture. It is an array of model states, not of measurements: rows carry
no ordering that identifies a subject, and the file has no identifiers of any kind.

## What is deliberately not here

| Missing | Why |
|---|---|
| `pwdb_cycles.npz`, `mimicbp_cycles.npz`, `vitaldb_cycles.npz`, `multibeat_cyc.npz` | waveforms; large, and two of the three sources require registration or a data use agreement |
| `real_cycles*.npz` | the applied cohort; its ethics approval does not permit sharing |
| `real_features.csv`, `labels_master.csv`, `ai_prelabel_cm.csv`, `subject_table.csv`, `cm_quality*.csv` | subject-level rows of the applied cohort |
| `ablation_p0.csv`, `physician_annotation_form.csv` | both carry applied-cohort subject codes |
| `corpus_v2_record.csv`, `corpus_v2_subject.csv` | per-record rows across four sources, including the applied cohort |
| `pwdb_features.csv`, `pwdb_truth.csv` | 18 MB, and regenerable from the public PWDB archive with `00_prepare/extract_pwdb.py` |

A script that needs one of these will stop at the file open, with the path it wanted. That
is the intended behaviour: the failure names the missing input rather than silently
substituting something else.
