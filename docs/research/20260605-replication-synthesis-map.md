---
created: 20260605
status: complete
title: Replication Synthesis Map - Mission Text Binary Classifier
reports:
  - docs/research/20260605-replication-weak-supervision-noisy-labels.md
  - docs/research/20260605-replication-imbalanced-validation.md
  - docs/research/20260605-replication-calibration-prevalence.md
  - docs/research/20260605-replication-short-text-model-alternatives.md
  - docs/research/20260605-replication-religious-nonprofit-mission.md
  - docs/research/20260605-replication-religious-vs-nonreligious-prompts.md
---

# Replication Synthesis Map: Religious Mission Text Classifier

## Completion Check

All six expected `replication-seeker` reports and scratchpads were written successfully. The prompt-focused agent returned an empty task message after the crash, but its report and scratchpad exist and contain the required sections.

| Workstream | Report | Scratchpad | Status |
|---|---|---|---|
| Weak supervision, LLM labels, noisy-label fine-tuning | `docs/research/20260605-replication-weak-supervision-noisy-labels.md` | `docs/research/notebooks/20260605-replication-weak-supervision-noisy-labels-scratchpad.md` | Complete |
| Imbalanced validation and metrics | `docs/research/20260605-replication-imbalanced-validation.md` | `docs/research/notebooks/20260605-replication-imbalanced-validation-scratchpad.md` | Complete |
| Calibration and prevalence estimation | `docs/research/20260605-replication-calibration-prevalence.md` | `docs/research/notebooks/20260605-replication-calibration-prevalence-scratchpad.md` | Complete |
| Short-text model alternatives | `docs/research/20260605-replication-short-text-model-alternatives.md` | `docs/research/notebooks/20260605-replication-short-text-model-alternatives-scratchpad.md` | Complete |
| Religious nonprofit mission classification | `docs/research/20260605-replication-religious-nonprofit-mission.md` | `docs/research/notebooks/20260605-replication-religious-nonprofit-mission-scratchpad.md` | Complete |
| Religious vs nonreligious prompt/codebook artifacts | `docs/research/20260605-replication-religious-vs-nonreligious-prompts.md` | `docs/research/notebooks/20260605-replication-religious-vs-nonreligious-prompts-scratchpad.md` | Complete |

## Highest-Value Reusable Artifacts

| Need | Best artifact(s) | Why useful |
|---|---|---|
| LLM annotation workflow | Gilardi Dataverse; Heseltine Dataverse; `npangakis/gpt_annotate` | Prompt logging, repeated annotation, human/LLM comparison, validation-first workflow. |
| Weak supervision architecture | Snorkel; Snorkel tutorials; WRENCH; BOXWRENCH | Labeling functions, abstentions, label models, weak-label benchmark format. |
| Noisy-label transformer training | `uds-lsv/BERT-LNL`; `gtfintechlab/SiDyP`; AlleNoise | BERT under realistic/injected noise and LLM-generated noisy-label calibration. |
| Imbalanced metrics | `precrec`; `hmeasure`; `micp`; Davis/Goadrich derivatives | PR/ROC, H-measure, balanced-accuracy intervals, threshold/cost references. |
| Validation framework | ValiText; Park & Montgomery Dataverse; Hopkins-King ReadMe | Construct/label/model/measure validation checklists and aggregate text measurement. |
| Calibration | `netcal`; `gpleiss/temperature_scaling`; Apple `ml-calibration`; `probmetrics` | Temperature/logistic/isotonic-style calibration, ECE, reliability diagrams. |
| Prevalence/quantification | QuaPy; QuantificationLib; LeQua 2022/2024; `freq-e`; `ppi_py` | CC/ACC/PCC/PACC, EM prior shift, prevalence CIs, prediction-powered inference. |
| Encoder model grid | Hugging Face BERT/RoBERTa/DeBERTa/DistilBERT/ELECTRA/ALBERT; Laurer Dataverse; Timoneda Dataverse | Reproducible model assets and social-science transformer comparison packages. |
| Nonprofit mission data | Ma `npo_classifier`; Santamarina Dataverse; NODC mission classifiers; UK-CAT | Mission/activity text, NTEE/purpose labels, IRS 1023-EZ mission corpora, charity taxonomy code. |
| Religious/nonreligious prompt artifacts | GivingTuesday `religious_org_v1`; GivingTuesday dataset; NCS/Pew/GSS/WVS/ARDA codebooks; Sider & Unruh typology | Closest public religious nonprofit classifier and authoritative wording/codebook domains. |

## Gaps and Caveats

| Gap | Practical implication |
|---|---|
| Several important papers have no reusable package | SaFER, LAFT, Törnberg, Lu & Smith, and some survey/codebook papers are conceptual unless reimplemented or manually inspected. |
| Some GitHub inspection was rate-limited | A few file manifests rely on landing pages or webfetch rather than full API-recursive inspection. |
| GivingTuesday exact GPT-4 prompt is not fully public | Its model/dataset are usable for benchmarking, but prompt reproduction is partial. |
| ANGEL nonprofit search repo is a placeholder | The paper is relevant, but code/data are not currently reusable. |
| NTEE/IRS labels are weak administrative priors | Use them as noisy labels or validation strata, not as the final construct definition. |
| Proprietary sources remain attractive but non-reproducible | Candid/GuideStar and Cause IQ may help validation if licensed, but cannot anchor open replication alone. |

## Recommended Reuse Order

1. Build weak-labeling and prompt logging around Snorkel, `gpt_annotate`, WRENCH patterns, and GivingTuesday prompt/domain examples.
2. Use Santamarina/NODC/IRS 1023-EZ mission data and NTEE/IRS religion labels as weak labels and benchmark strata.
3. Compare against GivingTuesday `religious_org_v1` as an external teacher/baseline, not ground truth.
4. Fine-tune DistilBERT, RoBERTa, DeBERTa, and ELECTRA with the scaffold from the short-text model report.
5. Evaluate with the imbalanced-validation report's metric bundle: PR-AUC, precision/recall, F1/F-beta, MCC, balanced accuracy, threshold curves, and uncertainty.
6. Calibrate probabilities with scikit-learn or `netcal`, then use QuaPy/`ppi_py` patterns for prevalence estimates and inference corrections.
7. Preserve prompts, labels, predictions, split manifests, model revisions, calibration objects, and audit samples so this project can publish its own replication package later.

## Prompt/Codebook Domains To Carry Forward

Use a multi-label codebook and derive the binary label after evidence capture:

| Domain | Carry-forward rule |
|---|---|
| `religious_purpose_explicit` | Positive when mission/purpose states worship, ministry, evangelism, religious education, faith formation, scripture, prayer, God/Christ/Allah/Torah/Quran/Dharma-linked purpose. |
| `religious_identity_or_affiliation` | Positive when name/text identifies a church, mosque, synagogue, temple, congregation, denomination, diocese, religious order, or named tradition. |
| `religious_service_content` | Positive when program text includes prayer, religious study, chaplaincy, pastoral care, worship, sacraments, religious counseling, missionary activity, or religious publishing/schooling. |
| `religious_governance_or_authority` | Probable positive/review when governance, sponsorship, or authority is tied to a denomination, diocese, congregation, or religious body. |
| `spiritual_or_faith_inspired_ambiguous` | Keep separate when only spirituality, sacredness, faith-inspired language, meditation/healing, saint names, or historic roots are present. |
| `administrative_religion_prior` | Treat NTEE X, IRS religious purpose, and old IRS religious activity codes as weak priors. |
| `negative_or_nonreligious_evidence` | Negative when text explicitly says secular/nonreligious/without regard to religion, or has no explicit religious/spiritual evidence. |

Require exact evidence spans for every positive, probable-positive, or ambiguous domain.
