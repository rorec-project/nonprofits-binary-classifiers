---
created: 20260606
status: complete
title: Tech Synthesis Map - Literature-Prompt Matched Agents
reports:
  - docs/research/20260606-tech-llm-weak-supervision-noisy-labels.md
  - docs/research/20260606-tech-imbalanced-text-evaluation.md
  - docs/research/20260606-tech-calibration-quantification-prevalence.md
  - docs/research/20260606-tech-short-text-model-alternatives.md
  - docs/research/20260606-tech-religious-nonprofit-mission.md
  - docs/research/20260606-tech-religious-identity-prompts.md
  - docs/research/20260606-tech-religious-vs-nonreligious-mission-prompts.md
---

# Tech Synthesis Map: Literature-Prompt Matched Agents

## Completion Check

Seven `tech-seeker` agents were launched using the exact prompts previously sent to the seven `literature-seeker` agents. No replication-matched prompts were launched.

| Prior literature workstream | Tech report | Scratchpad | Note |
|---|---|---|---|
| LLM annotators, weak supervision, noisy labels | `docs/research/20260606-tech-llm-weak-supervision-noisy-labels.md` | `docs/research/notebooks/20260606-tech-llm-weak-supervision-noisy-labels-scratchpad.md` | Complete |
| Imbalanced text evaluation | `docs/research/20260606-tech-imbalanced-text-evaluation.md` | `docs/research/notebooks/20260606-tech-imbalanced-text-evaluation-scratchpad.md` | Complete |
| Calibration, quantification, prevalence | `docs/research/20260606-tech-calibration-quantification-prevalence.md` | `docs/research/notebooks/20260606-tech-calibration-quantification-prevalence-scratchpad.md` | Complete |
| Short-text model alternatives | `docs/research/20260606-tech-short-text-model-alternatives.md` | `docs/research/notebooks/20260606-tech-short-text-model-alternatives-scratchpad.md` | Complete |
| Religious nonprofit mission classification | `docs/research/20260606-tech-religious-nonprofit-mission.md` | `docs/research/notebooks/20260606-tech-religious-nonprofit-mission-scratchpad.md` | Complete |
| Religious identity prompts/triggers | `docs/research/20260606-tech-religious-identity-prompts.md` | `docs/research/notebooks/20260605-tech-religious-identity-prompts-scratchpad.md` | Complete; separated from the literature handoff. |
| Religious vs nonreligious mission prompts | `docs/research/20260606-tech-religious-vs-nonreligious-mission-prompts.md` | `docs/research/notebooks/20260606-tech-religious-mission-prompts-scratchpad.md` | Complete; separated from the literature handoff. |

## Technical Implementation Takeaways

| Area | Primary technical direction |
|---|---|
| Weak supervision and LLM labels | Use a validation-first workflow: prompt variants, repeated LLM labels, disagreement tracking, Snorkel-style labeling functions with abstentions, and clean human audit labels. |
| Noisy-label transformer training | Start with plain encoder fine-tuning, then compare label smoothing, confidence weighting, soft-label training, high-loss/disagreement filtering, and seed ensembles. |
| Imbalanced metrics | Make PR-AUC/average precision, precision, recall, F1/F-beta, MCC, balanced accuracy, threshold curves, and bootstrap intervals core outputs; keep ROC-AUC secondary. |
| Calibration and prevalence | Prefer calibrated probabilities and probabilistic classify-and-count for prevalence; compare Platt/logistic, temperature scaling, and isotonic calibration; use ACC/PACC/EM as sensitivity checks. |
| Model alternatives | Use TF-IDF and MiniLM/SBERT baselines, then DistilBERT, RoBERTa, DeBERTa, ELECTRA, and optionally ModernBERT or small generative LMs as comparison arms. |
| Religious nonprofit data | Use NTEE X, IRS religious-purpose codes, Santamarina/NODC mission corpora, GivingTuesday `religious_org_v1`, and NCS/ARDA as weak labels, benchmarks, or audit strata. |
| Religious prompt/codebook design | Code separate domains first: explicit purpose, identity/affiliation, service content, governance/authority, ambiguous spirituality, secular roots, administrative prior, and negative/secular evidence. |

## Package And Artifact Shortlist

| Task | Suggested implementation assets |
|---|---|
| LLM annotation workflow | `gpt_annotate`, Gilardi/Heseltine prompt logging patterns, OpenAI API batch logging if used, structured JSON outputs. |
| Weak supervision | Snorkel, WRENCH examples, BOXWRENCH patterns for realistic weak-label evaluation. |
| Transformer fine-tuning | Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`; model IDs: `distilbert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`, `google/electra-base-discriminator`. |
| Embedding baseline | `sentence-transformers/all-MiniLM-L6-v2` plus scikit-learn logistic regression. |
| Metrics | scikit-learn metrics, `precrec` as R reference, threshold tables, bootstrap CIs. |
| Calibration | scikit-learn calibration tools, `netcal`, temperature-scaling pattern from `gpleiss/temperature_scaling`. |
| Prevalence/quantification | QuaPy for CC/ACC/PCC/PACC/EM; `ppi_py` for prediction-powered inference; `freq-e` for prevalence intervals. |
| Religious nonprofit benchmark | GivingTuesday `religious_org_v1` and `religious_orgs_training`; Santamarina/NODC mission data; Ma `npo_classifier`; UK-CAT for external stress testing. |

## Experiment Scaffold

1. Freeze data splits and record `DATA_OF_CHOICE`, label source, seed, and class prevalence.
2. Build human-audited labels before tuning prompts or thresholds.
3. Generate weak labels from rules, administrative labels, and LLM prompt variants with abstentions and evidence spans.
4. Train baselines: TF-IDF logistic/SVM and MiniLM embeddings plus logistic regression.
5. Train encoder models with learning rates `{1e-5, 2e-5, 3e-5, 5e-5}`, epochs `{3, 5, 8, 10}`, batch sizes `{16, 32}`, max length `{128, 256}`, and at least 5 seeds.
6. Evaluate on clean audit labels with the imbalanced metric bundle and subgroup error checks.
7. Calibrate finalist probabilities on a held-out calibration set.
8. Estimate prevalence by averaging calibrated probabilities and report uncertainty.
9. Publish prompts, codebook, label-function definitions, split manifests, predictions, calibration objects, and metric JSON outputs.

## Prompt/Codebook Domains

| Domain | Rule |
|---|---|
| `religious_purpose_explicit` | Mission/purpose explicitly states worship, ministry, evangelism, religious education, faith formation, scripture, prayer, or deity/tradition-linked purpose. |
| `religious_identity_or_affiliation` | Name/text identifies a religious institution, denomination, tradition, congregation, diocese, parish, synagogue, mosque, temple, church, order, or faith affiliation. |
| `religious_service_content` | Program/activity text includes prayer, scripture study, chaplaincy, worship, sacraments, religious counseling, missionary work, evangelism, pastoral care, or religious schooling. |
| `religious_governance_or_authority` | Governance, sponsorship, authority, staff selection, or affiliation is tied to a religious body. |
| `religious_resources_or_networks` | Religious funding, partner networks, denominational support, or religious community resources are present; use as review/probable evidence, not decisive alone. |
| `spiritual_or_faith_inspired_ambiguous` | Spiritual or faith-inspired terms occur without clear tradition, identity, or religious activity. |
| `secular_with_religious_roots` | Historical religious roots or name, but current mission/activity is secular or explicitly nonsectarian. |
| `administrative_religion_prior` | NTEE X, IRS religious purpose, old IRS religious activity codes, or similar administrative labels. |
| `negative_or_secular_evidence` | Explicit secular/nonreligious wording, `without regard to religion`, or no explicit religious evidence. |

Require exact evidence spans for any positive, probable-positive, or ambiguous code.
