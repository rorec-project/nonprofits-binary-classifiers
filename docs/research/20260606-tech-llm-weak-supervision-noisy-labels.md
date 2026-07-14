---
created: 20260606
agent: tech-seeker
scratchpad: docs/research/notebooks/20260606-tech-llm-weak-supervision-noisy-labels-scratchpad.md
status: complete
title: Tech - LLM Weak Supervision Noisy Labels
topic: llm annotator; weak supervision; noisy labels; transformer fine-tuning
---

# Tech: LLM Annotation, Weak Supervision, and Noisy Labels for Mission-Text Classification

## Search Strategy and Exact Search Queries Used

I started from the local replication handoff `docs/research/20260605-replication-weak-supervision-noisy-labels.md`, then searched forward/backward around central LLM-annotation, data-programming, WRENCH, BERT-noise, and LLM-label calibration papers. Exact external queries used:

1. `2026 LLM as annotator text classification validity bias reliability prompt sensitivity human agreement computational social science NLP`
2. `LLM annotator social science ChatGPT outperforms crowd workers text annotation Gilardi DOI Pangakis validation Törnberg 2025 Social Science Computer Review`
3. `data programming Snorkel weak supervision WRENCH benchmark label model DOI Ratner Zhang 2021 weak supervision NLP transformers`
4. `BERT fine-tuning noisy labels label noise robustness co-teaching label smoothing bootstrapping sample selection NLP text classification 2022 2023 2024 2025`
5. `LLM generated labels train BERT classifier noisy labels confidence filtering calibration active learning prediction powered inference text annotation 2024 2025 2026`
6. `prediction powered inference text annotations language model labels Egami Fong Grimmer Roberts Stewart 2023 2024 design based supervised learning text as data annotation error`
7. `Snorkel Python current version LabelModel PandasLFApplier documentation PyPI 2026 weak supervision WRENCH BERT-LNL SiDyP GitHub`
8. `"Knowledge Distillation in Automated Annotation" "Supervised Text Classification with LLM-Generated Training Labels" authors 2024 NLP+CSS`
9. `"Automated Annotation with Generative AI Requires Validation" authors DOI arXiv 2306.00176 "Can Large Language Models Transform Computational Social Science" DOI 2024`
10. `"SaFER: A Robust and Efficient Framework for Fine-tuning BERT-based Classifier with Noisy Labels" authors "Is BERT Robust to Label Noise" Zhu 2022 authors venue "Calibrating Pre-trained Language Classifiers" SiDyP authors`
11. `"Language Models in the Loop: Incorporating Prompting into Weak Supervision" authors 2024 DOI ACM`

## From Replication Packages

Same-topic handoff found: `docs/research/20260605-replication-weak-supervision-noisy-labels.md`. Preferred code/package leads:

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| Gilardi et al. 2023 | Python/R | ChatGPT zero-shot template; annotation/evaluation scripts | LLM zero-shot annotation and human/MTurk comparison | https://doi.org/10.7910/DVN/PQYF6M |
| Heseltine & Clemm von Hohenberg 2024 | R/Python | `PrimaryAnalysis_Final.R`, `CodingAnalysis.R`, `Python.rar` | GPT/human political-text coding and downstream model comparisons | https://doi.org/10.7910/DVN/V2P6YL |
| Pangakis et al. | Python | `gpt_annotate.py`, sample notebook | Validation-first LLM annotation workflow | https://github.com/npangakis/gpt_annotate |
| Snorkel | Python | `snorkel/labeling`, docs/tutorials | Labeling functions + probabilistic label model | https://github.com/snorkel-team/snorkel |
| WRENCH | Python | `wrench/labelmodel`, `wrench/endmodel` | Weak-supervision benchmark with label/end models | https://github.com/JieyuZ2/wrench |
| BERT-LNL | Python | `main.py`, `noise_functions.py`, `trainers/` | BERT fine-tuning under label noise | https://github.com/uds-lsv/BERT-LNL |
| SiDyP | Python/Shell | `src/`, `scripts/train.sh`, `datasets/llm/...` | Calibration/iterative refinement for LLM-generated noisy labels | https://github.com/gtfintechlab/SiDyP |

## Package Recommendations

| Language | Package | Version | Key Function | Source |
|----------|---------|---------|-------------|--------|
| Python | Snorkel | v0.10.0; Python >=3.11 | `PandasLFApplier`, `LabelModel(cardinality=2)` for LFs and probabilistic labels | https://pypi.org/project/snorkel/ ; https://snorkelproject.org/get-started/ |
| Python | WRENCH / `ws-benchmark` | latest repo release v1.1; install note `ws-benchmark==1.1.2rc0` | Benchmark label models/end models; compare majority vote, Snorkel, MeTaL, BERT/COSINE | https://github.com/JieyuZ2/wrench |
| Python | Hugging Face `transformers` + `datasets` | Project pins future/minimums in `pyproject.toml`: `transformers>=5.9.0`, `datasets>=4.8.5`; verify before install | Fine-tune BERT/RoBERTa/DistilBERT with early stopping, weighted losses, label smoothing | local `pyproject.toml` |
| Python | BERT-LNL codebase | research code, 2022 | Baselines for noisy-label BERT: vanilla, label smoothing, co-teaching, noise matrix | https://github.com/uds-lsv/BERT-LNL |
| Python | SiDyP | KDD 2025 code | LLM-generated label denoising/calibration via dynamic prior and simplex diffusion | https://github.com/gtfintechlab/SiDyP |
| R | `dsl` | current project package | Design-based supervised learning for valid downstream inference with LLM/surrogate labels | http://naokiegami.com/dsl/ |

## Key Papers to Read and Cite

| # | Paper | Authors / year | Venue/source | DOI/link | Why it matters |
|---|-------|----------------|--------------|----------|----------------|
| 1 | *Data Programming: Creating Large Training Sets, Quickly* | Ratner, De Sa, Wu, Selsam & Ré, 2016 | NeurIPS | https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly | Classic foundation for programmatic labeling functions, label-model denoising, and noise-aware discriminative training. |
| 2 | *Snorkel: Rapid Training Data Creation with Weak Supervision* | Ratner et al., 2017 | PVLDB | https://doi.org/10.14778/3157794.3157797 | Practical system version of data programming; central for LF coverage/conflict/abstention workflow. |
| 3 | *WRENCH: A Comprehensive Benchmark for Weak Supervision* | Zhang, Yu, Li, Wang, Yang, Yang & Ratner, 2021 | NeurIPS Datasets & Benchmarks | https://openreview.net/forum?id=Q9SKS5k8io | Standard benchmark for comparing label models and end models, including BERT-style end models under weak labels. |
| 4 | *Language Models in the Loop: Incorporating Prompting into Weak Supervision* | Smith, Fries, Hancock & Bach, 2024 | ACM/IMS Journal of Data Science | https://doi.org/10.1145/3617130 | Treats LLM prompts as weak supervision sources/labeling functions; directly bridges LLM prompting and Snorkel-style aggregation. |
| 5 | *ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks* | Gilardi, Alizadeh & Kubli, 2023 | PNAS | https://doi.org/10.1073/pnas.2305016120 | Early high-impact evidence that zero-shot ChatGPT can beat MTurk on several political/news annotation tasks, but prompt/task caveats remain. |
| 6 | *Automated Annotation with Generative AI Requires Validation* | Pangakis, Wolken & Fasching, 2023 | arXiv | https://doi.org/10.48550/arXiv.2306.00176 | Validation-first workflow; shows LLM performance varies strongly by task and dataset. |
| 7 | *Keeping Humans in the Loop: Human-Centered Automated Annotation with Generative AI* | Pangakis & Wolken, 2025 | ICWSM | https://doi.org/10.1609/icwsm.v19i1.35883 | Peer-reviewed extension: 27 tasks across protected CSS datasets; argues human validation labels remain essential. |
| 8 | *Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels* | Pangakis & Wolken, 2024 | NLP+CSS | https://doi.org/10.18653/v1/2024.nlpcss-1.9 | Most directly relevant to fine-tuning BERT/RoBERTa/DistilBERT on GPT-4 labels across CSS tasks. |
| 9 | *Can Large Language Models Transform Computational Social Science?* | Ziems, Held, Shaikh, Chen, Zhang & Yang, 2024 | Computational Linguistics | https://doi.org/10.1162/coli_a_00502 | Broad CSS benchmark: LLMs can augment annotation, but fine-tuned classifiers often remain stronger for classification. |
| 10 | *Large Language Models Outperform Expert Coders and Supervised Classifiers at Annotating Political Social Media Messages* | Törnberg, 2025 | Social Science Computer Review | https://doi.org/10.1177/08944393241286471 | Strong pro-LLM evidence on political social media; useful counterpoint to validation/cautionary papers. |
| 11 | *Best Practices for Text Annotation with Large Language Models* | Törnberg, 2024 | Sociologica | https://sociologica.unibo.it/article/view/19461 | Practical standards: structured prompts, prompt stability, validation, reproducibility, ethics/legal constraints. |
| 12 | *Is BERT Robust to Label Noise? A Study on Learning with Noisy Labels in Text Classification* | Zhu, Hedderich, Zhai, Adelani & Klakow, 2022 | Insights from Negative Results in NLP | https://doi.org/10.18653/v1/2022.insights-1.8 | Key caution: BERT is fairly robust to random injected noise but weak-supervision/feature-dependent noise is harder; early stopping matters. |
| 13 | *SaFER: A Robust and Efficient Framework for Fine-tuning BERT-based Classifier with Noisy Labels* | Qi, Tan, Qu, Xu & Qi, 2023 | ACL Industry Track | https://aclanthology.org/2023.acl-industry.38/ | BERT-specific noisy-label framework with label-agnostic early stopping and contrastive/structural learning. |
| 14 | *Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance* | Wang et al., 2023 | EMNLP Findings | https://aclanthology.org/2023.findings-emnlp.834/ | LAFT uses LLM confidence scores as external guidance for separating clean/hard/noisy samples; conceptually useful though code was not reusable in handoff. |
| 15 | *Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement* | Ye, Shah, Zhang & Chava, 2025 | KDD | https://arxiv.org/abs/2505.19675 | Current, directly targeted method for BERT classifiers trained on zero/few-shot LLM labels; reports large gains from label diffusion/refinement. |
| 16 | *Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of LLMs* | Egami, Hinck, Stewart & Wei, 2023 | NeurIPS | https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html | Essential if classifier outputs become variables in regressions/estimation; high accuracy alone does not guarantee unbiased downstream inference. |
| 17 | *Prediction-Powered Inference* | Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, 2023 | Science | https://www.science.org/doi/10.1126/science.adi6000 | General statistical framework for combining many machine predictions with fewer gold labels for valid inference. |
| 18 | *VariErr NLI: Separating Annotation Error from Human Label Variation* | Weber-Genzel, Peng, De Marneffe & Plank, 2024 | ACL | https://doi.org/10.18653/v1/2024.acl-long.123 | Reminds that disagreement may be signal, not noise; relevant for ambiguous mission texts. |
| 19 | *What Is Actually Being Annotated? Inter-Prompt Reliability as a Measurement Problem in LLM-Based Social Science Labeling* | 2026 preprint | arXiv | https://arxiv.org/abs/2604.16413 | Recent prompt-sensitivity framework; supports multi-prompt reliability and prompt aggregation rather than single-prompt labels. |
| 20 | *Large Language Models Reproduce Racial Stereotypes When Used for Text Annotation* | 2026 preprint | arXiv | https://arxiv.org/pdf/2603.13891 | Important caveat on bias: model labels can shift with implicit identity/dialect cues; relevant for nonprofit mission language if correlated with group identity. |

## Synthesis for Training a Binary Classifier on Noisy/LLM/Programmatic Labels

1. **Treat labels as measurements, not truth.** LLM labels and rule labels are surrogate measurements with random and systematic error. High held-out accuracy can still bias downstream prevalence or regression estimates if errors are correlated with organization type, text length, region, denomination, or mission-writing style.
2. **Use a two-layer labeling architecture.** Combine: (a) deterministic labeling functions for high-precision religious/non-religious cues, (b) LLM prompts as additional labeling functions, and (c) a small human gold/audit set. Aggregate with Snorkel-style probabilistic labels rather than immediately collapsing to hard labels.
3. **Measure LF/prompt behavior before model training.** Report coverage, overlap, conflict, abstention rate, class balance, prompt-prompt agreement, LLM-human agreement, and subgroup performance.
4. **BERT can tolerate some noise, but weak-supervision noise is not benign.** Zhu et al. show BERT is robust to random label flips but more vulnerable to feature-dependent weak-supervision noise. Mission text will likely have feature-dependent noise: churches, schools, hospitals, cultural associations, and social-service nonprofits may share vocabulary.
5. **Early stopping and audit-set design are core.** Noisy validation sets can mislead, especially at high or feature-dependent noise. Keep a frozen human-labeled validation/test set, stratified by label source, model confidence, length, and high-disagreement cases.
6. **LLM-generated labels need extra diagnostics.** Use repeated prompts, low temperature for production annotation, prompt paraphrase/order perturbations, and cross-model disagreement. Do not report final quality on the same labels used for prompt selection.
7. **Confidence filtering is useful but risky.** Filtering to high-consistency LLM labels can improve precision but may remove hard/minority/borderline examples and shift the training distribution. Compare “all labels” vs “high-confidence only” vs “confidence-weighted loss”.
8. **Human-in-the-loop should target disagreement, not random cleanup only.** Sample for audit from: LLM-vs-rule conflicts, high classifier uncertainty, high-loss training points, minority/rare organization types, and near-threshold predicted probabilities.

## Implementation Examples

### Example 1: Snorkel-style aggregation for mission-text labels

**Source**: Snorkel get-started docs and PyPI v0.10.0 — https://snorkelproject.org/get-started/ ; https://pypi.org/project/snorkel/
**Package version**: Snorkel 0.10.0; Python >=3.11

```python
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

ABSTAIN, NONRELIGIOUS, RELIGIOUS = -1, 0, 1

@labeling_function()
def lf_religious_terms(x):
    text = x.text.lower()
    return RELIGIOUS if any(t in text for t in ["church", "ministry", "gospel", "worship"]) else ABSTAIN

@labeling_function()
def lf_llm_label(x):
    # Use a persisted LLM label column; do not call APIs inside LF execution.
    return int(x.llm_label) if x.llm_label in [0, 1] else ABSTAIN

lfs = [lf_religious_terms, lf_llm_label]
L_train = PandasLFApplier(lfs).apply(df_train)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)

df_train["weak_label"] = label_model.predict(L_train, tie_break_policy="abstain")
df_train[["p_nonreligious", "p_religious"]] = label_model.predict_proba(L_train)
df_train = df_train[df_train.weak_label != ABSTAIN]
```

**Notes**: Keep LLM outputs persisted with prompt/model/date/temperature metadata. Use probabilities for confidence weighting or audits; do not silently discard abstentions without reporting coverage.

### Example 2: Experimental grid for BERT fine-tuning on noisy labels

**Source**: Zhu et al. 2022; SaFER 2023; Pangakis & Wolken 2024; SiDyP 2025.
**Package version**: Local project currently declares `torch>=2.12.0`, `transformers>=5.9.0`, `scikit-learn>=1.9.0`; verify availability before lock/update.

```python
EXPERIMENTS = [
    "bert_hard_weak_labels",
    "bert_llm_only_labels",
    "bert_snorkel_prob_labels",
    "bert_high_confidence_only",          # e.g., max weak-label prob >= .8
    "bert_confidence_weighted_loss",      # sample weight = max weak-label prob
    "bert_label_smoothing_005",
    "bert_disagreement_reviewed_labels",  # replace audited conflicts with human labels
]

AUDIT_STRATA = [
    "llm_rule_disagree",
    "low_weak_label_confidence",
    "classifier_probability_0.4_to_0.6",
    "minority_predicted_class",
    "short_or_generic_mission_text",
]
```

**Notes**: The first baseline should be simple BERT/RoBERTa with fixed seeds and early stopping on human validation labels. Add noisy-label methods only if they improve human-held-out F1/MCC/calibration, not just weak-label agreement.

## Practical Recommendations for Experiments and Validation

1. **Create three human-labeled splits**: prompt-development set, prompt-selection validation set, and untouched final test/audit set. Do not tune prompts on the final set.
2. **Log every labeling run**: prompt text, model ID, provider, temperature, date, response format, parsing failures, and raw answer.
3. **Run at least two prompt variants** and one label-order perturbation. Use disagreement as an uncertainty feature.
4. **Compare label sources**: human-only small model, LLM-only labels, rule-only labels, majority vote, Snorkel label model, and hybrid human-corrected labels.
5. **Report robust metrics** for binary/imbalanced classification: macro F1, minority-class precision/recall, balanced accuracy, MCC, PR-AUC, calibration curves/Brier score, and prevalence error.
6. **Use confidence cautiously**: evaluate whether filtering high-confidence LLM labels improves test performance or just removes hard cases.
7. **Audit by strata**: label-source disagreement, predicted class, confidence deciles, mission-text length, organization category, geography if available, and terms likely to proxy identity or denomination.
8. **Human-in-the-loop loop**: review conflicts and uncertain examples; add corrected labels; retrain; compare gains against a random-audit baseline.
9. **Check temporal/model drift** if using proprietary LLMs: rerun a fixed canary set when model IDs change.
10. **If outputs feed economic/social inference**, use DSL/PPI-style correction or at least report sensitivity of prevalence/regression results to plausible false-positive/false-negative rates.

## Version and Compatibility Notes

- **Snorkel**: v0.10.0 is current on PyPI in search results; requires Python >=3.11. This repository requires Python >=3.13, so test install compatibility before adopting.
- **WRENCH**: useful as benchmark/reference, but latest release signal is older (`v1.1`; install note `ws-benchmark==1.1.2rc0`). Prefer isolated environment if used.
- **Local dependency pins look future-facing**: `transformers>=5.9.0`, `torch>=2.12.0`, `scikit-learn>=1.9.0`, `pandas>=3.0.3` in `pyproject.toml` may not correspond to current public stable releases in many environments; verify with `uv lock` before implementation.
- **BERT-LNL**: research code from 2022; valuable for baselines but may require dependency modernization.
- **LAFT**: conceptually relevant, but upstream handoff found the official GitHub effectively empty; do not depend on it as reusable code.
- **SiDyP**: newest and most targeted to LLM-generated labels; code exists, but as a 2025 research repo it should be treated as experimental and compared to simpler baselines.
- **Closed LLM APIs**: reproducibility risk from model updates; pin exact model version when available and persist raw outputs.

## Gaps, Caveats, and Read-First Papers

### Read first

1. Pangakis & Wolken (2024), *Knowledge Distillation in Automated Annotation* — direct analogue for training BERT-family classifiers on LLM labels.
2. Zhu et al. (2022), *Is BERT Robust to Label Noise?* — best caution against assuming weak-label noise behaves like random flips.
3. Ratner et al. (2016/2017) + WRENCH (2021) — foundation for programmatic labels and label aggregation.
4. Gilardi et al. (2023), Pangakis & Wolken (2025), Ziems et al. (2024) — balanced evidence on LLM annotation validity.
5. Egami et al. (2023/2024) or PPI (2023) — if classifier outputs become variables in downstream applied-economics analysis.

### Main caveats

- Mission-text labels may be conceptually ambiguous: “religious” can refer to identity, activities, affiliation, doctrine, or beneficiary group. Clarify construct before optimizing labels.
- LLM agreement with humans is task-specific; do not import performance claims from political tweets/news without a mission-text audit.
- Prompt sensitivity and label-order effects are real enough to require multi-prompt checks.
- Bias audits should include implicit identity and dialect/style cues, not only explicit protected-class terms.
- High-confidence filtering can underrepresent hard borderline nonprofits and hurt recall.

### Gaps

- Little published work specifically on nonprofit mission statements and religious/non-religious labels.
- Few reusable packages implement modern LLM-label denoising recipes as stable libraries; much is research code.
- Need a public mission-text benchmark with human labels, LFs, LLM prompt outputs, and calibrated BERT predictions.
