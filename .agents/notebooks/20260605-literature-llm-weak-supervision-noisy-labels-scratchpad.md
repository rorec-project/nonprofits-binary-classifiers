# Literature Scratchpad: LLM Annotators, Weak Supervision, and Noisy Labels for Mission Text Classification

Created: 2026-06-05
Topic Keywords: LLM annotation; weak supervision; noisy labels; transformer fine-tuning; text classification

---

## Problem Definition

### Research Questions

- Primary: Which papers discuss or apply LLM-as-annotator/judge, weak supervision/programmatic labels, and transformer fine-tuning on noisy labels for binary text classification in computational social science/applied economics NLP?
- Secondary: What validity, bias, reliability, prompt sensitivity, human-vs-LLM agreement, label aggregation, robustness, active learning, calibration, and audit-set lessons are most relevant for classifying organization mission text?

### Context & Constraints

- Timeline/Budget: Concise report requested; include classics plus recent 2023-2026 peer-reviewed or high-quality working papers/arXiv.
- Source priority: NLP/ML conferences and journals; computational social science, political science, sociology, and applied economics venues; central working papers/arXiv if recent.

### Success Criteria

- [ ] Cover 12-20 key papers spanning LLM annotation, weak supervision, and noisy-label transformer training.
- [ ] Include search strategy and exact queries.
- [ ] Trace citations backward/forward/sideways from key papers.
- [ ] Provide synthesis and practical experiment/validation recommendations.
- [ ] Include required handoff blocks with explicit entries or empty markers.

## Research Log

### Entry 1: Setup

THOUGHT: I need to establish the search target before querying sources.
ACTION: Define topic keywords and success criteria.
OBSERVATION:

- Source: User request and local literature-seeker templates.
- Citation: N/A.
- Date: 2026-06-05.
- Key Facts: Search must cover LLM-as-annotator/judge, weak supervision/data programming/Snorkel, and robust transformer fine-tuning under noisy LLM/rule-generated labels.
- Confidence: High because scope is explicitly specified by user.
- New Questions: Which 2023-2026 LLM annotation validity studies are most central, and which noisy-label methods transfer best to BERT fine-tuning?

### Entry 2: LLM Annotation in CSS and Political Text

THOUGHT: Need authoritative CSS/political science anchors on whether LLMs can substitute for or augment human coders.
ACTION: Search["2026 LLM as annotator text classification labels validity bias reliability prompt sensitivity human agreement computational social science"; "Gilardi Alizadeh Kubli ChatGPT outperforms crowd workers text annotation PNAS 2023 DOI"; "Ziems large language models transform computational social science text annotation 2024"; "Pangakis automated annotation with generative AI requires validation"; "Heseltine Clemm von Hohenberg large language models substitute human experts political text"].
OBSERVATION:

- Source: PNAS, Computational Linguistics, Research & Politics, Social Science Computer Review, Political Science Research and Methods, arXiv.
- Citation: Gilardi et al. (2023); Ziems et al. (2024); Pangakis et al. (2023); Heseltine & Clemm von Hohenberg (2024); Törnberg (2024); Ornstein et al. (2025); Mellon et al. (2024).
- Date: 2026-06-05.
- Key Facts: LLM annotation can be highly cost-effective but task-specific validation is non-negotiable; binary tasks often work better than multi-class/long-text/subjective constructs; hybrid disagreement adjudication improves accuracy.
- Confidence: High for published papers; Medium for arXiv-only prompt-sensitivity and BERT-risk papers.
- New Questions: Which validation design best fits mission-text binary labels?

### Entry 3: Weak Supervision and Prompted Labeling Functions

THOUGHT: Need foundations for programmatic labels, noisy labeling functions, and prompt-as-LF recipes.
ACTION: Search["Data Programming Creating Large Training Sets Quickly Ratner"; "Snorkel Rapid Training Data Creation with Weak Supervision DOI"; "WRENCH Comprehensive Benchmark for Weak Supervision"; "Language Models in the Loop Incorporating Prompting into Weak Supervision DOI"].
OBSERVATION:

- Source: NeurIPS, PVLDB, AAAI, NeurIPS Datasets & Benchmarks, ACM/IMS Journal of Data Science.
- Citation: Ratner et al. (2016, 2017, 2019); Zhang et al. (2021); Smith et al. (2024).
- Date: 2026-06-05.
- Key Facts: Labeling functions should be treated as noisy, overlapping, abstaining sources; probabilistic label models outperform naive heuristics when LF accuracies/correlations matter; prompts can be decomposed into multiple LFs and denoised via Snorkel-style aggregation.
- Confidence: High due to stable venues and DOIs.
- New Questions: How many rule/LLM LFs are enough for mission text, and how to audit LF dependence?

### Entry 4: Noisy Labels and BERT/Transformer Fine-Tuning

THOUGHT: Need evidence on fine-tuning encoder classifiers with weak/LLM labels and practical mitigation strategies.
ACTION: Search["BERT fine tuning noisy labels text classification label noise robustness"; "SaFER robust efficient framework fine-tuning BERT classifier noisy labels"; "Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance"; "Feeding LLM Annotations to BERT Classifiers at Your Own Risk"; "Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement"].
OBSERVATION:

- Source: ACL/EMNLP Findings/ICNLSP/arXiv.
- Citation: Zhu et al. (2022); Qi et al. (2023); Wang et al. (2023); Agro & Aldarmaki (2023); Lu & Smith (2025); Ye et al. (2025).
- Date: 2026-06-05.
- Key Facts: BERT is fairly robust to random noise but vulnerable to feature-dependent/programmatic/LLM errors; early stopping, self-supervised/contrastive regularization, LLM confidence guidance, ensembles, and iterative denoising are promising but not universal.
- Confidence: High for ACL/EMNLP papers; Medium for 2025 arXiv LLM-label-specific results.
- New Questions: Whether mission-text label noise is random, feature-dependent, or construct-boundary divergence.

## Draft Output

### Papers Found

See final report: `.agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md`.

### Coverage Assessment

Searched web across ACL Anthology, PNAS, SAGE, Cambridge Core, ACM DL, NeurIPS proceedings, PVLDB, arXiv, and OpenReview. Date range 2016-2026. Coverage strongest for NLP/ML/CSS; weaker for applied economics-specific mission-text classification.

### Citation Chains

Backward: LLM annotation papers cite Grimmer & Stewart validation principles, crowdsourcing, supervised text classification, Snorkel/data programming, BERT. Forward/sideways: Gilardi -> Pangakis/Heseltine/Törnberg/Ornstein/Mellon; Snorkel -> WRENCH/Language Models in the Loop; Zhu noisy-BERT -> SaFER/LAFT/realistic label-noise/LLM-label denoising.
