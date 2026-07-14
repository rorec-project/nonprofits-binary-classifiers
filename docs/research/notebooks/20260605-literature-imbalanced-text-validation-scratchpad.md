# Literature Scratchpad: Imbalanced Text Classification Validation for Social Science Mission Text

Created: 2026-06-05
Topic Keywords: imbalanced text classification; rare-class detection; social measurement; validation metrics; noisy labels

---

## Problem Definition

### Research Questions

- Primary: Which academic papers discuss evaluation metrics and validation designs for imbalanced binary text classification, especially rare/minority-class detection in computational social science or applied-economics-style population measurement?
- Secondary: Which papers connect classifier metrics to construct validity, hand-coded validation, coder reliability, prevalence uncertainty, noisy labels/weak supervision, threshold choice, and downstream population-scale measurement?

### Context & Constraints

- Timeline/Budget: Concise report requested; prioritize classic foundations plus 2023-2026 updates where relevant.
- Scope: Computational social science, applied economics, political science, sociology, survey methodology, information retrieval, NLP, and ML evaluation.

### Success Criteria

- [ ] Identify 10-18 key papers with DOI/link and why each matters.
- [ ] Include exact search queries and source coverage.
- [ ] Trace citations backward/forward/sideways from central papers.
- [ ] Synthesize best-practice metrics and validation designs for imbalanced short-text social measurement.
- [ ] Include explicit handoff citation and dataset blocks.

## Research Log

### Entry 1: Setup

THOUGHT: I need breadth across ML metrics, IR/NLP evaluation, computational social science measurement, and noisy-label validation before narrowing to papers.
ACTION: Prepare initial topic key set and scratchpad.
OBSERVATION:

- Source: User request and local templates.
- Citation: None.
- Date: 2026-06-05.
- Key Facts: Final output must include search strategy, papers, synthesis, recommendations, gaps, and papers to read first.
- Confidence: High (direct user requirements).
- New Questions: Which recent 2023-2026 papers are most relevant to social-science classifier validation under label noise and imbalance?

## Draft Output

### Papers Found

Core metric foundations: Davis & Goadrich (2006); Saito & Rehmsmeier (2015); Fawcett (2006); Drummond & Holte (2006); Hand (2009); Hernández-Orallo et al. (2012); Brodersen et al. (2010); Chicco & Jurman (2020/2021); Cohen (1960); Matthews (1975).

Social measurement / text-as-data: Hopkins & King (2010); Grimmer & Stewart (2013); Gentzkow, Kelly & Taddy (2019); Keith & O'Connor (2018); Nelson et al. (2021); Birkenmaier et al. (ValiText, 2023/2024); Toward trustworthy SML text measures (PSRM 2025); Codebook LLMs (Political Analysis 2025); Gilardi et al. (2024); recent short-text evaluation paper (2024); BOXWRENCH (NeurIPS 2024); AlleNoise (AISTATS 2025).

### Coverage Assessment

Sources checked: web search across Cambridge Core, PLOS, JMLR, Springer/Machine Learning, ACL Anthology, PMLR, arXiv, SAGE, NBER/RePEc-adjacent economics text-as-data materials, and author PDFs. Date range: 1960-2026.

### Citation Chains

Backward chains: Saito & Rehmsmeier cites Davis & Goadrich, Fawcett, Drummond & Holte. Hernández-Orallo et al. links threshold choice to Hand, Drummond & Holte, proper scoring/calibration. Social measurement chain: PSRM 2025 cites Grimmer & Stewart, Grimmer/Roberts/Stewart book, Kapoor et al.; ValiText cites content-analysis validity/reliability tradition. Aggregate measurement chain: Hopkins & King -> Keith & O'Connor -> 2025/2026 prediction-powered/prevalence calibration work.
