# Tech Scratchpad: Imbalanced Text Classification Evaluation for Social Science

Created: 2026-06-06
Topic Keywords: imbalanced text classification; evaluation metrics; validation design; social science measurement; noisy labels

---

## Problem Definition

### Research Questions

- Primary: What evaluation metrics and validation designs are current best practice for imbalanced binary short-text classifiers used for social-science population measurement?
- Secondary: How should precision/recall, PR-AUC, ROC-AUC, balanced accuracy, MCC, kappa, cost/utility, threshold curves, construct validity, hand-coded validation, prevalence uncertainty, noisy labels, and domain drift be reported?

### Upstream Inputs

- Replication-seeker output: `.agents/docs/20260605-replication-imbalanced-validation.md`
- Mode: replication-informed

### Success Criteria

- [x] Identify 10-18 key papers with stable citations or links.
- [x] Cover classic metrics foundations, CSS/social-science measurement, weak/noisy supervision, and recent 2023-2026 work where relevant.
- [x] Produce concise Markdown report and required Tech Seeker handoff artifact.
- [x] Version compatibility/deprecation notes translated into current metric/tooling caveats where package-specific versions are not central.

## Research Log

### Entry 1: Upstream discovery

THOUGHT: I need to reuse any same-topic replication-seeker handoffs before searching externally.
ACTION: Glob[.agents/docs/*-replication-*.md] and frontmatter filtering
OBSERVATION:

- Source: Local .agents/docs replication outputs
- Version: N/A
- Key Facts: Candidate same-topic files exist; filtering still pending.
- Confidence: Medium until frontmatter and handoff sections are read.
- New Questions: Which upstream output has the most relevant code-file handoff?

### Entry 2: Metric foundations and imbalanced classification

THOUGHT: I need classic and recent sources on ROC/PR, cost curves, MCC, balanced accuracy, kappa, and prevalence-sensitive metrics.
ACTION: WebSearch[imbalanced classification precision recall ROC PR AUC MCC balanced accuracy Cohen kappa F-beta cost curves DOI]
OBSERVATION:

- Source: PLOS ONE, Machine Learning, ACM/ICML, Springer/Pattern Recognition/ML venues
- Version: N/A
- Key Facts: Davis & Goadrich (2006), Fawcett (2006), Drummond & Holte (2006), Saito & Rehmsmeier (2015), Hand (2009), Brodersen et al. (2010), Chicco & Jurman (2020/2021), and recent 2024-2026 work support multi-metric, cost-aware reporting.
- Confidence: High for classic metric foundations; Medium for 2026 sources until peer reception matures.
- New Questions: How to adapt metric recommendations for population-scale measurement.

### Entry 3: Social-science validation and measurement

THOUGHT: I need social-science-specific sources on construct validity, hand-coded validation, inter-coder reliability, transparency, and final-measure validation.
ACTION: WebSearch[computational social science text as data validation measurement construct validity supervised machine learning text classification hand coded validation inter coder reliability]
OBSERVATION:

- Source: Political Analysis, JEL, Political Communication, Communication Methods and Measures, PSRM, PS: Political Science & Politics
- Version: N/A
- Key Facts: Grimmer & Stewart (2013) establish problem-specific validation; Gentzkow et al. (2019) frames economics text-as-data; Song et al. (2020) shows imperfect human labels distort validation; Birkenmaier et al. and Park & Montgomery provide recent validation/reporting frameworks.
- Confidence: High.
- New Questions: Need explicit guidance for aggregate prevalence.

### Entry 4: Population measurement and noisy labels

THOUGHT: I need sources that distinguish individual prediction from prevalence/aggregate measurement and sources on noisy/weak labels.
ACTION: WebSearch[Hopkins King quantification prevalence estimation noisy labels weak supervision text classification benchmark]
OBSERVATION:

- Source: AJPS, ACL, PMLR, arXiv, ACM/IEEE
- Version: N/A
- Key Facts: Hopkins & King (2010), Forman (2005/2008), Bella et al. (2010), Keith & O'Connor (2018), BOXWRENCH (2025), AlleNoise (2025), and NoisyAG-News (2024) motivate gold-standard audit samples, quantification-specific evaluation, and caution about instance-dependent label noise.
- Confidence: High for established quantification sources; Medium for newest arXiv work.
- New Questions: None critical.

## Draft Output

### Packages Found

scikit-learn, QuaPy, precrec, hmeasure, irr/krippendorff.

### Code Snippets

Added conceptual Python metric-report and adjusted-count snippets to handoff artifact.

### Version Notes

Recorded caveats on PR-AUC definitions, ROC-AUC under imbalance, MCC/kappa prevalence sensitivity, validation-set requirements, and quantification assumptions.
