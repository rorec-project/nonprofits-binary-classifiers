# Tech Scratchpad: LLM Weak Supervision and Noisy Labels for Text Classification

Created: 2026-06-06
Topic Keywords: llm annotator, weak supervision, noisy labels, transformer fine-tuning

---

## Problem Definition

### Research Questions

- Primary: What academic and technical literature supports binary classification of organization mission text using LLM-generated, rule-generated, or programmatic noisy labels?
- Secondary: What implementation/validation recipes are current best practice for fine-tuning BERT-like classifiers under label noise and weak supervision?

### Upstream Inputs

- Replication-seeker output: `docs/research/20260605-replication-weak-supervision-noisy-labels.md`
- Mode: replication-informed

### Success Criteria

- [x] Search strategy and exact queries recorded.
- [x] 12-20 key papers identified with source links/DOIs when available.
- [x] Practical recommendations translated into experiments for noisy/LLM/programmatic labels.
- [x] Version/package/implementation caveats noted where relevant.

## Research Log

### Entry 1: Setup and upstream discovery

THOUGHT: Need to identify same-topic replication outputs before external exploration.
ACTION: Glob[docs/research/*-replication-*.md]
OBSERVATION:

- Source: local docs/research
- Version: N/A
- Key Facts: Candidate replication files found; next step is topic/status filtering.
- Confidence: High for local file enumeration.
- New Questions: Which candidate has agent/status/topic matching this request?

### Entry 2: Upstream handoff selected

THOUGHT: Need same-topic replication package/code context before external literature search.
ACTION: Grep/Read[`20260605-replication-weak-supervision-noisy-labels.md` frontmatter and handoff block]
OBSERVATION:

- Source: `docs/research/20260605-replication-weak-supervision-noisy-labels.md`
- Version: N/A
- Key Facts: Selected same-topic handoff; useful code/package leads include Gilardi Dataverse, Pangakis `gpt_annotate`, Snorkel, WRENCH, BERT-LNL, SiDyP.
- Confidence: High; local metadata has `agent: replication-seeker`, `status: complete`, matching topic keywords.
- New Questions: Need current bibliographic metadata and package compatibility.

### Entry 3: External literature and package verification

THOUGHT: Need authoritative sources for LLM annotation validity, weak supervision, and noisy-label BERT fine-tuning.
ACTION: WebSearch[11 exact queries recorded in output]
OBSERVATION:

- Source: PNAS, ACL Anthology, ACM, NeurIPS, KDD/arXiv, ICWSM, PyPI/GitHub.
- Version: Snorkel v0.10.0; WRENCH v1.1 / `ws-benchmark==1.1.2rc0`; local pyproject dependency minimums checked.
- Key Facts: LLM labels can be useful but require task-specific validation; Snorkel/WRENCH remain best weak-supervision frameworks; BERT is robust to random noise but weak/feature-dependent and LLM-generated noise need audits, early stopping, and calibration/denoising baselines.
- Confidence: Medium-high; peer-reviewed core sources plus recent preprints for 2026 frontier caveats.
- New Questions: Exact model/library versions should be rechecked before implementation because local pins appear future-facing.

## Draft Output

### Packages Found

Snorkel, WRENCH, Hugging Face transformers/datasets, BERT-LNL, SiDyP, R `dsl`.

### Code Snippets

Snorkel LF aggregation example; experimental grid for BERT noisy-label fine-tuning.

### Version Notes

Snorkel requires Python >=3.11; local project Python >=3.13 should be compatible in principle but needs install testing. WRENCH and BERT-LNL are older research code. SiDyP is new research code. LAFT official repo was effectively empty in upstream handoff.
