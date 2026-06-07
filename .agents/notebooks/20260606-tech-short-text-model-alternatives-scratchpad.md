# Tech Scratchpad: Short-Text Model Alternatives

Created: 20260606
Topic Keywords: short-text; model-alternatives; BERT; LLM

---

## Problem Definition

### Research Questions

- Primary: Which BERT-like encoders and small open-weight LMs should be empirically compared for short-text binary classification of organization mission statements?
- Secondary: What fine-tuning recipes, metrics, and reproducibility practices are current best practice for small/noisy social-science text datasets?

### Upstream Inputs

- Replication-seeker output: `.agents/docs/20260605-replication-short-text-model-alternatives.md`
- Mode: replication-informed

### Success Criteria

- [x] Concise literature-backed Markdown report with 12-20 key papers.
- [x] Search queries and source links recorded.
- [x] Practical experiment grid recommendations included.
- [x] Version/deprecation/package compatibility notes included.

## Research Log

### Entry 1: Upstream replication handoff

THOUGHT: I need to reuse same-topic replication package evidence before searching broadly.
ACTION: Glob/read `.agents/docs/*-replication-*.md`; select complete same-topic file.
OBSERVATION:

- Source: `.agents/docs/20260605-replication-short-text-model-alternatives.md`
- Version: local artifact created 20260605
- Key Facts: Found handoff code packages for Sun et al., Gururangan et al., Laurer et al., Timoneda & Vallejo Vera, Ma, Zhang et al., Li et al., Political DEBATE.
- Confidence: High for repository existence in upstream handoff; individual file content still needs selective verification.
- New Questions: Which recent 2023-2026 small-LM papers best update the encoder-only evidence?

### Entry 2: Current literature and docs verification

THOUGHT: I need to verify whether recent small open-weight LMs change the default encoder-first recommendation.
ACTION: WebSearch current 2024-2026 BERT-vs-LLM text-classification comparisons; Context7 docs for Transformers/Sentence-Transformers/PEFT; websearch scikit-learn calibration docs.
OBSERVATION:

- Source: ACL Anthology, arXiv, Cambridge/SAGE pages, Hugging Face docs, scikit-learn docs.
- Version: Transformers v4.57.x/v5 docs; current Sentence-Transformers and PEFT docs; scikit-learn stable docs indexed as 1.8/1.9.
- Key Facts: Fine-tuned encoder-only models remain default with moderate labels; LLM prompting is competitive for no-label/tiny-label or knowledge-intensive tasks; embeddings are a strong cheap baseline; LoRA/SFT is a comparison arm with higher operational cost.
- Confidence: High for direction of evidence; Medium for frontier 2025-2026 arXiv papers pending peer review.
- New Questions: Need project-specific benchmark to determine exact winner under mission-statement labels and label noise.

## Draft Output

### Packages Found

Transformers, Sentence-Transformers, PEFT, scikit-learn.

### Code Snippets

Encoder fine-tuning, MiniLM embeddings + logistic regression, LoRA sequence classification.

### Version Notes

Record exact package versions and HF revisions; avoid isotonic calibration with small calibration sets; use `TaskType.SEQ_CLS` for PEFT classification.
