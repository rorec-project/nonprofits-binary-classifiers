# Tech Scratchpad: Religious Identity Prompts and Textual Triggers

Created: 20260605
Topic Keywords: religious identity; prompts; survey wording; nonprofit mission; lexical triggers

---

## Problem Definition

### Research Questions

- Primary: Which academic literatures, official survey instruments, codebooks, and computational/NLP sources provide wording, prompts, annotation items, and lexical triggers for identifying religious identity or religious mission from text?
- Secondary: How should those sources inform LLM prompts, weak-supervision labeling functions, and human annotation instructions for binary religious mission classification?

### Upstream Inputs

- Replication-seeker output: `docs/research/20260605-replication-religious-vs-nonreligious-prompts.md` and `docs/research/20260605-replication-religious-nonprofit-mission.md`
- Mode: replication-informed

### Success Criteria

- [ ] Capture 12-25 high-priority sources across surveys, sociology/communication, nonprofit/org studies, CSS/NLP, and economics/public policy.
- [ ] Extract concrete question wording, codebook items, prompt phrases, and lexical triggers.
- [ ] Note implementation caveats, deprecation/recency where relevant for code/model artifacts.
- [ ] Save final report in `docs/research/` with requested filename prefix.

## Research Log

### Entry 1: Upstream replication handoff

THOUGHT: I need existing same-topic replication outputs to avoid duplicating package/code artifact discovery and to seed literature/source names.
ACTION: glob/read local `docs/research/*-replication-*.md`; selected religious prompt and religious nonprofit mission outputs.
OBSERVATION:

- Source: `docs/research/20260605-replication-religious-vs-nonreligious-prompts.md`, `docs/research/20260605-replication-religious-nonprofit-mission.md`
- Version: local completed outputs dated 20260605
- Key Facts: Open artifacts include GivingTuesday `religious_org_v1`, Santamarina bespoke NPO taxonomies, Paxton glossary/stemmer files, UK-CAT, NODC mission taxonomies, Ma NTEE classifier, and NLP religion datasets.
- Confidence: High for repository/artifact leads; still need external verification and broader survey/literature coverage.
- New Questions: Which official instruments provide exact wording, and which peer-reviewed sources define faith-based/faith-related organizations?

### Entry 2: Official survey/codebook wording

THOUGHT: I need authoritative question wording for religious affiliation, denomination, practice, salience, and cross-national comparability.
ACTION: WebSearch queries for GSS 2024 R3, Pew religious affiliation/NPORS/RLS wording, WVS Wave 8, and NCS cumulative codebook.
OBSERVATION:

- Source: NORC GSS 2024 Codebook R3; Pew 2018 FAQ, 2021 NPORS wording, 2025 RLS bridge/topline; WVS Wave 8 questionnaire v11; NCS I-IV cumulative codebook.
- Version: GSS 2024 R3 published 2026-03; Pew 2025/2021; WVS Wave 8 2024-2026; NCS cumulative 2020.
- Key Facts: Strong reusable wordings include “What is your present religion, if any?”, “religious preference,” “Would you describe yourself as born-again or evangelical Christian?”, “Aside from weddings and funerals...attend religious services,” “Outside of religious services...pray,” “Is your congregation formally affiliated with a denomination...?”
- Confidence: High; official survey/codebook sources.
- New Questions: Need translate these constructs into organization-text annotation dimensions.

### Entry 3: FBO/nonprofit and computational artifacts

THOUGHT: I need organization-level typologies, codebook dimensions, and public classifier artifacts for mission-text religious classification.
ACTION: WebSearch/WebFetch for Sider & Unruh; Bielefeld & Cleveland; Ebaugh/Chafetz/Pipes; Ma; Fyall/Gugerty/Moore; Santamarina; NODC; GivingTuesday; UK-CAT.
OBSERVATION:

- Source: SAGE/Wiley abstracts; nonprofit-open-data docs; Santamarina replication site; GivingTuesday Hugging Face model card; UK-CAT GitHub README.
- Version: Foundational FBO sources 2001-2013; nonprofit text classification 2018-2026; GivingTuesday dataset/model updated 2025 and accessed 2026; UK-CAT uses Python scripts and regex tags, public release 2021 with README updated for Python 3.13/2026 CIC data note.
- Key Facts: Sider & Unruh separate organization and program religiosity; Ebaugh et al. show religious imagery in public face/mission; NODC benchmark reports Religious purpose ICR .97 and ML accuracy .92; GivingTuesday BERT model classifies nonprofit name/mission/activity and reports BERT macro F1 .76 vs Llama macro F1 .27.
- Confidence: High for core claims; model-card claims should be treated as benchmark/teacher rather than peer-reviewed ground truth.
- New Questions: Need final recommendations emphasizing ambiguity handling and evidence spans.

## Draft Output

### Packages Found

Relevant implementation artifacts: GivingTuesday `religious_org_v1` (Transformers/Hugging Face), NODC mission classifiers (R/quanteda), Santamarina bespoke NPO taxonomies (R/quanteda), UK-CAT regex/logistic scripts.

### Code Snippets

Use multi-field classification over organization name + mission + activity/program text; return evidence spans and distinguish identity, activity, auspice, historical roots, and ambiguous spiritual language.

### Version Notes

Official survey instruments have mode/version changes (GSS 2024 web wording; Pew RLS bridge survey). Hugging Face model card is current but not peer-reviewed; use as benchmark/teacher. Administrative labels (NTEE X, IRS Religious Purpose) are high-value weak labels but incomplete.
