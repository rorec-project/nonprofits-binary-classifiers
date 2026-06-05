# Literature Scratchpad: Religious Nonprofit Mission Classification

Created: 2026-06-05
Topic Keywords: religious nonprofits; mission text classification; NTEE; Form 990; organizational taxonomy

---

## Problem Definition

### Research Questions

- Primary: Which papers, reports, and datasets discuss or apply NLP/text classification methods to identify religious missions or nonprofit organizational purpose from short organization text?
- Secondary: Which administrative/directory taxonomies and datasets (IRS Form 990, NTEE, NCCS/Urban Institute, GuideStar/Candid, charity registries, Cause IQ, similar sources) can support labels or validation? What measurement pitfalls affect religious/nonprofit mission classification?

### Context & Constraints

- Timeline/Budget: User requested a concise Markdown report for a computational social science / applied economics NLP project. Prioritize applied economics, nonprofit studies, sociology of religion, public administration, computational social science, NLP, peer-reviewed sources, reputable reports/datasets, and high-quality working papers.

### Success Criteria

- [ ] Search breadth covers nonprofit NLP classification, religious/faith-based organization measurement, organizational taxonomies, and administrative/directory data.
- [ ] Identify 10-18 key papers/reports/datasets with DOI/URL and practical relevance.
- [ ] Trace backward/forward/sideways from key classification/taxonomy sources and verify dataset claims from provider or reputable secondary sources.
- [ ] Produce actionable synthesis on construct validity, misclassification, benchmark labels, validation data, and audit samples.

## Research Log

### Entry 1: Initial source discovery

THOUGHT: Needed breadth across nonprofit NLP/text classification and administrative nonprofit datasets.
ACTION: Search[`"nonprofit" "mission statements" text classification NTEE Form 990 2026`]
OBSERVATION:

- Source: SAGE, Cambridge/Voluntas, GitHub, Nonprofit Open Data Collective.
- Citation: Ma (2021); Fyall, Moore & Gugerty (2018); Litofcenko, Karner & Maier (2020); Santamarina, Lecy & van Holm (2023); Nonprofit Mission Classifiers docs.
- Date: 2026-06-05.
- Key Facts: Core literature uses NTEE, IRS Form 990 mission/program text, IRS 1023-EZ purpose codes, organization names, and bespoke taxonomies. Religion-related benchmarks exist for NTEE X and IRS religious purpose.
- Confidence: High; journal/provider pages corroborate.
- New Questions: How to define religious mission beyond NTEE X?

### Entry 2: Religious/FBO construct validity

THOUGHT: Needed religious/faith-based nonprofit measurement sources and typologies.
ACTION: Search[`"faith-based" nonprofit classification IRS NTEE religious organizations measurement study`]; Search[`Sider Unruh typology faith based organizations religious characteristics mission statement Smith Sosin varieties faith-related agencies DOI`]
OBSERVATION:

- Source: SAGE NVSQ, Wiley JSSR/PAR, IRS/NCCS, ICPSR/NCS.
- Citation: Bielefeld & Cleveland (2013); Sider & Unruh (2004); Smith & Sosin (2001); Becker (2003); Scheitle & Dougherty (2015); National Congregations Study.
- Date: 2026-06-05.
- Key Facts: FBO definitions are multidimensional; organization-level religiosity differs from program-level religiosity; mission/name signals are visible but incomplete; churches are generally excepted from Form 990 filing.
- Confidence: High.
- New Questions: Need external benchmarks for congregations and non-filing religious groups.

### Entry 3: Dataset/taxonomy documentation

THOUGHT: Needed official data sources for labels/validation.
ACTION: Search[`NTEE NCCS GuideStar Candid Cause IQ IRS Form 990 nonprofit classification dataset religious`]; Search[`religious congregations nonprofit data IRS Form 990 churches exempt filing National Congregations Study ARDA`]
OBSERVATION:

- Source: NCCS/Urban, IRS, Candid PCS, Cause IQ, NCS, ARDA/ICPSR.
- Citation: NCCS NTEE docs; IRS BMF/e-file/SOI docs; Candid PCS; Cause IQ help; NCS cumulative file.
- Date: 2026-06-05.
- Key Facts: NTEE_IRS and NTEE_NCCS differ; NTEE-V2 separates industry from organization type; Candid PCS adds subject/population/auspice; Cause IQ provides proprietary/manual NTEE updates; NCS covers congregations absent from Form 990.
- Confidence: High.
- New Questions: How to combine administrative labels with hand-coded audits.

### Entry 4: Citation tracing and pitfalls

THOUGHT: Needed backward/sideways tracing from Ma, Fyall, Litofcenko, Santamarina.
ACTION: Search[`"Automated Coding Using Machine Learning and Remapping" "Beyond NTEE Codes" "Litofcenko" nonprofit classification`]; Search[`Grønbjerg 1994 using NTEE classify nonprofit organizations assessment human service regional applications Lampkin Barman NTEE`]
OBSERVATION:

- Source: SAGE/Springer/Cambridge/NCCS/ISTR.
- Citation: Grønbjerg (1994); Barman (2013); Hodgkinson & Weitzman (1994); Lampkin & Stengel; UK Charity Classification.
- Date: 2026-06-05.
- Key Facts: Known issues: single-code NTEE cannot capture multiple missions; incomplete/incorrect labels; administrative undercoverage; taxonomy choice changes observed sector composition; fine-grained text classification can be difficult when text is short/noisy.
- Confidence: High.
- New Questions: None critical.

## Draft Output

### Papers Found

See `.agents/docs/20260605-literature-religious-nonprofit-classification.md`.

### Coverage Assessment

Searched web/journal/provider sources: SAGE, Springer/Voluntas, Cambridge Core, Wiley, NCCS/Urban, IRS, Candid, Cause IQ, ICPSR/ARDA/NCS, UK Charity Classification, GOV.UK, GitHub replication pages.

### Citation Chains

Core chain: Grønbjerg (1994) / NTEE history -> Fyall et al. (2018), Litofcenko et al. (2020) -> Ma (2021), Santamarina et al. (2023), word-embedding/UK charity classification work. Religious construct chain: Smith & Sosin (2001), Becker (2003), Sider & Unruh (2004) -> Bielefeld & Cleveland (2013), FBO empirical reports and applied economics working paper definitions.
