# Literature Scratchpad: Religious vs Nonreligious Mission Prompts

Created: 20260605
Topic Keywords: religious mission prompts; faith-based organization codebooks; nonprofit mission classification; religion survey wording; weak supervision triggers

---

## Problem Definition

### Research Questions

- Primary: Which academic works, official codebooks, datasets, and high-quality reports provide prompt-like wording, annotation instructions, survey questions, definitions, dictionaries, or lexical triggers that can guide classification of short organization mission/activity text as religious vs non-religious?
- Secondary: How should these sources shape project label domains, positive/negative/ambiguous rules, evidence-span requirements, LLM prompts, and weak-supervision labeling functions?

### Context & Constraints

- Timeline/Budget: Targeted same-day search; use existing synthesis map and prior reports as background; emphasize 2020-2026 LLM/CSS prompt work while including classic FBO and survey-codebook sources.
- Scope: Literature/source discovery only; no coding pipeline changes.
- Most authoritative source types: peer-reviewed nonprofit/sociology/CSS papers; official survey/codebook documentation from Pew, GSS/NORC, WVS/EVS, NCS/ARDA; official nonprofit taxonomies/dataset documentation (NTEE/NCCS/IRS/Candid); reproducible model/dataset cards when peer-reviewed sources are thin.

### Success Criteria

- [ ] Report 12-25 key sources with links/DOIs and source relevance.
- [ ] Extract actual prompt-like wording: LLM prompts, annotation instructions, survey questions, codebook criteria, dictionary categories, or lexical triggers.
- [ ] Propose a concrete multi-domain prompt/codebook architecture with binary derivation rules and evidence-span requirements.
- [ ] Include authoritative definitions and boundary-case warnings.
- [ ] Include required literature-seeker handoff blocks: Papers, Coverage Notes, Citation List, Datasets Mentioned.

## Research Log

### Entry 1: Existing synthesis background

THOUGHT: I need to avoid duplicating prior reports and identify the remaining gap: actual wording and prompt/codebook architecture.
ACTION: Read[`.agents/docs/20260605-literature-synthesis-map.md`, `.agents/docs/20260605-literature-religious-identity-prompts.md`, `.agents/docs/20260605-literature-religious-nonprofit-classification.md`, `.agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md`]
OBSERVATION:

- Source: local project reports
- Citation: Existing literature-seeker reports (20260605)
- Date: 2026-06-05
- Key Facts: Prior synthesis recommends multi-label religious mission dimensions; identifies Sider & Unruh, Bielefeld & Cleveland, Smith & Sosin, Ebaugh/Chafetz/Pipes, Ma, Fyall et al., Santamarina et al., Pew/GSS/WVS/NCS/ARDA, and LLM-as-annotator/weak-supervision papers as read-first sources.
- Confidence: High because these reports include DOI/source links and align with requested anchors.
- New Questions: Need fresh verification and additional exact wording from official/current sources and recent LLM/CSS prompt papers.

### Entry 2: Official survey and congregation codebook wording

THOUGHT: I need exact prompt-like wording for religious identity, nonreligion, denomination, practice, and salience.
ACTION: Search[`2026 Pew GSS WVS National Congregations Study religion question wording codebook religious affiliation denomination religious person attendance prayer`; `2026 WVS EVS ARDA Measurement Wizard religion survey question wording religious person spiritual person God importance affiliation denomination`]
OBSERVATION:

- Source: GSS/NORC 2024 codebook; Pew 2025 questionnaire; NCS cumulative codebook; WVS Wave 8 questionnaire; ARDA Measurement Wizard.
- Citation: NORC (2024); Pew Research Center (2025); Chaves et al./NCS (2020); World Values Survey Association (2024); Bradburn et al. (2014).
- Date: searched 2026-06-05
- Key Facts: Pew uses “present religion, if any” and explicit atheist/agnostic/nothing categories; GSS uses “religious preference”; NCS asks whether congregation is formally affiliated and requests denomination/association names; WVS separates religion importance, God importance, attendance/prayer excluding weddings/funerals, religious person/not religious person/atheist, and denomination belonging.
- Confidence: High; official codebooks/questionnaires and peer-reviewed ARDA measurement article.
- New Questions: Need organization-level classification criteria.

### Entry 3: Faith-based organization typologies and organizational indicators

THOUGHT: I need authoritative organizational criteria that separate religious identity, authority, culture, resources, and program content.
ACTION: Search[`2026 Sider Unruh faith-based organization typology religious characteristics mission affiliation staff support program content indicators Bielefeld Cleveland Smith Sosin Ebaugh Chafetz Pipes`]
OBSERVATION:

- Source: SAGE/Wiley journal pages; Semantic Scholar/PDF excerpts; IU ORE coding report; Faith and Organizations Project.
- Citation: Sider & Unruh (2004); Bielefeld & Cleveland (2013); Smith & Sosin (2001/2002); Becker/Ebaugh/Chafetz/Pipes (2003/2006); Schneider et al. reports.
- Date: searched 2026-06-05
- Key Facts: Sider & Unruh define a sixfold faith-permeated to secular typology and insist organization and program religiosity can differ. Smith & Sosin use coupling to faith via resources, authority, and culture. Ebaugh/Chafetz/Pipes use visible symbols, religious mission/materials, prayer/religious materials, proselytizing, staffing, and staff understandings of religious purpose. ORE codebooks define identity as name/logo/tagline/mission/history/statement of belief and activity as program goals/products/services/funding.
- Confidence: High for peer-reviewed anchors; medium-high for report/codebook excerpts.
- New Questions: Need direct mission-text ML precedent and LLM prompt wording.

### Entry 4: Nonprofit mission text classifiers, NTEE/IRS, direct religious-org model

THOUGHT: I need closest classification setups using mission/activity text, administrative weak labels, and any explicit religious/nonreligious model cards.
ACTION: Search[`2026 nonprofit mission statement classification NTEE religious purpose IRS Form 990 Ma Fyall Santamarina Lecy codebook classifiers religious organization model mission activity text`]; Fetch[`https://huggingface.co/GivingTuesday/religious_org_v1`; `https://huggingface.co/datasets/GivingTuesday/religious_orgs_training`]
OBSERVATION:

- Source: Nonprofit Mission Classifiers; Ma NVSQ article/GitHub; Fyall et al. SAGE; IRS BMF documentation; NCCS NTEE docs; GivingTuesday Hugging Face model/dataset cards.
- Citation: Ma (2021); Fyall, Moore & Gugerty (2018); Santamarina, Lecy & van Holm (2023); NCCS/Urban (2026); IRS (2025/2026); GivingTuesday (2025/2026).
- Date: searched/fetched 2026-06-05
- Key Facts: Nonprofit Mission Classifiers use name, mission, program service text to predict NTEE and IRS 1023-EZ purpose codes; IRS activity codes include church/synagogue, religious order, mission, missionary activities, evangelism, religious publishing; NCCS warns NTEE is descriptive, single-code, and incomplete. GivingTuesday uses GPT-4 labels and BERT on name/mission/activity; definition: religious organizations have identity and mission derived from religious/spiritual tradition; prompt intent: find mentions/wording/terminology revealing affiliations. Dataset includes explicit positive examples (Bible studies, Gospel, Christian worldview, Catholic tradition, Jewish values) and noisy boundary examples (faith-based coalition context, church in school/hospital names).
- Confidence: High for official docs and peer-reviewed papers; medium-high for model card due incomplete full prompt/spec access.
- New Questions: Need LLM annotation/codebook prompt standards.

### Entry 5: LLM codebook/prompt methodology

THOUGHT: I need current 2020-2026 guidance on how to turn codebooks into LLM prompts and how to validate prompt sensitivity.
ACTION: Search[`2026 LLM as annotator computational social science prompt wording codebook text annotation Gilardi Pangakis Ziems Heseltine Törnberg Mellon codebook prompt`]
OBSERVATION:

- Source: Political Analysis/Cambridge; arXiv; Springer JCOSS; Sociologica/PDF; ICWSM; ACL Anthology.
- Citation: Halterman & Keith/Codebook LLMs (2024/2026); Törnberg (2024); Atreja et al. (2025); Ziems et al. (2024); Gilardi et al. (2023); Pangakis et al. (2023/2024); Smith et al. (2024).
- Date: searched 2026-06-05
- Key Facts: Codebook LLMs recommend semi-structured codebooks with label, definition, clarifications, negative clarifications, positive/negative examples, task instructions, and output reminder. Törnberg recommends prompt codebooks with context, question, constraints, enumerated options, uncertain option, JSON output, model/temperature logging. Prompt selection papers show label distributions and accuracy vary across semantically similar prompts.
- Confidence: High for peer-reviewed/accepted sources; medium for arXiv/preprint frontier work.
- New Questions: Need final synthesis and machine-readable handoff blocks.

### Entry 6: Authoritative definitions of religion/religious organization/nonreligion

THOUGHT: I need definitional anchors that prevent the classifier from smuggling in only Christian or overly functional definitions.
ACTION: Search[`2026 authoritative definitions religion religious organization religious identity spirituality nonreligion secular organization faith community sociology of religion nonprofit studies`]
OBSERVATION:

- Source: Stanford Encyclopedia of Philosophy; Springer Encyclopedia of Global Religion; OSCE/ODIHR guidelines; Pew 2025 RLS; World Religion Database glossary; nonreligion/secular studies book excerpt.
- Citation: Schilbrack/SEP; Berger/Springer entry; OSCE/ODIHR (2014); Pew Research Center (2025); Zuckerman/Galen/Pasquale sources.
- Date: searched 2026-06-05
- Key Facts: Religion definitions vary across substantive, functional, mixed, polythetic approaches. Religious organizations can be defined as identity/mission derived from religious/spiritual tradition and voluntary nonprofit status. Legal standards construe religion/belief broadly and emphasize self-definition. Pew defines religious nones as atheist, agnostic, or nothing in particular. Secular can mean not religious, church-state separation, or a positive worldview such as humanism/atheism/freethought.
- Confidence: High for SEP/official/Pew; medium for broad glossary/book excerpts.

## Verification and Synthesis Notes

- Cross-referenced seed sources across existing local reports and fresh web results.
- Working papers/model cards were checked against official docs or peer-reviewed anchors where possible.
- Direct mission-text religious LLM prompt evidence is thin; GivingTuesday is the closest direct source but does not expose a full GPT-4 prompt in the public model card.
- Final recommendation: use a multi-domain codebook with explicit evidence spans and an ambiguous/review class; do not derive a binary label from single ambiguous tokens.

## Draft Output

### Papers Found

See `.agents/docs/20260605-literature-religious-vs-nonreligious-mission-prompts.md`.

### Coverage Assessment

Searched official survey/codebook sites, journal pages, nonprofit classifier docs, IRS/NCCS docs, Hugging Face, ACL/Cambridge/Springer/SAGE/Wiley, and definitional reference sources. Date range 1912 theoretical roots through 2026 docs/method papers.

### Citation Chains

Seed chains: Sider & Unruh -> Bielefeld/Cleveland, Ebaugh/Chafetz/Pipes, Smith/Sosin; Ma/Fyall/Santamarina -> nonprofit mission classifiers/NTEE docs; Pew/GSS/WVS/NCS/ARDA -> survey wording and measurement caveats; Gilardi/Pangakis/Ziems/Smith et al. -> LLM prompt/weak supervision design.

Quality gate: Required output blocks included: Papers, Coverage Notes, Handoff Citation List, Handoff Datasets Mentioned. Custom requested blocks included: Search Strategy, prompt-like wording extraction, prompt/codebook architecture, definitions, pitfalls, read-first/direct implications.
