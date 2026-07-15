# Literature Scratchpad: Religious Identity Prompts and Textual Triggers

Created: 2026-06-05
Topic Keywords: religious identity prompts; survey wording; faith-based organizations; NLP lexicons; nonprofit mission text

---

## Problem Definition

### Research Questions

- Primary: Which academic papers, official survey instruments, codebooks, and datasets provide wording, lexical cues, codebook criteria, or classifier features for identifying religious identity, affiliation, religiosity, faith-based organizations, religious nonprofits, congregations, institutions, and related groups from text or questionnaire responses?
- Secondary: What concrete survey questions, prompt phrasings, annotation rules, weak-supervision triggers, and caveats can inform LLM prompts and human annotation instructions for classifying religious mission or identity from nonprofit/organizational text?

### Context & Constraints

- Timeline/Budget: Single-session literature search; prioritize peer-reviewed and official instrument/codebook sources; include classics and recent 2020-2026 sources; breadth across sociology of religion, survey methodology, computational social science/NLP, nonprofit studies, communication, theology/religious studies, public policy, and applied economics.
- Authoritative sources: Official instruments/codebooks (GSS, Pew, WVS/EVS, NCS, ARDA, IRS/NTEE/NCCS where relevant), peer-reviewed journal articles, university/established working papers, and high-value reports containing explicit coding or question wording.

### Success Criteria

- [ ] Identify 12-25 high-priority sources with stable links/DOIs where available.
- [ ] Extract concrete wording/triggers from official instruments or literature, distinguishing affiliation, identity, practice, belief, salience, mission, auspice, denomination, and faith-inspired ambiguity.
- [ ] Cover survey wording, organizational/nonprofit coding, computational text classification/lexicons, and applied policy/economics uses.
- [ ] Record exact search queries and coverage gaps.
- [ ] Produce machine-readable citation and dataset handoff tables.

## Research Log

### Entry 1: Setup

THOUGHT: I need to define topic keys and authoritative source classes before searching.
ACTION: Initialized scratchpad with topic keywords and problem definition.
OBSERVATION:

- Source: Local templates and user request.
- Citation: None.
- Date: 2026-06-05.
- Key Facts: Search must extract practical wording/triggers and caveats for religious mission classification.
- Confidence: High (directly from user scope and project workflow).
- New Questions: Need official question wordings and organizational coding schemes.

### Entry 2: Official survey and codebook wording

THOUGHT: Need authoritative wording for affiliation, denomination, born-again identity, practice, salience, and congregational classification.
ACTION: Search[`2026 GSS religion items codebook religious preference denomination born again question wording`; `Pew Research Center religion survey question wording religious affiliation born again evangelical 2026 codebook`; `World Values Survey European Values Study religion question wording denomination religious person spirituality 2026 questionnaire`; `National Congregations Study codebook congregation religious tradition denomination question wording 2026`]
OBSERVATION:

- Source: GSS/NORC codebooks; Pew methodology/questionnaires; WVS questionnaire; NCS cumulative codebook/ARDA.
- Citation: NORC (2024); Pew Research Center (2018, 2021, 2025); WVS (2024); Chaves et al. / NCS (2020).
- Date: 2026-06-05 search.
- Key Facts: Pew “What is your present religion, if any?” and born-again wording; GSS RELIG/DENOM/ATTEND/PRAY/REBORN; WVS “religious person,” God importance, prayer, service attendance, denomination; NCS formal denomination/association and DENCODE/TRAD3.
- Confidence: High (official instruments/codebooks).
- New Questions: Need organizational FBO typologies and computational triggers.

### Entry 3: Organizational/nonprofit FBO typologies

THOUGHT: Need how nonprofit studies define faith-based organizations and what observable indicators map to text labels.
ACTION: Search[`Sider Unruh faith-based organization typology religious characteristics indicators mission staff funding program 2004`; `Bielefeld Cleveland defining faith-based organizations nonprofit studies classification religious identity mission statements`; `Smith Sosin varieties of faith-related agencies faith-based organizations dimensions religiosity auspices nonprofit`; `faith-based nonprofit organizations mission statement religious identity classification codebook textual indicators Form 990`; `Ebaugh Chafetz Pipes faith-based social service coalitions religious symbols mission statement prayer service staff organizational religiosity 2006 DOI`]
OBSERVATION:

- Source: SAGE/Wiley journal pages; university repositories; CMACS publications page; Hugging Face/Nonprofit Open Data leads.
- Citation: Sider & Unruh (2004); Smith & Sosin (2001); Bielefeld & Cleveland (2013); Ebaugh et al. (2003, 2005, 2006).
- Date: 2026-06-05 search.
- Key Facts: Strong indicators include mission/name/history/imagery, formal affiliation, board/staff faith requirements, religious funding/support, religious practices, prayer/scripture/proselytizing, and program integration. Important distinction between organization and program religiosity.
- Confidence: High for peer-reviewed sources; medium for repository codebooks.
- New Questions: Need NLP/weak-supervision examples and pitfalls.

### Entry 4: Computational text classification and lexicons

THOUGHT: Need computational social science/NLP methods that use keywords, lexicons, annotation schemes, classifiers, or prompts for religious identity from text.
ACTION: Search[`computational social science NLP religious identity text classification keywords lexicon annotation religion 2020 2026`; `religion lexicon dictionary social media NLP religious identity affiliation classification paper`; `nonprofit mission classification machine learning NTEE religion related text Form 990 mission statements paper`; `weak supervision dictionary religion text classification religious organization mission statement codebook lexical triggers`]
OBSERVATION:

- Source: ACL Anthology; PeerJ/PMC; Political Analysis; NVSQ; Nonprofit Open Data Collective; GivingTuesday model docs.
- Citation: Chen et al. (2014); Ma (2021); Fyall et al. (2018); Manerba et al. (2023); Sircar et al. (2023); Sachs et al. (2026); Morlan et al. (2025); GivingTuesday (2026).
- Date: 2026-06-05 search.
- Key Facts: Dictionaries are transparent but brittle; classifiers improve coverage; named-denomination and practice terms give high precision; recent religiolect work highlights beyond-keyword shibboleths and cross-register/cultural bias.
- Confidence: High for peer-reviewed sources; medium for model docs/preprints.
- New Questions: Need measurement caveats for wording and construct validity.

### Entry 5: Survey-methodology caveats

THOUGHT: Need pitfalls around social desirability, nonresponse, one-step vs two-step questions, nones, spirituality vs religion, and cross-national comparability.
ACTION: Search[`survey methodology religion question wording religious affiliation preference identity social desirability attendance overreporting`; `religion spirituality survey measurement question wording religious nones affiliation salience construct validity`; `cross-national religion survey question wording comparability religious affiliation WVS EVS ISSP Pew pitfalls`; `Measuring religion survey questions affiliation belief behavior belonging codebook National Academy social science religion measurement`]
OBSERVATION:

- Source: Survey Practice; Pew methods; GSS methodological reports; JSSR; Methodological Innovations; GESIS/EVS/ISSP documentation.
- Citation: Smith & Kim (2007); Bradburn et al. (2014); Pickel et al. (2016); Pew (2021); Hackett & Conrad (2026); Burge (2020).
- Date: 2026-06-05 search.
- Key Facts: “If any” and explicit no-religion options reduce presumptive wording; mode and response order change estimates; affiliation, practice, belief, salience, and spirituality are separate constructs; Western/Christian assumptions bias cross-national measurement.
- Confidence: High.
- New Questions: None critical.

## Draft Output

### Papers Found

See final report: `docs/research/20260605-literature-religious-identity-prompts.md`.

### Coverage Assessment

Covered official survey/codebook documentation, sociology/nonprofit FBO typologies, computational text/NLP, administrative nonprofit mission classification, and survey-methodology caveats. Gaps remain in public, peer-reviewed, non-Christian organization-level religious-expression lexicons and cross-national nonprofit classifiers.

### Citation Chains

Seed chain traced: Sider & Unruh -> Bielefeld & Cleveland review -> Ebaugh/Chafetz/Pipes measures -> recent organizational identity/communication work; Smith & Sosin -> faith-related agency coupling; Pew/GSS/WVS/NCS -> question wording and codebook variants; nonprofit mission coding -> NTEE/IRS 990/1023-EZ classifiers; social media/NLP -> religion identity, religiolects, religious hate, name-inference.
