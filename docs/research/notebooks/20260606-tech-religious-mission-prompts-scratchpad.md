# Tech Scratchpad: Religious vs Nonreligious Mission Prompt Literature

Created: 20260606
Topic Keywords: religious mission prompts; nonprofit classification; faith-based codebooks; survey religion items; weak supervision

---

## Problem Definition

### Research Questions

- Primary: What academic sources, codebooks, survey instruments, dictionaries, and prompt-like annotation instructions should inform implementation of religious vs nonreligious mission-text labels?
- Secondary: Which wording, dimensions, boundary rules, and weak-supervision triggers are current best practice for LLM prompts and codebooks?

### Upstream Inputs

- Replication-seeker output: Pending upstream check
- Mode: standalone / replication-informed pending

### Success Criteria

- [ ] Existing synthesis and related reports inspected.
- [ ] Exact search queries recorded.
- [ ] 12-25 key sources summarized with links/DOIs.
- [ ] Actual prompt-like wording extracted where available.
- [ ] Proposed prompt/codebook architecture written with positive/negative/ambiguous rules and evidence-span requirements.
- [ ] Pitfalls and weak-supervision implications documented.

## Research Log

### Entry 1: Setup

THOUGHT: I need to start from accessible project synthesis and related reports, then fill gaps with targeted web/documentation searches.
ACTION: Inspect existing `docs/research` files and upstream replication outputs.
OBSERVATION:

- Source: local project files
- Version: n/a
- Key Facts: Existing reports and synthesis map exist; same-topic replication output exists at `docs/research/20260605-replication-religious-vs-nonreligious-prompts.md` with handoff block and concrete artifacts.
- Confidence: High — local files read directly.
- New Questions: Which newer/direct artifacts should be elevated in a revised report?

### Entry 2: Targeted web verification

THOUGHT: I need to verify direct prompt/codebook artifacts rather than only relying on literature summaries.
ACTION: Web search and fetch GivingTuesday model/dataset, Codebook LLMs, ICWSM prompt design, ORE/IU codebook, FORFA codebook, Paxton dictionary.
OBSERVATION:

- Source: Hugging Face GivingTuesday `religious_org_v1` and `religious_orgs_training`; Cambridge/Political Analysis Codebook LLMs; ICWSM 2025 Atreja et al.; IU ScholarWorks ORE report; Lipscomb FORFA codebook; Pamela Paxton dictionary page.
- Version: 2025-2026 web artifacts where applicable; classic sources retained from 2001-2018.
- Key Facts:
  - GivingTuesday directly uses GPT-4 labels from nonprofit name, mission, and activities; definition: identity and mission derived from religious/spiritual tradition.
  - Codebook LLMs recommends semi-structured labels with label, definition, clarifications, negative clarifications, examples, instruction, output reminder, and behavioral tests.
  - Atreja et al. show label distributions and compliance vary by prompt design; prompts must be frozen/evaluated.
  - ORE/IU codebook uses conservative default: no known religious expression until evidence; identity vs activity coding; explicit evidence rules.
  - FORFA codebook offers operational codes for secular, secular with religious roots, specific traditions, spiritually ambiguous/indigenous, and multireligious, with concrete lexical triggers and image/logo caveats.
  - Paxton/Ressler/Fulton dictionary has 900 words for identifying nonprofits with religious missions in Form 990/990EZ text.
- Confidence: High for fetched/public artifacts; medium-high for search-result-extracted PDF text where direct text fetch returned PDF binary.
- New Questions: None blocking final report.

### Entry 3: Output synthesis

THOUGHT: The final report should consolidate existing local work plus newly elevated operational codebooks into prompt/codebook architecture.
ACTION: Write revised `docs/research/20260605-literature-religious-vs-nonreligious-mission-prompts.md`.
OBSERVATION:

- Source: Local and web sources above.
- Version: Current as of 2026-06-06.
- Key Facts: Positive class should be multi-domain; binary label derived from explicit purpose/activity/high-confidence identity; ambiguous/historical/spiritual cases should be retained for audit, not collapsed.
- Confidence: High.

## Draft Output

### Packages Found

Not applicable — literature and prompt/codebook architecture search.

### Code Snippets

Not applicable — no software implementation snippets requested.

### Version Notes

Track source years and current relevance; no package version notes expected.
