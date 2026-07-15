# Replication Scratchpad: Weak Supervision Noisy Labels and LLM Annotation

Created: 20260605
Topic Keywords: LLM annotation; weak supervision; noisy labels; transformer fine-tuning; reproducibility artifacts

---

## Problem Definition

### Research Questions

- Primary: Where are replication packages, code repositories, datasets, prompts, benchmarks, and reproducibility materials for priority LLM-as-annotator, weak-supervision, noisy-label, and transformer fine-tuning papers?
- Secondary: Which artifacts are most reusable for a binary nonprofit mission-text classifier using LLM-generated and weak labels?

### Input Mode

- [x] DOI/URL/citations from literature-seeker
- [x] Freeform paper citations
- [ ] Topic string (bounded bootstrapping)

### Success Criteria

- [x] Use upstream literature reports where available
- [ ] Search Dataverse, OpenICPSR, OSF, Zenodo, and GitHub for priority papers
- [ ] Inspect open package contents where feasible
- [ ] Record data-only/code-only/prompt-only/restricted/author-hosted status

## Research Log

### Entry 1: Upstream scope

THOUGHT: I need to bound the search using the existing literature report and user priority list.
ACTION: Read `docs/research/20260605-literature-llm-weak-supervision-noisy-labels.md` and synthesis map.
OBSERVATION:

- Source: local literature reports
- Repository: local project reports
- Key Facts: Citation handoff includes Gilardi, Ziems, Pangakis, Heseltine, Ratner, Snorkel, Smith, Zhu, SaFER, LAFT, Lu & Smith, Ye et al.; user adds Törnberg and benchmark/prompt/artifact sources.
- Confidence: High for bounded scope.
- New Questions: Need repository/package verification and manifests.

## Draft Output

### Packages Found

Pending repository searches.

### Repository Coverage

Pending.

### Not Found

Pending.


### Entry 2: Repository search summary

THOUGHT: I needed to search all target repositories before declaring packages missing.
ACTION: Queried Dataverse, OSF, Zenodo APIs; OpenICPSR site searches; GitHub APIs/fallback pages; ACL/OpenReview/PapersWithCode-style web searches.
OBSERVATION:

- Verified Dataverse packages: Gilardi (62 files), Heseltine (8 files).
- Verified GitHub/code artifacts: gpt_annotate, Snorkel, WRENCH, BERT-LNL, transfer-distant-transformer-african, SiDyP; LAFT repo is effectively empty.
- Not found/unverified: Törnberg package, SaFER official code, Lu & Smith code/data, Smith et al standalone repo, exact Ziems GitHub data directory.
- Confidence: Medium-High; GitHub API rate limits required webfetch fallback for some manifests.
