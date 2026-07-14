# Replication Scratchpad: Imbalanced Text Validation

Created: 20260605
Topic Keywords: imbalanced text classification; validation metrics; social measurement; weak supervision; reproducibility artifacts

---

## Problem Definition

### Research Questions

- Primary: Where are replication packages, code, datasets, notebooks, benchmark scripts, validation checklists, and reproducibility materials for priority papers/artifacts on imbalanced binary text classification evaluation and validation designs?
- Secondary: Which artifacts are directly reusable for this project’s evaluation module and reporting checklist?

### Input Mode

- [x] DOI/URL/citations from literature-seeker
- [x] Freeform paper citations
- [ ] Topic string (bounded bootstrapping)

### Success Criteria

- [x] Use upstream literature reports as bounded scope where available.
- [ ] Search Dataverse, OpenICPSR, OSF, Zenodo, and GitHub for each priority paper/artifact before marking missing.
- [ ] Inspect package contents where open repository APIs expose manifests.
- [ ] Record search evidence, access restrictions, last-updated signals, and usefulness for the project.

## Research Log

### Entry 1: Upstream scope

THOUGHT: I need to bound the replication search by existing literature reports and the user’s priority list.
ACTION: Read `docs/research/20260605-literature-imbalanced-text-validation.md` and `docs/research/20260605-literature-synthesis-map.md`.
OBSERVATION:

- Source: `docs/research/20260605-literature-imbalanced-text-validation.md`, handoff lines 92-125.
- Repository: Local project report.
- Key Facts: Citation scope includes Davis & Goadrich; Saito & Rehmsmeier; Fawcett; Drummond & Holte; Hand; Hernández-Orallo et al.; Brodersen et al.; Chicco et al.; Hopkins & King; Grimmer & Stewart; Gentzkow et al.; Keith & O'Connor; Nelson et al.; ValiText; PSRM 2025; Gilardi et al.; BOXWRENCH/WRENCH; AlleNoise.
- Confidence: High; report is complete and topic-matched.
- New Questions: Need repository/package availability and reusable file manifests.

### Entry 2: Repository-wide API search

THOUGHT: I need to check Dataverse, Zenodo, and OSF for each priority paper/artifact before declaring missing.
ACTION: Queried APIs with exact-title/author-key phrases for Davis/Goadrich; Saito/Rehmsmeier; Fawcett; Drummond/Holte; Hand; Hernandez-Orallo/Flach/Ferri; Brodersen; Chicco/Warrens/Jurman; Hopkins/King; Grimmer/Stewart; Gentzkow/Kelly/Taddy; Nelson; ValiText; PSRM trustworthy SML; BOXWRENCH; WRENCH; AlleNoise.
OBSERVATION:

- Source: Dataverse API, Zenodo API, OSF API.
- Repository: Dataverse/Zenodo/OSF.
- Key Facts: Broad API hits were mostly irrelevant except later exact DOI verification. OSF exact title pass returned 0 for all priority queries. Zenodo exact API pass discovered no priority metric-paper deposits through search; AlleNoise Zenodo record was found by targeted web search and inspected separately.
- Confidence: Medium-high; API search can be noisy, but DOI and GitHub targeted searches verified actual artifacts.
- New Questions: Need targeted web and GitHub inspection for author-hosted/software artifacts.

### Entry 3: Verified Dataverse packages

THOUGHT: Social-science text validation papers are likely to have Dataverse replication datasets.
ACTION: Inspected Dataverse manifests for `doi:10.7910/DVN/NV0SZJ`, `doi:10.7910/DVN/FQBHP8`, `doi:10.7910/DVN/AFBW80`, and `doi:10.7910/DVN/PQYF6M`.
OBSERVATION:

- Source: Harvard Dataverse API.
- Repository: Dataverse.
- Key Facts: Hopkins & King has 28 files including R scripts, coding instructions, text corpora, and `readme.txt`; Grimmer & Stewart has `ReplicationFile.zip`; Park & Montgomery PSRM 2025 has `ParkMontgomery_ReplicationData.tar.gz` (518MB); Gilardi et al. has 62 files including R scripts, Python ChatGPT template, annotated data, batch results, and plots.
- Confidence: High; DOI-backed manifests inspected.
- New Questions: Dataverse ZIP/TAR internals were not expanded; only API manifests recorded.

### Entry 4: Verified GitHub/software artifacts

THOUGHT: Metric and benchmark papers often release software on GitHub or author pages rather than archival replication repositories.
ACTION: Inspected GitHub manifests for ReadMeV1, ValiTex, stronger-than-you-think/BOXWRENCH, WRENCH, AlleNoise, Future of Coding, hmeasure, precrec, pr2roc, AUCCalculator adaptation, and Brodersen micp.
OBSERVATION:

- Source: GitHub Contents API.
- Repository: GitHub.
- Key Facts: Verified root and key subdirectory contents; recorded pushed dates. Key reusable files include ValiTex `R/app_ui.R`, `data/framework.rda`; WRENCH `wrench/evaluation.py`, examples; BOXWRENCH `end_model_training/pipelines.py`, `val_size_experiment.py`; AlleNoise `allenoise/full_dataset.csv`, `category_classifier/prepare_dataset.py`; hmeasure `R/library_metrics.R`; precrec `R/main_evalmod.R`; Future of Coding data and notebooks/scripts.
- Confidence: High for repository existence and manifests; medium for paper-specific replication claims where repo is a later package rather than the exact paper archive.

### Entry 5: OpenICPSR and author fallback

THOUGHT: Before marking missing, I need to check OpenICPSR and author pages for non-Dataverse social-science packages and older metric software.
ACTION: Searched `site:openicpsr.org` with exact title/author queries and fetched author pages for Hopkins-King/ReadMe and cost-curve software.
OBSERVATION:

- Source: OpenICPSR web search; Gary King site; Robert Holte cost-curve page.
- Repository: OpenICPSR / author-hosted.
- Key Facts: No exact OpenICPSR packages found for priority papers; results were unrelated text/ML projects. Hopkins-King author page points to current Harvard Dataverse DOI `10.7910/DVN/NV0SZJ` and ReadMe GitHub. Drummond-Holte has an author-hosted Cost Curves page; no modern DOI archive/package manifest found.
- Confidence: Medium-high; OpenICPSR search is web-only and can miss JS-indexed content, but exact title queries found no relevant records.

## Draft Output

### Packages Found

See `docs/research/20260605-replication-imbalanced-validation.md`.

### Repository Coverage

Dataverse, OpenICPSR, OSF, Zenodo, GitHub searched; key DOI/GitHub manifests inspected.

### Not Found

Paper-specific packages not found for Fawcett (2006), Hand (2009), Hernandez-Orallo et al. (2012), Chicco/Warrens/Jurman (2021), Gentzkow/Kelly/Taddy (2019 survey), and Saito/Rehmsmeier (2015 PLOS paper; later precrec package exists).
