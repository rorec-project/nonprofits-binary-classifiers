# Replication Scratchpad: Calibration and Prevalence Replication Materials

Created: 20260605
Topic Keywords: classifier calibration; prevalence quantification; transformer calibration; social measurement; misclassification correction

---

## Problem Definition

### Research Questions

- Primary: Where are replication packages, maintained libraries, datasets, notebooks, benchmarks, and reproducibility materials for calibration, quantification/prevalence, and population-scale social measurement papers?
- Secondary: Which artifacts are reusable for this repository: calibration curves/ECE/Brier/log-loss, PCC/ACC/PACC, bootstrap prevalence CIs, PPI, and misclassification correction?

### Input Mode

- [x] DOI/URL/citations from literature-seeker
- [x] Freeform paper citations
- [x] Topic string (bounded bootstrapping)

### Success Criteria

- [x] Use local literature reports as scope context
- [ ] Search Dataverse, OpenICPSR, OSF, Zenodo, GitHub / artifact pages for priority papers and reusable maintained packages
- [ ] Inspect open package contents where possible
- [ ] Save final report under `docs/research/` with required handoff blocks

## Research Log


### Entry 1: Archive/API sweep

THOUGHT: I needed repository-complete coverage for priority calibration, quantification, prevalence, and inference artifacts.
ACTION: Queried Harvard Dataverse API, Zenodo API, OSF API, OpenICPSR web search, GitHub code search, GitHub Contents API, and fallback webfetch for rate-limited GitHub.
OBSERVATION:
- Dataverse exact priority searches returned no matches for classic calibration/quantification papers; Hopkins-King replication is documented via legacy hdl:1902.1/12898 rather than a modern Dataverse DOI.
- Zenodo contains LeQua 2022/2024, PPI datasets, ppi_py archive, PyCalib, and Re-Assessing Classify-and-Count.
- OSF exact title searches returned no relevant public nodes.
- OpenICPSR searches returned no direct priority matches; general ML/economics replication hits were irrelevant to the requested papers.
- GitHub contains maintained/research code: gpleiss temperature_scaling, focal_calibration, netcal, QuaPy, quantificationlib, doc_prevalence/freq-e, ReadMeV1/readme2, ppi_py, multicalibrated_llm_measurement.

### Entry 2: Package inspection

THOUGHT: Verified packages must include file manifests and access status.
ACTION: Used GitHub Contents API until rate-limited, Zenodo records API, and GitHub webfetch fallback.
OBSERVATION:
- Key file manifests captured in final report.
- GitHub API rate limit affected facebookresearch/multicalibrated_llm_measurement; webfetch of repository README succeeded and provided structure.
- OpenICPSR is web-only; no exact matches found.

## Draft Output

### Packages Found

See final report `docs/research/20260605-replication-calibration-prevalence.md`.

### Repository Coverage

Dataverse, OpenICPSR, OSF, Zenodo, GitHub, PMLR/ACL/NeurIPS/publisher pages, author pages.

### Not Found

Standalone replication archives not found for Platt (1999), Zadrozny & Elkan (2002), Forman (2005/2008), Saerens et al. (2002), Bella et al. (2010), Gonzalez et al. (2017), Silva Filho et al. (2023), Hausman et al. (1998), Meyer & Mittag (2017). Implementations/surveys exist for many methods.
