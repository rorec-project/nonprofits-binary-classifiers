---
created: 20260606
agent: tech-seeker
scratchpad: .agents/notebooks/20260606-tech-religious-nonprofit-mission-scratchpad.md
status: complete
title: Tech - Religious Nonprofit Mission
topic: religious nonprofit mission; NTEE; Form 990; mission classification
---

# Tech: Religious Nonprofit Mission Classification Literature and Measurement

## From Replication Packages

Upstream: `.agents/docs/20260605-replication-religious-nonprofit-mission.md`.

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| Ma (2021) | Python/Jupyter | `API/`, `script/classification_algorithms/`, `reference/assign_NTEE/` | BERT/ML classifiers for NTEE coding | https://github.com/ma-ji/npo_classifier |
| Santamarina et al. (2023) | R/Python/docs | `docs/Preprocessing_Replication.html`, `docs/Classification_Bootstrapping_Replication`, `DATA/` | Quanteda preprocessing + Naive Bayes bootstrapped classifiers | https://github.com/fjsantam/bespoke-npo-taxonomies |
| Nonprofit Mission Classifiers | R | `DATA/MISSION.csv`, docs/vignettes, taxonomy pages | IRS text benchmark for NTEE and purpose-code classification | https://github.com/Nonprofit-Open-Data-Collective/machine_learning_mission_codes |
| IRS 990 e-filer database | R/Python | `BUILD_SCRIPTS/`, concordance files | Convert IRS XML to relational/indexed data | https://github.com/Nonprofit-Open-Data-Collective/irs-990-efiler-database |
| Mission taxonomies | R/HTML/CSV | `NTEE/`, `NTEEV2/`, `PCS/`, IRS activity/purpose codes | Crosswalk/taxonomy support files | https://github.com/Nonprofit-Open-Data-Collective/mission-taxonomies |
| UK-CAT | Python/Jupyter | `src/ukcat`, ICNP/TSO notebooks, `data/ukcat.csv` | Regex tagger + logistic regression classifier | https://github.com/charity-classification/ukcat |
| GivingTuesday religious classifier | Python/Transformers | model card/config/tokenizer/model weights | BERT religious-org segmentation benchmark | https://huggingface.co/GivingTuesday/religious_org_v1 |

## Package Recommendations

This user request is literature/dataset oriented rather than package-selection oriented. Practical tooling/data recommendations:

| Language | Package/Data Source | Version | Key Function | Source |
|----------|---------------------|---------|-------------|--------|
| R/Python | IRS Form 990 e-file XML | 2026 postings available | Mission/program-service text extraction | https://www.irs.gov/charities-non-profits/form-990-series-downloads |
| R/Python | IRS EO BMF | Updated 2026-05 | EIN registry, NTEE, filing status fields | https://www.irs.gov/charities-non-profits/exempt-organizations-business-master-file-extract-eo-bmf |
| R/Python | NCCS NTEE/NTEEV2 | Current NCCS docs | Official and research NTEE labels/crosswalks | https://nccs.urban.org/nccs/resources/ntee/ |
| R | Nonprofit Mission Classifiers | 2019-2023 docs | Mission classifier benchmarks and IRS purpose-code labels | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ |
| Python | UK-CAT | Current GitHub project | Open charity activity/religion tags and ICNP/TSO benchmark | https://charityclassification.org.uk/ |
| API/proprietary | Candid PCS / Cause IQ | PCS Nov. 2024; current Cause IQ docs | External taxonomy/validation labels; proprietary caveats | https://taxonomy.candid.org/ ; https://www.causeiq.com/help/how-to-articles/find-organizations-according-their-missions-and-programs/ |
| Data | NCS/ARDA/ICPSR | Cumulative waves 1998-2019; ICPSR updated 2025 | Congregation benchmark and question wording | https://www.nationalcongregationsstudy.org/data-documentation ; https://www.icpsr.umich.edu/web/ICPSR/studies/3471 |

## Implementation Examples

### Example 1: Label schema implied by literature

**Source**: Sider & Unruh (2004), Bielefeld & Cleveland (2013), NCCS/IRS/NODC docs.
**Package version**: N/A.

```text
Primary labels:
- religious_purpose_explicit
- religious_identity_or_affiliation
- religious_service_content
- religion_related_NTEE_X
- faith_inspired_ambiguous

Derived binary v1:
positive = explicit religious purpose OR religious service content OR high-confidence identity/affiliation;
treat NTEE X and IRS religious purpose as noisy priors/benchmarks, not sole ground truth.
```

**Notes**: Avoid defining the target as latent religiosity. Define it as observable religious mission/expression in the available short text.

### Example 2: Validation/audit design

**Source**: Ma (2021), Fyall et al. (2018), Santamarina et al. (2023), IRS/NCCS/NCS docs.
**Package version**: N/A.

```text
Audit strata:
- random sample from deployment population
- predicted positives/negatives and near-threshold records
- NTEE X vs non-X conflicts
- IRS religious-purpose vs nonreligious-purpose conflicts
- hospitals, schools, international NGOs, foundations, human-service agencies
- names containing St./Saint, church, ministry, mosque, synagogue, temple, Catholic, Jewish, Islamic, Hindu, Buddhist, Sikh
- non-Christian and ambiguous spiritual-language cases
```

**Notes**: Report precision/recall/F1, PR-AUC, calibration, prevalence error, and subgroup error audits; do not rely on accuracy alone.

## Version and Compatibility Notes

- IRS Form 990 XML and EO BMF are current official sources with 2026 update/download pages, but Form 990 excludes many churches and some church-affiliated organizations from annual filing.
- NCCS distinguishes official `NTEE_IRS` from research-updated `NTEE_NCCS`; treat both as noisy labels with documented provenance.
- Candid PCS was updated in November 2024 and includes facets beyond NTEE, including organization type and religious/governmental auspice; access/licensing may limit reproducibility.
- Cause IQ reports manual/algorithmic classification enhancements and multiple NTEE codes, useful for validation if licensed, but proprietary labels should not be treated as transparent gold standard.
- UK-CAT is open and useful for stress-testing regex/logistic classification logic, but UK charity law/text fields differ from U.S. IRS/NCCS contexts.
