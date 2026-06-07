---
created: 20260605
agent: replication-seeker
scratchpad: .agents/notebooks/20260605-replication-religious-nonprofit-mission-scratchpad.md
status: complete
title: Replication - Religious Nonprofit Mission Classification
topic: religious nonprofits; mission classification; nonprofit taxonomies; Form 990; faith-based organizations
---

# Replication: Religious Nonprofit Mission Classification

## Search Strategy and Exact Queries Used

Scope was bounded by the local literature reports:

- `.agents/docs/20260605-literature-synthesis-map.md`
- `.agents/docs/20260605-literature-religious-nonprofit-classification.md`
- `.agents/docs/20260605-literature-religious-identity-prompts.md`

Repository/API queries used:

1. Dataverse API: `/api/v1/search?q={query}&type=dataset&per_page=10` for:
   - `Automated Coding Using Machine Learning Remapping U.S. Nonprofit Sector`
   - `Beyond NTEE Codes nonprofit activity mission statement content coding`
   - `Methods Classifying Nonprofit Organizations semi-automated text Litofcenko Karner Maier`
   - `How to Code a Million Missions bespoke nonprofit activity codes`
   - `Nonprofit Mission Classifiers machine learning mission codes`
   - `Angel enterprise search nonprofit industry Haq`
   - `UK Charity Classification UK-CAT ICNP TSO`
   - `National Congregations Study cumulative`
   - `Typology Religious Characteristics Social Service Educational Organizations Programs Sider Unruh`
   - `Defining Faith-Based Organizations Bielefeld Cleveland`
2. Dataverse file-list API:
   - `doi:10.7910/DVN/BL6XLW`
   - `doi:10.7910/DVN/EO2HIM`
   - `doi:10.7910/DVN/4GZJSK`
3. Zenodo API: fielded title queries for the same priority titles; no target package matches.
4. OSF API: `filter[title]` for the same priority titles; no target package matches.
5. OpenICPSR web queries:
   - `site:openicpsr.org ("Automated Coding Using Machine Learning" OR "Beyond NTEE Codes" OR "How to Code a Million Missions" OR "Litofcenko" OR "Sider Unruh" OR "Bielefeld Cleveland" OR "Bespoke NPO Taxonomies")`
   - `site:openicpsr.org OR site:icpsr.umich.edu "Form 990 Mission Glossary" "Mission Stemmer" Paxton Velasco Ressler`
6. Web/GitHub queries:
   - `"Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector" Ji Ma data code`
   - `"Beyond NTEE Codes" "replication" "GitHub" "data" Fyall Gugerty Moore`
   - `"Replication Data for: Bespoke NPO Taxonomies" Harvard Dataverse DOI`
   - `"Angel" "enterprise search" "nonprofit" "Haq" "GitHub" "model"`
   - `"Litofcenko" "Karner" "Maier" nonprofit classification replication data code GitHub`
   - `NCCS NTEE codes Urban Institute documentation religion related X codebook`

## Verified Packages

| Paper | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|-------|-----------|-----|----------------|---------------|-------------|------------|
| Ma (2021), *Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector* | GitHub | https://github.com/ma-ji/npo_classifier | Jupyter Notebook/Python | UCF train/test, classifier outputs, NTEE assignment references, remapped sector link | Open; GitHub API rate-limited during file inspection | Top-level 5 dirs + README |
| Santamarina, Lecy & van Holm (2023), *How to Code a Million Missions* | Harvard Dataverse + GitHub | https://fjsantam.github.io/bespoke-npo-taxonomies/ | R, Python/docs | IRS 1023-EZ raw/cleaned, preprocessed text corpora, classifier RData/RDS outputs | Open | 12 + 3 + 70 Dataverse files |
| Nonprofit Open Data Collective, mission classifiers | GitHub Pages + GitHub | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | R | `MISSION.csv`, `MISSION.rds`, NTEE/purpose labels, IRS e-file/1023-EZ sources | Open; docs incomplete in places | Repo top-level DATA/docs/README |
| UK Charity Classification / UK-CAT | GitHub + project docs | https://github.com/charity-classification/ukcat | Python, Jupyter | UK-CAT schema, ICNP/TSO schema, manual labels, all-charity classifications, ML model workflow | Open, CC BY 4.0 | Repo data/docs/notebooks/src + 1 release |
| National Congregations Study cumulative file | ICPSR / NCS / ARDA | https://doi.org/10.3886/ICPSR03471.v6 | Codebooks, setups (ICPSR generated) | 5,333 congregation survey records, questionnaires, codebook, weights; some restricted geography/linked files | Public via ICPSR; restricted components via NORC/ICPSR | ICPSR datasets + codebooks; web-only listing |
| GivingTuesday religious organization classifier | Hugging Face | https://huggingface.co/GivingTuesday/religious_org_v1 | Python/Transformers | BERT model, tokenizer/config, linked training dataset | Open, Apache-2.0 model; internal notebooks partly inaccessible | HF model files; dataset 498 viewer rows shown |
| ANGEL enterprise search for nonprofit industry | GitHub + ACL Anthology | https://github.com/saifulhaq95/ANGEL | None released | Paper PDF only; search/evaluation data described but not released | Open placeholder; code not released | 2 files: README, PDF |

## Public Datasets and Official Administrative Data

| Artifact | Stable URL/DOI | Contents | Access Type | Update Signal | Variables/Text Fields Relevant to Religious Mission Classification | Reproducibility Caveats |
|---|---|---|---|---|---|---|
| IRS Form 990 e-file XML downloads | https://www.irs.gov/charities-non-profits/form-990-series-downloads | Annual/monthly XML ZIPs and index CSVs | Official public admin data | Page reviewed/updated 20-May-2026; 2026 files present | Organization name; Form 990 Part I Line 1 mission; Form 990 Part III program service accomplishments; 990-EZ mission/program text | Churches and some church-affiliated orgs are filing-exempt; XML schemas vary by year/form |
| IRS Business Master File / Exempt Organization metadata | https://www.irs.gov/pub/irs-soi/eo-info.pdf | EO/BMF metadata and public SOI context | Official public admin data | IRS/SOI maintained | EIN, name, subsection, NTEE where present, ruling/date/status fields | NTEE missing/incorrect for some orgs; BMF is registry metadata, not rich text |
| IRS 1023-EZ approvals | https://www.irs.gov/charities-non-profits/exempt-organizations-form-1023-ez-approvals | Exemption application approval files | Official public admin data | IRS year files | `MISSION`, `Nteecode`, binary purpose fields including religious purpose in Santamarina/NODC use | Only 1023-EZ applicants; variable availability differs by year |
| NCS cumulative file | https://doi.org/10.3886/ICPSR03471.v6 | Survey of U.S. congregations, 1998/2006-07/2012/2018-19 | Public via ICPSR membership/researcher access; some restricted | Version V6, 2025-06-03 | Denomination/formal affiliation, worship, programs, religious education, social services, weights | Congregations only, not nonprofit registry; access may require ICPSR login |
| NCS official documentation | https://www.nationalcongregationsstudy.org/data-documentation | Codebook, questionnaires, weights docs, ARDA links | Public docs | Active NCS website | Exact wording for congregation affiliation/activity/program variables | Data download routed to ARDA/ICPSR; sensitive files require NORC process |
| NaNDA civic/social/religious organizations | https://doi.org/10.3886/E207966V1 | Counts/densities of churches, mosques, synagogues, third places by tract/ZCTA, 1990-2021 | OpenICPSR web-only/public | 2024 deposit | Religious organization categories from establishment data | Aggregate geography, not mission text; NETS/Dun & Bradstreet provenance |

## Restricted or Proprietary Datasets / Data Products

| Artifact | Stable URL/DOI | Contents | Access Type | Update Signal | Relevant Fields | Reproducibility Caveats |
|---|---|---|---|---|---|---|
| Candid/GuideStar data products and PCS coding | https://taxonomy.candid.org/ | PCS taxonomy; organization/grant coding in Candid products | Taxonomy open CC BY 4.0; underlying GuideStar/Candid data often licensed/proprietary | Current PCS site | Subject, population, organization type, support strategy, transaction type; religion-related subjects/org types | Taxonomy reusable, but coded organization-level data may require subscription/licensing |
| Cause IQ classifications | https://www.causeiq.com/help/how-to-articles/find-organizations-according-their-missions-and-programs/ | Proprietary Types/Issues/Characteristics, NTEE, NAICS; manual/multiple NTEE updates | Proprietary/subscription | Page © 2026 | Types include Churches; Issues include Religion; Characteristics include Religious/Christian; NTEE X; NAICS 813110 | Excellent validation candidate but not reproducible unless licensed |
| NCS restricted geography/linked files | https://www.nationalcongregationsstudy.org/data-documentation | Detailed geography and GSS-linked congregation data | Restricted via NORC/ICPSR process | NCS/ICPSR current | Geocodes, linked respondent/congregation data | Requires application; cannot redistribute |
| ANGEL non-profit-search/evaluation data | https://aclanthology.org/2023.emnlp-industry.77.pdf | 594K fund-givers, 194K fund-seekers, 463×100 evaluation matrix described | Not publicly released | EMNLP 2023 paper; GitHub says code coming soon | IRS 990-derived mission descriptions, cause/beneficiary matching labels | Startup/domain-expert annotations and data unavailable; not reproducible as-is |

## Code and Model Repositories

| Artifact | URL | Contents | Access Type | Update Signal | Relevant Variables/Text Fields | Reproducibility Caveats |
|---|---|---|---|---|---|---|
| `ma-ji/npo_classifier` | https://github.com/ma-ji/npo_classifier | NTEE classifier API, UCF benchmarks, scripts, outputs, reference assignments | Open GitHub | 338 commits; no formal release observed | Text descriptions of nonprofits; NTEE classes including X/religion | GitHub API rate-limited; some remapped data author-hosted, not DOI-archived |
| `fjsantam/bespoke-npo-taxonomies` | https://github.com/fjsantam/bespoke-npo-taxonomies | Replication website/docs/data for Million Missions | Open GitHub | 45 commits; no release | 1023-EZ `MISSION`, purpose codes, NTEE major groups | Main data are on Dataverse; GitHub says relevant files hosted where space permits |
| NODC `machine_learning_mission_codes` | https://github.com/Nonprofit-Open-Data-Collective/machine_learning_mission_codes | R code/data for mission statement classifiers | Open GitHub | 322 commits; no release | Names, mission fields, program service text, NTEE, IRS purpose codes incl. religious | Data page notes incomplete documentation; benchmark sample construction not fully documented |
| NODC `irs-990-efiler-database` | https://github.com/Nonprofit-Open-Data-Collective/irs-990-efiler-database | R/Python scripts and concordance for IRS 990 XML to relational tables | Open, GPL-2.0 | 47 commits; moved to `irs990efile` | Form 990/990-EZ mission/program text fields | Legacy repo; README says project moved |
| NODC `mission-taxonomies` | https://github.com/Nonprofit-Open-Data-Collective/mission-taxonomies | Taxonomy/crosswalk files for NTEE, NAICS, PCS, IRS activity/purpose codes | Open GitHub | 57 commits | NTEE X, IRS religious purpose/activity codes | Crosswalk repo, not labels for specific orgs |
| UK-CAT `charity-classification/ukcat` | https://github.com/charity-classification/ukcat | Python module, notebooks, data outputs, regex rules, logistic model workflow | Open, CC BY 4.0 | 118 commits; latest release v0.3 Nov 15 2021; docs show May 14 2026 | Charity name, activities, objects; UK-CAT religion tags RL101-RL305; ICNP/TSO category | Some Airtable-fetch steps require credentials; model pickle generated from local data |
| GivingTuesday `religious_org_v1` | https://huggingface.co/GivingTuesday/religious_org_v1 | Public BERT classifier, model card, linked training dataset | Open, Apache-2.0 | Dataset updated Nov 18 2025; 1 download last month | Name, mission statement, activities; predicts religious affiliation / religious vs not | Training notebook partly internal; GPT-4 labels and synthetic underrepresented examples need audit |
| `saifulhaq95/ANGEL` | https://github.com/saifulhaq95/ANGEL | README + camera-ready PDF | Open placeholder | 9 commits; no release | Nonprofit mission descriptions used in paper | README: “Code Coming soon!”; no code/data currently available |

## Taxonomies, Codebooks, and Reusable Label Schemas

| Artifact | Stable URL/DOI | Contents | Access Type | Update Signal | Religious-Relevant Fields/Codes | Caveats |
|---|---|---|---|---|---|---|
| NCCS/NTEE code categories | https://nccs.urban.org/nccs/widgets/ntee_tables/ntee_descriptions.html | NTEE hierarchy and descriptions | Public | Urban/NCCS web docs | Level 1 REL; Major Group X Religion-Related; X20 Christian, X00 Religion general in related docs | Primary-purpose taxonomy; can miss multi-mission religious orgs |
| NCCS NTEE two-page PDF | https://nccs.urban.org/nccs/pubs/ntee-two-page-2005.pdf | Compact NTEE code list | Public | Legacy 2005 PDF | X01, X02, X03, X05, X11, X12, X20, X21, X22, X30, X40, X50, X70, X80, X81, X82, X84/X85, X90, X99 | Legacy but useful for codebook prompts |
| NODC `mission-taxonomies` | https://github.com/Nonprofit-Open-Data-Collective/mission-taxonomies | NTEE, NTEEV2, PCS, NAICS, IRS activity and purpose codes | Public | 57 commits | IRS Tax Exempt Purpose Codes include religious purpose; NTEE X | Crosswalk quality should be verified before production labels |
| Candid PCS | https://taxonomy.candid.org/ | Subjects, populations, org type, transaction type, support strategies | Open taxonomy, CC BY 4.0 | Current site | Religion-related subject/org-type terms | Does not by itself provide organization labels unless Candid data licensed |
| UK-CAT religion tags | https://charityclassification.org.uk/data/tag_list/ | 24 categories, 17 subcategories, 230 tags | Open, CC BY 4.0 | Site dated May 14 2026 | Religion RL101-RL108, RL200-RL206, RL300-RL305; SO107 religious/racial/cross-border harmony; PR101 Clergy | UK charity context; regex/rule bias should be audited before U.S. reuse |
| NCS codebook/questionnaires | https://www.nationalcongregationsstudy.org/data-documentation | Exact question wording and weights | Public docs | Current NCS site | Denomination, formal affiliation, worship/prayer/religious education/program variables | Survey congregations, not administrative nonprofit text |
| Sider & Unruh typology | https://doi.org/10.1177/0899764003257494 | Faith-based organization typology: identity, mission, founding, affiliation, governance, staff, support, practices, content | Article/codebook-like typology | 2004 article | Strong conceptual codebook for labels | No public machine-readable replication package found |
| Bielefeld & Cleveland FBO definition | https://doi.org/10.1177/0899764013484090 | Measurement/definition caveats for FBOs | Article | 2013 article | Distinguishes organizational identity, funding, mission, service | No public package found |
| Smith & Sosin faith-related agencies | https://doi.org/10.1111/0033-3352.00137 | Varieties of faith-related agencies | Article | 2001/2002 article | Auspice and faith-related agency dimensions | No public package found |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | API queries for all priority titles; file API for `doi:10.7910/DVN/BL6XLW`, `EO2HIM`, `4GZJSK` | Santamarina package verified; title searches otherwise noisy/no target top hits | File counts: 12 raw/cleaned, 3 preprocessed, 70 classification files |
| OpenICPSR | Site-scoped exact priority query; Paxton glossary/stemmer query | No exact priority replication packages; adjacent NaNDA religious org datasets found | OpenICPSR is web-only; no file API inspection |
| OSF | API `filter[title]` for priority titles | 0 returned for target titles | Ma repo links OSF preprint DOI, not package |
| Zenodo | API fielded `title:"..."` priority queries | 0 exact target title matches | Initial broad queries were irrelevant/rate-limited once |
| GitHub | Web search/fetch for exact project repos | Ma, Santamarina, NODC, UK-CAT, ANGEL found | GitHub API rate-limited; used webfetch landing pages |
| Hugging Face | Direct fetch GivingTuesday model | 1 model + linked dataset | Public model artifact, not academic replication package |
| Official docs | IRS, NCS/ICPSR, NCCS fallback, Candid, Cause IQ | Multiple official/admin/proprietary sources | Distinguished from research packages in tables above |

## Papers With No Package Found

- Fyall, Moore & Gugerty (2018), *Beyond NTEE Codes* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, SAGE/web. Author website: not systematically scraped beyond web discovery. No public package found; article describes dictionary/content coding but no archived code/data found.
- Litofcenko, Karner & Maier (2020), semi-automated nonprofit classification — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, Springer/WU pages. Author website: WU publication page checked. No public sample-code repository found despite article abstract saying sample code is provided.
- Sider & Unruh (2004), FBO typology — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, web. Author website: not found/checked in depth. No machine-readable artifact; use article typology as codebook.
- Bielefeld & Cleveland (2013), defining FBOs — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, web. Author website: not checked in depth. No package found.
- Smith & Sosin (2001/2002), varieties of faith-related agencies — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, web. Author website: not checked in depth. No package found.
- Becker (2003) / Ebaugh, Chafetz & Pipes FBO measures — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, web. Author/institution fallback found CMACS reference in upstream report but no open dataset/codebook package verified.
- ANGEL (Haq et al. 2023) — Searched: GitHub, ACL, Dataverse, OSF, Zenodo, OpenICPSR. Package found only as README/PDF; code/data/model artifacts not released.

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| Universal Classification Files (UCF) | GitHub | Ma (2021) | Folder of train/test benchmark files | Nonprofit text descriptions for NTEE classification | Open | https://github.com/ma-ji/npo_classifier/tree/master/dataset/UCF |
| Remapped U.S. nonprofit sector | Author/GitHub-linked | Ma (2021) | Author-hosted data link | 439k+ nonprofits multi-labeled per paper | Open but author-hosted | https://jima.me/?ntee_remap |
| Raw/cleaned IRS 1023-EZ data | Harvard Dataverse | Santamarina et al. (2023) | CSV/XLSX/TAB | 2017-2019 1023-EZ approvals; 2018/2019 mission field | Open | https://doi.org/10.7910/DVN/BL6XLW |
| Preprocessed mission corpora | Harvard Dataverse | Santamarina et al. (2023) | CSV | 104,072 mission statements; minimal/standard/custom cleaning | Open | https://doi.org/10.7910/DVN/EO2HIM |
| Classification outputs for purpose/NTEE variables | Harvard Dataverse | Santamarina et al. (2023) | RData/RDS | Naive Bayes bootstrap results incl. `Orgpurposereligious` and `ntmaj10rel` | Open | https://doi.org/10.7910/DVN/4GZJSK |
| NODC mission training data | GitHub | Nonprofit Mission Classifiers | CSV/RDS | IRS e-file mission/program text + 1023-EZ labels incl. religious purpose | Open | https://github.com/Nonprofit-Open-Data-Collective/machine_learning_mission_codes/tree/master/DATA |
| UK-CAT all charity tags | GitHub | UK Charity Classification | CSV | Active/inactive UK charities with UK-CAT and ICNP/TSO outputs | Open, CC BY 4.0 | https://github.com/charity-classification/ukcat/tree/main/data |
| GivingTuesday religious org training dataset | Hugging Face | GivingTuesday model | HF dataset | 2k+ training/validation + 500 test described; viewer showed 498 rows | Open | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training |
| NCS cumulative file | ICPSR/NCS/ARDA | NCS | ICPSR datasets/codebooks | U.S. congregations, 1998-2019 | Public/restricted mix | https://doi.org/10.3886/ICPSR03471.v6 |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| Ma (2021) | Python/Jupyter | `API/`, `script/classification_algorithms/`, `reference/assign_NTEE/` | BERT/ML classifiers for NTEE coding | https://github.com/ma-ji/npo_classifier | Open | https://github.com/ma-ji/npo_classifier |
| Santamarina et al. (2023) | R/Python/docs | `docs/Preprocessing_Replication.html`, `docs/Classification_Bootstrapping_Replication`, `DATA/`, `data/` | Quanteda preprocessing + Naive Bayes bootstrapped classifiers | https://github.com/fjsantam/bespoke-npo-taxonomies | Open | https://github.com/fjsantam/bespoke-npo-taxonomies |
| Nonprofit Mission Classifiers | R | `DATA/MISSION.csv`, docs/vignettes, taxonomy pages | Benchmark mission classifiers over IRS text | https://github.com/Nonprofit-Open-Data-Collective/machine_learning_mission_codes | Open | https://github.com/Nonprofit-Open-Data-Collective/machine_learning_mission_codes |
| IRS 990 e-filer database | R/Python | `BUILD_SCRIPTS/`, `MASTER_CONCORDANCE_V0.csv`, `Build-Efiler-Index.md` | Convert IRS XML to relational/indexed data | https://github.com/Nonprofit-Open-Data-Collective/irs-990-efiler-database | Open, GPL-2.0; legacy | https://github.com/Nonprofit-Open-Data-Collective/irs-990-efiler-database/tree/master/BUILD_SCRIPTS |
| Mission taxonomies | R/HTML/CSV | `NTEE/`, `NTEEV2/`, `PCS/`, `irs-tax-exempt-purpose-codes/`, `irs-activity-codes/` | Crosswalk/taxonomy support files | https://github.com/Nonprofit-Open-Data-Collective/mission-taxonomies | Open | https://github.com/Nonprofit-Open-Data-Collective/mission-taxonomies |
| UK-CAT | Python/Jupyter | `src/ukcat`, `notebooks/icnptso-machine-learning-test.ipynb`, `data/ukcat.csv`, `data/icnptso.csv` | Regex tagger + logistic regression ICNP/TSO classifier | https://github.com/charity-classification/ukcat | Open, CC BY 4.0 | https://github.com/charity-classification/ukcat/tree/main/src/ukcat |
| GivingTuesday religious classifier | Python/Transformers | `config`, tokenizer, safetensors model; model card usage snippet | BERT text classification for religious org labels | https://huggingface.co/GivingTuesday/religious_org_v1 | Open, Apache-2.0 | https://huggingface.co/GivingTuesday/religious_org_v1/tree/main |
| ANGEL | None released | README/PDF only | ColBERT/SLA/DPRF described; code not released | https://github.com/saifulhaq95/ANGEL | Open placeholder | None found |

## Practical Recommendations for This Project

1. **Best reusable benchmark labels:** Santamarina Dataverse (`Orgpurposereligious`, `ntmaj10rel`) + NODC `MISSION.csv`/`MISSION.rds` + NTEE X from NCCS/IRS/BMF.
2. **Best external validation sources:** NCS/ARDA/ICPSR for true congregations; UK-CAT religion tags as a cross-national stress test; Cause IQ/Candid if licensed.
3. **Best reusable code:** Start with NODC IRS e-file parsing and mission-taxonomy crosswalks; compare against Ma’s NTEE classifier and UK-CAT’s regex/logistic design.
4. **Best model artifact:** GivingTuesday `religious_org_v1` is the closest public religious nonprofit model. Treat it as a benchmark/teacher, not ground truth, because GPT-4 labels and synthetic examples need auditing.
5. **Validation caveat:** Administrative labels (`NTEE X`, IRS religious purpose) are high-value weak labels but incomplete for faith-affiliated hospitals, schools, international NGOs, foundations, and filing-exempt churches.
