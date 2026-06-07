---
created: 20260605
agent: replication-seeker
scratchpad: .agents/notebooks/20260605-replication-imbalanced-validation-scratchpad.md
status: complete
title: Replication - Imbalanced Validation
topic: imbalanced text classification; validation metrics; social measurement; weak supervision; reproducibility artifacts
---

# Replication: Imbalanced Binary Text Classification Validation

## Search Strategy and Exact Queries Used

Scope came from `.agents/docs/20260605-literature-imbalanced-text-validation.md` plus the user’s priority list. I searched Dataverse, OpenICPSR, OSF, Zenodo, GitHub, JMLR/ACL/PLOS/OpenReview/Cambridge artifact links, and author pages.

Exact repository/API queries used:

- API batch title/author keys: `Davis Goadrich precision recall ROC curves`; `Saito Rehmsmeier precision recall plot imbalanced`; `Fawcett introduction ROC analysis`; `Drummond Holte cost curves`; `Hand coherent alternative AUC H measure`; `Hernandez-Orallo Flach Ferri expected classification loss`; `Brodersen balanced accuracy posterior distribution`; `Chicco Warrens Jurman Matthews correlation coefficient`; `Hopkins King automated nonparametric content analysis ReadMe`; `Grimmer Stewart text as data promise pitfalls`; `Gentzkow Kelly Taddy text as data`; `Nelson Burk Knudsen McCall Future of Coding`; `ValiText validation framework computational text based measures`; `trustworthy measures supervised machine learning text PSRM`; `BOXWRENCH benchmarking weak supervision realistic tasks`; `WRENCH weak supervision benchmark`; `AlleNoise noisy labels text classification benchmark`.
- Web/GitHub targeted queries: `"A Method of Automated Nonparametric Content Analysis for Social Science" replication data Hopkins King ReadMe Dataverse 1902.1/12898`; `"ValiText" "github" "validation framework"`; `"BOXWRENCH" "GitHub" "Stronger Than You Think"`; `"WRENCH" "weak supervision" benchmark GitHub datasets code`; `"AlleNoise" noisy label text classification benchmark GitHub Zenodo`.
- Metric/software queries: `"Hand" "H measure" AUC R package hmeasure GitHub`; `"Cost Curves" Drummond Holte code R package costcurve GitHub`; `"The Relationship Between Precision-Recall and ROC Curves" code implementation GitHub Davis Goadrich`; `"A Unified View of Performance Metrics" Hernandez-Orallo Flach Ferri software expected classification loss`; `"The Balanced Accuracy and Its Posterior Distribution" code MATLAB R Brodersen balanced accuracy posterior`.
- Social-science artifact queries: `"The Future of Coding" Nelson Burk Knudsen McCall replication data OSF GitHub Dataverse`; `"Text as Data" "Gentzkow" "Kelly" "Taddy" replication code GitHub Dataverse AEA`; `"Text as Data: The Promise and Pitfalls" Grimmer Stewart replication materials code`; `"Toward a Framework for Creating Trustworthy Measures with Supervised Machine Learning for Text" replication data code PSRM 2025`; `"Large Language Models as a Substitute for Human Experts" Gilardi Alizadeh Kubli replication data OSF GitHub Harvard Dataverse`.
- OpenICPSR web-only checks: `site:openicpsr.org "Davis" "Goadrich" "Precision-Recall" "ROC"`; `site:openicpsr.org "Hopkins" "King" "Automated Nonparametric Content Analysis"`; `site:openicpsr.org "Grimmer" "Stewart" "Text as Data"`; `site:openicpsr.org "Gentzkow" "Kelly" "Taddy" "Text as Data"`; `site:openicpsr.org "imbalanced" "text classification" "replication" OR "validation"`.

## Verified Packages

| Paper | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|-------|-----------|-----|----------------|---------------|-------------|------------|
| Hopkins & King (2010), automated nonparametric content analysis | Harvard Dataverse | https://doi.org/10.7910/DVN/NV0SZJ | R | Yes: coded controls, text corpora, Rdata, codebook/coding instructions | Open | 28 |
| ReadMe software for Hopkins & King | GitHub | https://github.com/iqss-research/ReadMeV1 | R/Python bridge | Demo only | Open, academic/noncommercial license noted by author page | 11 root items |
| Grimmer & Stewart (2013), Text as Data | Harvard Dataverse | https://doi.org/10.7910/DVN/FQBHP8 | Archive bundle; likely R/code inside ZIP | Yes, replication bundle | Open | 1 ZIP |
| Park & Montgomery (2025), trustworthy supervised ML text measures | Harvard Dataverse | https://doi.org/10.7910/DVN/AFBW80 | Archive bundle | Yes: replication tarball for Senate confirmation-hearing tone measure | Open | 1 TAR.GZ |
| Gilardi, Alizadeh & Kubli (2023/2024), LLM annotation | Harvard Dataverse | https://doi.org/10.7910/DVN/PQYF6M | R, Python | Yes: annotation spreadsheets, batch model outputs, plots | Open; tweet text may be partially restricted to IDs per paper | 62 |
| ValiText / ValiTex validation framework | GitHub + GESIS toolbox | https://github.com/lukasbirki/ValiTex | R/Shiny | Checklist framework data (`framework.rda`) | Open | 25 root items |
| Nelson et al. (2021), The Future of Coding | GitHub | https://github.com/lknelson/future-of-coding | Jupyter, R, Stata, Python | Partial: sharable data, DTM/hand codes, classifier outputs; full article text excluded by copyright | Open partial | 8 root items; 7 data files |
| Updating/Future of Coding Revisited (2025) | GitHub | https://github.com/lknelson/future-of-coding-revisited | Jupyter/Python | Partial: hand/LLM codes and outputs; full news text not shared | Open partial | 11 root items |
| BOXWRENCH / Stronger Than You Think | GitHub/OpenReview | https://github.com/jeffreywpli/stronger-than-you-think | Python, Jupyter | Benchmark/task assets and result folders; some datasets external/WRENCH-format | Open | 5 root items; 16 `end_model_training` items |
| WRENCH weak-supervision benchmark | GitHub + Hugging Face dataset | https://github.com/JieyuZ2/wrench ; https://huggingface.co/datasets/jieyuz2/WRENCH | Python | Yes: benchmark dataset format; datasets available through HF snapshot | Open | 9 root items; 20 examples |
| AlleNoise noisy-label text benchmark | GitHub + Zenodo | https://github.com/allegro/AlleNoise ; https://doi.org/10.5281/zenodo.11500851 | Python | Yes: `full_dataset.csv`, taxonomy mapping, data sheet | Open | 5 root items; Zenodo 1 data-sheet file |
| hmeasure package for Hand H-measure | GitHub/CRAN | https://github.com/canagnos/hmeasure | R | Example/vignette data only | Open | 16 root items |
| precrec PR/ROC package by Saito & Rehmsmeier | GitHub/CRAN | https://github.com/evalclass/precrec | R/C++ | Example data | Open | 24 root items |
| Brodersen posterior balanced accuracy | MATLAB Central + GitHub `micp` | https://www.mathworks.com/matlabcentral/fileexchange/29244-computing-the-posterior-balanced-accuracy ; https://github.com/kaybrodersen/micp | MATLAB, R | Examples only | Open | MATLAB archive; 13 `micp` root items |
| Davis & Goadrich PR/ROC AUCCalculator derivatives | Author PDF link + GitHub adaptations | http://mark.goadrich.com/programs/AUC/ ; https://github.com/deborah-chasman/auc_orig_points ; https://github.com/ameya98/pr2roc | Java, Python | Test/demo data only | Author/GitHub open where available | 5 root items (`auc_orig_points`); 10 root items (`pr2roc`) |
| Drummond & Holte cost curves | Author page / Weka implementation | http://webdocs.cs.ualberta.ca/~holte/CostCurves/ ; https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.6.9/weka/classifiers/evaluation/CostCurve.html | Java/Weka | No paper data found | Author-hosted / library docs | Web-only |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | API batch over 17 exact title/author-key queries; DOI inspections for `10.7910/DVN/NV0SZJ`, `FQBHP8`, `AFBW80`, `PQYF6M` | 4 verified DOI packages | Broad API search noisy; exact DOI/article pages verified social-science packages. |
| OpenICPSR | Five `site:openicpsr.org` exact-title/author queries listed above | 0 exact priority packages | Returned unrelated ML/text datasets; no priority replication package verified. |
| OSF | API `filter[title]` batch over 17 title/author-key queries | 0 | No exact public OSF nodes found for priority list. |
| Zenodo | API batch over 17 title/author-key queries; record API for AlleNoise `11500851` | 1 verified Zenodo record | AlleNoise Zenodo record has DOI and data sheet; code/data primarily on GitHub. |
| GitHub | Web discovery + Contents API for `ReadMeV1`, `ValiTex`, `stronger-than-you-think`, `wrench`, `AlleNoise`, `future-of-coding`, `hmeasure`, `precrec`, `pr2roc`, `auc_orig_points`, `micp` | 12 verified repositories | Captured root/subdirectory manifests and last push dates. |
| Author pages / journal artifact links | Gary King, Robert Holte, JMLR, Cambridge, OpenReview, PLOS/PNAS/PMLR pages | Several artifact links | Used when archived repository was missing or article page named replication DOI. |

## Reusable Metric Implementations

| Artifact | Stable URL | Main reusable contents | Last-updated signal | Usefulness for this project |
|---|---|---|---|---|
| `precrec` | https://github.com/evalclass/precrec | Fast PR/ROC, AUC, partial AUC, confidence intervals, plotting; key files `R/main_evalmod.R`, `R/g_auc.R`, `src/` | GitHub pushed 2025-05-15 | Strong reference for PR-AUC/ROC plotting semantics; R not directly in Python module but validates metric definitions. |
| `hmeasure` | https://github.com/canagnos/hmeasure | H-measure, AUC, min weighted/total loss, ROC/H plots; key `R/library_metrics.R`, `R/library_plotting.R` | Pushed 2019-02-26 | Useful optional expected-loss/H-measure reference if reporting cost-sensitive aggregate metric. |
| Weka `CostCurve` | https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.6.9/weka/classifiers/evaluation/CostCurve.html | Cost curve generation with probability-cost function, normalized expected cost, threshold | Weka docs for 3.6.9 | Good algorithmic reference for threshold/cost curves; not directly reusable in Python. |
| Brodersen posterior balanced accuracy | https://www.mathworks.com/matlabcentral/fileexchange/29244-computing-the-posterior-balanced-accuracy | MATLAB functions `bacc_mean`, `bacc_p`, `bacc_ppi`, beta convolution utilities | Published 2010-11-02 | Useful design for confidence/credible intervals around balanced accuracy. |
| `micp` | https://github.com/kaybrodersen/micp | R inference on balanced classification accuracy; key `R/micp.stats.R` | Pushed 2022-03-20 | More modern balanced-accuracy inference reference. |
| `pr2roc` | https://github.com/ameya98/pr2roc | Python PR↔ROC curve conversion and resampling; `pr2roc/pr_curve.py`, `roc_curve.py` | Pushed 2020-07-20 | Useful for checking PR interpolation pitfalls from Davis & Goadrich. |
| `auc_orig_points` | https://github.com/deborah-chasman/auc_orig_points | Java adaptation of Davis/Goadrich AUCCalculator with PR point output | Pushed 2017-10-17 | Historical reference only; prefer scikit-learn plus explicit interpolation documentation. |

## Validation Frameworks and Checklists

| Artifact | Stable URL | Contents | Usefulness for this project |
|---|---|---|---|
| ValiText/ValiTex | https://github.com/lukasbirki/ValiTex ; https://kodaqs-toolbox.gesis.org/github.com/lukasbirki/tool_valitext/index/ | R/Shiny checklist app, `data/framework.rda`, UI/server files; use cases include supervised and prompt-based classification | Best source for a project reporting checklist: substantive/structural/external validation evidence. |
| Park & Montgomery PSRM framework | https://doi.org/10.1017/psrm.2025.10042 ; https://doi.org/10.7910/DVN/AFBW80 | Text-to-measure reporting framework and replication tarball; validates labels, held-out model, final measure | Directly useful for audit-set design: separate label, model, and final-measure validation groups. |
| Future of Coding | https://github.com/lknelson/future-of-coding | Hand-coded news data proxies, SML/dictionary/topic outputs, accuracy graph scripts | Useful example of comparing hand coding, dictionaries, SML, and unsupervised text measures. |
| Future of Coding Revisited | https://github.com/lknelson/future-of-coding-revisited | LLM prompt/few-shot/zero-shot scripts, inter-rater notebooks, accuracy metrics | Useful for LLM-label audit reporting and agreement calculations. |
| Hopkins-King ReadMe | https://doi.org/10.7910/DVN/NV0SZJ ; https://github.com/iqss-research/ReadMeV1 | Codebook/coding instructions and aggregate prevalence estimation examples | Useful for aggregate prevalence validation and warning against individual accuracy-only reporting. |

## Datasets and Benchmarks

| Dataset/Benchmark | Source | Format / files | Coverage | Access |
|---|---|---|---|---|
| Hopkins-King replication corpora | Harvard Dataverse `10.7910/DVN/NV0SZJ` | `.tab`, `.txt`, `.zip`, `.R`, `.Rdata`, `.doc`; 28 files | Blogs/speeches/congress/Enron/immigration examples for aggregate content analysis | Open |
| Grimmer-Stewart Text as Data replication | Harvard Dataverse `10.7910/DVN/FQBHP8` | `ReplicationFile.zip` (5.9MB) | Political text validation examples | Open |
| Park-Montgomery PSRM replication | Harvard Dataverse `10.7910/DVN/AFBW80` | `ParkMontgomery_ReplicationData.tar.gz` (518MB) | Senate confirmation-hearing tone measure; crowd/expert validation design | Open |
| Gilardi et al. LLM annotation | Harvard Dataverse `10.7910/DVN/PQYF6M` | 62 files: `.R`, `.py`, `.xlsx`, `.csv`, `.tab`, plots | Tweets/news/congress annotation tasks; RA, MTurk, ChatGPT outputs | Open; some raw tweet text limitations |
| WRENCH | https://huggingface.co/datasets/jieyuz2/WRENCH | JSON benchmark format; train/valid/test splits; weak labels | 22 weak-supervision datasets, many binary text tasks (SMS, YouTube, IMDB, Yelp, Spouse, CDR) | Open |
| BOXWRENCH | https://github.com/jeffreywpli/stronger-than-you-think | WRENCH-format tasks, LF design assets, benchmark scripts | Realistic weak supervision tasks with imbalance/domain expertise | Open, but some data may be external |
| AlleNoise | https://github.com/allegro/AlleNoise | `allenoise/full_dataset.csv` (33.8MB), `category_mapping.csv`, `metadata.json`, `data_sheet.pdf` | 502k+ product titles, noisy and clean labels, ~5.6k categories | Open |
| Future of Coding data | https://github.com/lknelson/future-of-coding/tree/master/data | CSV/Rdata outputs; no full copyrighted article text | 1,253 hand-coded news articles with derived features/results | Open partial |

## Useful for This Project’s Evaluation Module or Reporting Checklist

1. **Core metric module**: implement confusion matrix, positive-class precision/recall, F1/Fβ, balanced accuracy, MCC, Cohen’s κ where human/model agreement is relevant, average precision/PR-AUC, ROC-AUC as secondary, Brier/log-loss if probabilities are used. Use `precrec`, `pr2roc`, and Davis/Goadrich as interpolation/visualization references.
2. **Threshold/reporting curves**: add threshold table over candidate thresholds: predicted-positive rate, precision, recall, Fβ, FP/1,000, FN/1,000, balanced accuracy, MCC, expected loss. Use Drummond-Holte/Weka cost curve and Hernandez-Orallo expected-loss framing as conceptual references.
3. **Uncertainty around metrics**: bootstrap CIs for PR-AUC/F1/MCC and consider Brodersen-style posterior intervals for balanced accuracy.
4. **Validation checklist**: adapt ValiText plus Park & Montgomery into a project checklist: construct definition, codebook, label acquisition, double-coding/adjudication, held-out random audit, positive-enriched audit, score-stratified/near-threshold audit, external/prevalence validation, subgroup error checks.
5. **Weak-label evaluation**: use WRENCH/BOXWRENCH patterns for separating weak labels from clean validation labels and for recording label-function coverage/conflict/abstention.
6. **LLM-label audit**: use Gilardi and Future-of-Coding-Revisited as examples for preserving prompts/model versions, comparing to human labels, and reporting inter-rater agreement.

## Papers With No Package Found

- Davis & Goadrich (2006), PR vs ROC — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, author pages. Official paper mentions Java AUCCalculator at Mark Goadrich’s old URL; I found GitHub derivatives (`auc_orig_points`, `pr2roc`) but no DOI-backed paper replication package.
- Saito & Rehmsmeier (2015), PLOS ONE PR under imbalance — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, PLOS/author queries. No exact PLOS paper replication package found; later `precrec` R package by the authors is verified and reusable.
- Fawcett (2006), Introduction to ROC Analysis — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub. No paper-specific replication package found; use standard library implementations.
- Drummond & Holte (2006), Cost Curves — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, author pages. Author-hosted cost-curve page exists and Weka implements cost curves; no DOI-backed package/manifest found.
- Hand (2009), AUC critique/H-measure — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub. No paper-specific replication package found; `hmeasure` R package by Hand/Anagnostopoulos collaborators is verified.
- Hernández-Orallo, Flach & Ferri (2012), expected loss/threshold metrics — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, JMLR page. No replication package found for the 2012 JMLR paper. Related later cost-uncertainty work refers to Ferri et al. repository, but not verified as this paper’s replication archive.
- Chicco, Warrens & Jurman (2021), MCC — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub. No paper-specific package found; MCC is already implemented in scikit-learn/R metric libraries.
- Gentzkow, Kelly & Taddy (2019), Text as Data survey — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, AEA/NBER/SSRN. No replication package found; it is a survey/introduction rather than an empirical replication article.

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| Hopkins-King replication corpora and controls | Harvard Dataverse | Hopkins & King (2010) | `.tab`, `.txt`, `.zip`, `.Rdata` | Aggregate social-science content-analysis examples | Open | https://dataverse.harvard.edu/api/access/datafile/1506179 |
| Hopkins-King coding instructions | Harvard Dataverse | Hopkins & King (2010) | `.doc` | Hand-coding guidance/codebook | Open | https://dataverse.harvard.edu/api/access/datafile/1506200 |
| Grimmer-Stewart replication bundle | Harvard Dataverse | Grimmer & Stewart (2013) | `.zip` | Political text-as-data validation examples | Open | https://dataverse.harvard.edu/api/access/datafile/2418053 |
| Park-Montgomery replication data | Harvard Dataverse | Park & Montgomery (2025) | `.tar.gz` | Senate hearing tone labels/model validation | Open | https://dataverse.harvard.edu/api/access/datafile/11696848 |
| Gilardi et al. annotation datasets | Harvard Dataverse | Gilardi, Alizadeh & Kubli | `.xlsx`, `.csv`, `.tab` | RA/MTurk/ChatGPT annotations for tweets/news/congress | Open partial | https://dataverse.harvard.edu/api/access/datafile/7194100 |
| WRENCH benchmark dataset | Hugging Face/GitHub | WRENCH | JSON | Weak-supervision text and sequence-tagging datasets | Open | https://huggingface.co/datasets/jieyuz2/WRENCH |
| AlleNoise full dataset | GitHub | AlleNoise | CSV | Product-title text classification with noisy and clean labels | Open | https://raw.githubusercontent.com/allegro/AlleNoise/main/allenoise/full_dataset.csv |
| Future of Coding hand-code/DTM data | GitHub | Nelson et al. (2021) | CSV/Rdata | Hand-coded inequality news features and classifier outputs | Open partial | https://raw.githubusercontent.com/lknelson/future-of-coding/master/data/mccall_doc_term_matrix_and_hand_code.Rdata |
| ValiText checklist framework data | GitHub | ValiText | RDA | Validation-step/checklist metadata | Open | https://raw.githubusercontent.com/lukasbirki/ValiTex/main/data/framework.rda |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| Hopkins & King (2010) / ReadMe | R | `R/prototype.R`, `demo/clinton.R` | Aggregate prevalence/nonparametric content analysis | https://github.com/iqss-research/ReadMeV1 | Open | https://raw.githubusercontent.com/iqss-research/ReadMeV1/master/R/prototype.R |
| Hopkins & King replication | R | `biasgraphs072108.R`, `kerrytest030108.R`, `synthetic032908.R` | Replication scripts | https://doi.org/10.7910/DVN/NV0SZJ | Open | https://dataverse.harvard.edu/api/access/datafile/1506182 |
| Gilardi et al. LLM annotation | Python/R | `03-01-chatgpt-Zeroshot-Task-template.py`, `02-01-mTurk_eval_data.R` | Prompted LLM annotation vs human/crowd evaluation | https://doi.org/10.7910/DVN/PQYF6M | Open | https://dataverse.harvard.edu/api/access/datafile/7218458 |
| ValiText | R/Shiny | `R/app_ui.R`, `R/app_server.R`, `R/run_app.R` | Validation checklist app | https://github.com/lukasbirki/ValiTex | Open | https://raw.githubusercontent.com/lukasbirki/ValiTex/main/R/app_ui.R |
| Future of Coding | Jupyter/R/Stata | `gen_graphs.do`, `02-Dictionaries/dictionary_method_w_all_terms.R` | Accuracy/validation comparisons of hand/SML/dictionary/topic methods | https://github.com/lknelson/future-of-coding | Open partial | https://raw.githubusercontent.com/lknelson/future-of-coding/master/gen_graphs.do |
| Future of Coding Revisited | Jupyter | `scripts-generate-accuracy-metrics/generate_accuracy_metrics.ipynb`, `scripts-inter-agreement/*` | LLM coding accuracy and inter-rater agreement | https://github.com/lknelson/future-of-coding-revisited | Open partial | https://raw.githubusercontent.com/lknelson/future-of-coding-revisited/main/scripts-generate-accuracy-metrics/generate_accuracy_metrics.ipynb |
| BOXWRENCH | Python/Jupyter | `end_model_training/pipelines.py`, `val_size_experiment.py`, `autows_bench_101/pipeline.py` | Weak-supervision benchmark pipelines and validation-size experiments | https://github.com/jeffreywpli/stronger-than-you-think | Open | https://raw.githubusercontent.com/jeffreywpli/stronger-than-you-think/main/end_model_training/pipelines.py |
| WRENCH | Python | `wrench/evaluation.py`, `examples/run_two_stage_pipeline_cls.py`, `examples/grid_search.py` | Weak-supervision benchmark/evaluation framework | https://github.com/JieyuZ2/wrench | Open | https://raw.githubusercontent.com/JieyuZ2/wrench/main/wrench/evaluation.py |
| AlleNoise | Python | `category_classifier/prepare_dataset.py`, `noise_generator/`, `category_classifier/bert_classifier/` | Noisy-label benchmark preparation and classifiers | https://github.com/allegro/AlleNoise | Open | https://raw.githubusercontent.com/allegro/AlleNoise/main/category_classifier/prepare_dataset.py |
| hmeasure | R | `R/library_metrics.R`, `R/library_plotting.R` | H-measure, AUC, loss/ROC metrics | https://github.com/canagnos/hmeasure | Open | https://raw.githubusercontent.com/canagnos/hmeasure/master/R/library_metrics.R |
| precrec | R/C++ | `R/main_evalmod.R`, `R/g_auc.R`, `R/pl4_calc_measures.R` | PR/ROC/AUC metrics and plots | https://github.com/evalclass/precrec | Open | https://raw.githubusercontent.com/evalclass/precrec/main/R/main_evalmod.R |
| Brodersen/micp | R | `R/micp.stats.R`, `R/vbicp.unb.R` | Bayesian inference for balanced classification accuracy | https://github.com/kaybrodersen/micp | Open | https://raw.githubusercontent.com/kaybrodersen/micp/main/R/micp.stats.R |
| Davis/Goadrich derivatives | Python/Java | `pr2roc/pr_curve.py`, `auc_orig_points.jar` | PR↔ROC conversion; historical AUCCalculator adaptation | https://github.com/ameya98/pr2roc ; https://github.com/deborah-chasman/auc_orig_points | Open | https://raw.githubusercontent.com/ameya98/pr2roc/master/pr2roc/pr_curve.py |

## Handoff Quality Gate

- Verified Packages: concrete entries present.
- Repository Search Log: concrete entries present.
- Papers With No Package Found: concrete entries present with repositories checked.
- Handoff: Datasets Found in Packages: concrete entries present.
- Handoff: Code Files Found in Packages: concrete entries present.
