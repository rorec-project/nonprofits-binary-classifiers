---
created: 20260605
agent: replication-seeker
scratchpad: docs/research/notebooks/20260605-replication-calibration-prevalence-scratchpad.md
status: complete
title: Replication - Calibration Prevalence
topic: classifier calibration; prevalence quantification; transformer calibration; social measurement; misclassification correction
---

# Replication: Calibration, Quantification, and Population-Scale Social Measurement

## Search Strategy and Exact Queries Used

Scope was seeded from `docs/research/20260605-literature-synthesis-map.md` and `docs/research/20260605-literature-calibrated-classifier-prevalence.md`, especially the handoff citation list. I treated artifacts as one of: **research replication archive**, **maintained software library**, **benchmark dataset**, or **author/project-hosted reproducibility material**.

### Repository/API queries

| Source | Exact query or endpoint pattern | Result summary |
|---|---|---|
| Harvard Dataverse API | `https://dataverse.harvard.edu/api/v1/search?type=dataset&per_page=5&q="On Calibration of Modern Neural Networks"` and analogous exact-title queries for Kull, Desai-Durrett, Mukhoti, Forman, Bella, Gonzalez, Hopkins-King, Keith-O'Connor, PPI, Meyer-Mittag, LeQua | No exact priority-paper matches in Dataverse API. Hopkins-King points to legacy `hdl:1902.1/12898`, not modern Dataverse DOI. |
| Zenodo API | `https://zenodo.org/api/records?size=5&q="Learning to quantify" Bella Ferri`; `q="Prediction-Powered Inference"`; `q="classifier calibration" ECE Brier reliability`; `q="quantification" "classify and count" "PACC"`; `q="LeQua" quantification benchmark` | Found LeQua 2022/2024, PPI data and code archives, PyCalib, Re-Assessing Classify-and-Count. |
| OSF API | `https://api.osf.io/v2/nodes/?page[size]=5&filter[public]=true&filter[title]={exact title}` for all priority titles | No relevant public OSF nodes found. |
| OpenICPSR web | `site:openicpsr.org/openicpsr "Hopkins" "King" "Automated Nonparametric Content Analysis"`; `"Prediction-Powered Inference" Angelopoulos`; `"Misclassification in Binary Choice Models" Meyer Mittag`; `"classifier calibration" "replication"`; `"quantification" "classify and count"` | No exact priority matches; hits were unrelated text/mining or general economics replication packages. |
| GitHub code search | `class ModelWithTemperature`; `DirichletCalibration`; `qp.method.aggregative.PACC`; `ppi_mean_ci`; `readme2`; `"Unbiased Prevalence Estimation with Multicalibrated LLMs"` | Found temperature scaling, focal calibration, PyHealth Dirichlet calibration, QuaPy, PPI, ReadMe/readme2, multicalibrated LLM prevalence repository. |
| Web search | `2026 calibration classifier library ECE Brier reliability diagram GitHub netcal PyCalib uncertainty calibration temperature scaling Dirichlet calibration`; `2026 quantification learning classify and count ACC PACC EM prior shift GitHub QuaPy quantificationlib LeQua Zenodo`; `Hopkins King ReadMe replication data software Gary King automated nonparametric content analysis GitHub CRAN`; `Keith O'Connor 2018 uncertainty-aware generative models inferring document class prevalence code data GitHub`; `Prediction-Powered Inference code data Zenodo GitHub ppi_py Angelopoulos Bates Fannjiang Jordan Zrnic 2023`; `"Unbiased Prevalence Estimation with Multicalibrated LLMs" code GitHub data artifacts 2026` | Found most author/project-hosted software and current maintained packages. |

## Verified Packages

| Paper / Artifact | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|-------|-----------|-----|----------------|---------------|-------------|------------|
| Guo et al. (2017), temperature scaling implementation | GitHub | https://github.com/gpleiss/temperature_scaling | Python / PyTorch | Demo/training scripts only; no large benchmark data | Open; maintained example/library; MIT | 8 top-level entries |
| Mukhoti et al. (2020), focal loss calibration | GitHub | https://github.com/torrvision/focal_calibration | Python, Jupyter | Data directory plus training/evaluation scripts; likely downloads/derived experiment data | Open research replication; MIT | 16 top-level entries |
| Kull et al. (2019), Dirichlet calibration project page | GitHub Pages | https://dirichletcal.github.io/ and https://github.com/dirichletcal/dirichletcal.github.io | HTML/docs; package links | Documents only at inspected repo | Open project material; no license detected | 3 top-level entries |
| PyCalib classifier calibration | Zenodo + GitHub snapshot | https://doi.org/10.5281/zenodo.5518877 | Python | Source archive only | Open software archive | 1 file (`perellonieto/PyCalib-v0.1.0.dev0.zip`) |
| netcal calibration framework | GitHub / PyPI docs | https://github.com/EFS-OpenSource/calibration-framework | Python | Examples, docs; no benchmark data bundled at top level | Open maintained library; Apache-2.0 | 11 top-level entries |
| Apple relplot / SmoothECE | GitHub | https://github.com/apple/ml-calibration | Python, Jupyter | Notebooks and generated figures | Open research software; license file present but GitHub NOASSERTION | 9 top-level entries |
| QuaPy quantification framework | GitHub | https://github.com/HLT-ISTI/QuaPy | Python | Dataset loaders for UCI, Twitter sentiment, reviews, LeQua, IFCB; not all raw data bundled | Open maintained library; BSD-3-Clause | 13 top-level entries |
| QuantificationLib | GitHub | https://github.com/AICGijon/quantificationlib | Python | Examples/tests; no large data at top level | Open maintained library; GPL-3.0 | 10 top-level entries |
| LeQua 2022 benchmark datasets | Zenodo | https://doi.org/10.5281/zenodo.6546188 | Dataset + readme | Yes: train/dev/test/test-prevalence ZIPs for T1A/T1B/T2A/T2B | Open benchmark dataset | 13 files |
| LeQua 2024 benchmark datasets | Zenodo | https://doi.org/10.5281/zenodo.11661820 | Dataset + readme | Yes: train/dev/test/test-prevalence ZIPs for T1-T4 | Open benchmark dataset | 13 files |
| Keith & O'Connor (2018), document prevalence replication | GitHub | https://github.com/slanglab/doc_prevalence | Python, shell, Jupyter | Yelp preprocessing scripts; raw Yelp data must be obtained separately | Open research replication; no license detected | 5 top-level entries |
| Keith & O'Connor freq-e software | GitHub / PyPI | https://github.com/slanglab/freq-e and https://pypi.org/project/freq-e/ | Python, Jupyter | Example data | Open software package; MIT | 6 top-level entries |
| Hopkins & King (2010), ReadMe software | GitHub + author page | https://github.com/iqss-research/ReadMeV1 ; http://gking.harvard.edu/readme/ | R, Python support | Demo/inst data; replication data is legacy handle | Open author/project-hosted software; no license detected | 12 top-level entries |
| Hopkins & King (2010), replication data | Murray Research Archive / handle | http://hdl.handle.net/1902.1/12898 | Data + replication files (legacy archive) | Yes, per AJPS/Wiley metadata | Public legacy archive; file listing not inspected programmatically | Web-only / legacy handle |
| Jerzak, King & Strezhnev readme2 | GitHub | https://github.com/iqss-research/readme-software | R | `data/`, `results/`, PDF docs | Open maintained-ish software; no license detected; pushed 2026-06-01 | 9 top-level entries |
| Angelopoulos et al. (2023), PPI package | GitHub + Zenodo | https://github.com/aangelopoulos/ppi_py ; https://doi.org/10.5281/zenodo.8403931 | Python | Example notebooks and package datasets loader; code archive on Zenodo | Open maintained library/repro package; MIT | 13 top-level entries; Zenodo has 1 source ZIP |
| Angelopoulos et al. (2023), PPI datasets | Zenodo | https://doi.org/10.5281/zenodo.8397451 | `.npz` datasets | Yes: census, plankton, alphafold, ballots, gene_expression, galaxies, forest | Open benchmark/replication dataset | 8 files |
| Unbiased Prevalence Estimation with Multicalibrated LLMs (2026) | GitHub | https://github.com/facebookresearch/multicalibrated_llm_measurement | Python, TeX, shell | Included CAP inference outputs/sample; ACS downloaded via `folktables`; CAP raw data downloaded separately | Open research replication; CC BY-NC; no releases; API inspection rate-limited, README fetched | README lists 4 main directories + root files |
| Binary Quantification and Dataset Shift (2024) | GitHub | https://github.com/pglez82/quant_datasetshift | Python, Jupyter | Scripts/notebooks; likely external Amazon reviews data | Open research replication; no license detected | 15+ top-level entries |
| Re-Assessing the Classify-and-Count Quantification Method | Zenodo | https://doi.org/10.5281/zenodo.4468277 | PDF only | No code/data | Open paper artifact, not replication package | 1 file |

## Calibration Libraries

| Artifact | Type | Stable URL / DOI | Contents | Update signal | Implementation relevance |
|---|---|---|---|---|---|
| `gpleiss/temperature_scaling` | Research code / reusable snippet | https://github.com/gpleiss/temperature_scaling | `temperature_scaling.py`, `demo.py`, `train.py`, `models/` | GitHub pushed 2025-07-26; widely forked | Best minimal PyTorch temperature scaling example; useful for BERT logits but older API. |
| `netcal` / calibration-framework | Maintained library | https://github.com/EFS-OpenSource/calibration-framework | `netcal.scaling` Logistic/Temperature/Beta; `netcal.metrics` ECE/MCE/ACE/MMCE; reliability diagrams; examples/docs | GitHub pushed 2026-04-16; Apache-2.0 | Strongest drop-in for Platt/logistic scaling, temperature, beta, ECE, reliability diagrams. Check Python 3.13 compatibility before adoption. |
| PyCalib | Software archive | https://doi.org/10.5281/zenodo.5518877 | Source ZIP for classifier calibration | Zenodo updated 2023-01-23 | Useful for Kull/Dirichlet-style calibration provenance; likely less maintained than netcal/probmetrics. |
| `dirichletcal` project page | Project/research page | https://dirichletcal.github.io/ | Documentation and links for Dirichlet calibration | GitHub pushed 2020-08-12 | Use as citation/provenance for Dirichlet calibration, not primary dependency. |
| `torrvision/focal_calibration` | Research replication archive | https://github.com/torrvision/focal_calibration | `Losses/`, `Metrics/`, `Net/`, `train.py`, `evaluate.py`, `temperature_scaling.py`, notebooks/data dirs | GitHub pushed 2024-01-10 | Good for focal-loss calibration experiments; not needed unless retraining loss is changed. |
| Apple `ml-calibration` / `relplot` | Research software | https://github.com/apple/ml-calibration | `src/`, `notebooks/figure1.ipynb`, `notebooks/paper_experiments.ipynb`, SmoothECE, binned ECE | GitHub pushed 2025-11-04 | Useful for principled reliability diagrams and smoothed ECE with CIs; good complement to standard ECE. |
| `probmetrics` | Maintained library | https://github.com/dholzmueller/probmetrics | Post-hoc calibrators: temperature, logistic/SMS/SVS, isotonic/Venn-Abers, Dirichlet optional; metrics: log-loss, Brier, ECE, smECE | GitHub result says pushed 2025-01-31+; active-looking | Advanced but heavier PyTorch package; useful if matrix/vector/structured scaling is needed. |
| PyHealth calibration | Domain library module | https://github.com/sunlabuiuc/PyHealth/blob/master/pyhealth/calib/calibration/dircal.py | TemperatureScaling, HistogramBinning, KCal, DirichletCalibration wrappers | Active PyHealth repo | Reusable patterns for post-hoc calibration classes, but healthcare-oriented. |

## Quantification / Prevalence Packages and Benchmarks

| Artifact | Type | Stable URL / DOI | Contents | Update signal | Implementation relevance |
|---|---|---|---|---|---|
| QuaPy | Maintained software library | https://github.com/HLT-ISTI/QuaPy | CC, ACC, PCC, PACC, EMQ/SLD, HDy, DyS, QuaNet, ensembles, protocols, metrics, dataset loaders | Pushed 2026-03-03; BSD-3-Clause | **Best default** for CC/ACC/PCC/PACC/EM benchmarking and quantification-oriented validation. |
| QuantificationLib | Maintained software library | https://github.com/AICGijon/quantificationlib | Baselines (`AC`), bag generators, metrics, binary/multiclass methods | Pushed 2024-04-06; GPL-3.0 | Good reference, but GPL may be less convenient for this project than QuaPy. |
| LeQua 2022 datasets | Benchmark dataset | https://doi.org/10.5281/zenodo.6546188 | 13 ZIP/readme files: T1A/T1B/T2A/T2B train_dev, test, test prevalences | Zenodo 2022-05-14 | Large textual quantification benchmark; use for stress-testing prevalence estimators under shift. |
| LeQua 2024 datasets | Benchmark dataset | https://doi.org/10.5281/zenodo.11661820 | 13 files: T1 binary, T2 multiclass, T3 ordinal, T4 covariate-shift binary | Zenodo 2024-06-17 | Especially useful because T4 explicitly covers binary quantification under covariate shift. |
| LeQua 2024 scripts | Benchmark evaluation scripts | https://github.com/HLT-ISTI/LeQua2024_scripts | Format checker and official evaluation script for MAE/MRAE/MNMD | GitHub result 2024+ | Useful for standardized prevalence error metrics. |
| Keith & O'Connor `doc_prevalence` | Research replication archive | https://github.com/slanglab/doc_prevalence | Yelp preprocessing, training/eval scripts, ReadMe comparison scripts, notebooks | Pushed 2019-03-21 | Useful for uncertainty-aware prevalence intervals; raw Yelp data access required. |
| `freq-e` | Author software package | https://github.com/slanglab/freq-e | `FreqEstimator`, `infer_freq_from_predictions()`, example data/tutorials | Pushed 2019-09-01; PyPI package | Directly useful for prevalence intervals from classifier probabilities. |
| `pglez82/quant_datasetshift` | Research replication archive | https://github.com/pglez82/quant_datasetshift | Prior/covariate/concept shift notebooks and scripts, QuaPy experiments | Pushed 2023-10-09 | Good stress-test designs for deployment shift; less central than QuaPy. |
| `facebookresearch/multicalibrated_llm_measurement` | Research replication archive | https://github.com/facebookresearch/multicalibrated_llm_measurement | Simulation, ACS, CAP LLM prevalence analysis, paper source; included CAP inference output/sample | Created 2026-04; no releases; CC BY-NC | Frontier reference for calibration-under-covariate-shift; not a simple library, but methods can inform subgroup calibration/audits. |
| MCGrad | Calibration library dependency | https://github.com/facebookincubator/MCGrad / https://mcgrad.dev | Multicalibration algorithm used by 2026 preprint | Referenced by paper/README | Use only after simpler calibration/PCC pipeline is stable; may help subgroup prevalence bias. |

## Social-Science Prevalence Datasets and Software

| Artifact | Type | Stable URL / DOI | Contents | Access status | Implementation relevance |
|---|---|---|---|---|---|
| Hopkins & King (2010) replication data | Legacy research archive | http://hdl.handle.net/1902.1/12898 | AJPS replication data/corpora per article metadata; UNF noted in Wiley page | Public legacy handle; file manifest not API-inspected | Historical benchmark for direct aggregate text prevalence. |
| ReadMeV1 | Author-hosted software | https://github.com/iqss-research/ReadMeV1 | R package structure, `R/`, `demo/`, `inst/`, `man/` | Open GitHub; no license detected | Useful conceptual baseline; aging dependency stack. |
| readme2 / improved ReadMe | Maintained-ish software | https://github.com/iqss-research/readme-software | R package, `data/`, `results/`, `readme.pdf`, instructions | Open GitHub; pushed 2026-06-01; no license detected | Stronger social-science direct prevalence baseline; requires R/TensorFlow/reticulate stack. |
| Keith & O'Connor Yelp prevalence | Research data workflow | https://github.com/slanglab/doc_prevalence | Scripts recreate Yelp review groups; raw Yelp Challenge data external | Partly open; raw Yelp terms restrict redistribution | Useful for CI/coverage evaluation over natural document groups. |
| CAP + ACS in multicalibrated LLM prevalence | Research replication data/workflow | https://github.com/facebookresearch/multicalibrated_llm_measurement | CAP inference output/sample included; CAP raw via Comparative Agendas Project; ACS via folktables | Open but CAP/ACS download dependencies; CC BY-NC repo | Closest 2026 example of population-shift prevalence with LLM/classifier labels. |
| PPI datasets | Benchmark/replication datasets | https://doi.org/10.5281/zenodo.8397451 | `census_healthcare.npz`, `census_income.npz`, `ballots.npz`, etc. | Open Zenodo | Useful examples for label+prediction+unlabeled inference workflows. |

## Econometric Inference / Misclassification-Correction Packages

| Artifact | Type | Stable URL / DOI | Contents | Update signal | Implementation relevance |
|---|---|---|---|---|---|
| `ppi_py` | Maintained software library + replication package | https://github.com/aangelopoulos/ppi_py ; https://doi.org/10.5281/zenodo.8403931 | `ppi_py/ppi.py`, `cross_ppi.py`, `baselines.py`, datasets loader, examples, tests | Pushed 2026-04-10; latest PyPI 0.2.3; MIT | **Best default** for prediction-powered CIs for prevalence/means and downstream regressions using audit labels. |
| PPI datasets | Research dataset archive | https://doi.org/10.5281/zenodo.8397451 | 8 `.npz` datasets | Zenodo 2023-10-02 | Use as templates for storing predictions + labels + unlabeled predictions. |
| Hausman, Abrevaya & Scott-Morton (1998) | Paper only located | https://doi.org/10.1016/S0304-4076(98)00015-3 | No replication package found in searched repositories | Not found | Use formulas/conceptual correction; no reusable code found. |
| Meyer & Mittag (2017) | Paper only located | https://doi.org/10.1016/j.jeconom.2017.06.012 | No replication package found in searched repositories | Not found | Use as econometric reference for binary-choice misclassification; no package found. |
| General imperfect-test prevalence code | Adjacent software | e.g. https://github.com/BfRstats/bayespem-validation-code ; https://github.com/kalilizhou/BiasCorrection.git | R/GitHub code for prevalence under imperfect tests / biased sampling | Not central to priority list | Useful only if modeling sensitivity/specificity like diagnostic testing. |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | Exact priority titles; `classifier calibration`; `quantification PACC`; `Hopkins King ReadMe` | 0 exact relevant API hits | Hopkins-King legacy handle found via article/web, not Dataverse API. |
| OpenICPSR | `site:openicpsr.org/openicpsr` queries for Hopkins-King, PPI, Meyer-Mittag, calibration, quantification | 0 exact relevant hits | Returned unrelated computational social-science/economics packages. |
| OSF | Exact title filters for all priority papers | 0 relevant public nodes | OSF not a major source for this scope. |
| Zenodo | Exact/keyword queries listed above | 6 relevant records | LeQua 2022/2024, PPI data/code, PyCalib, Re-Assessing CC. |
| GitHub | Code signatures and repository queries listed above | 15+ relevant repositories | Main source for maintained libraries and research code. GitHub API rate-limited late; webfetch fallback used for 2026 Meta repo. |
| PMLR/ACL/NeurIPS/artifact pages | Publisher/proceedings web searches for Guo, Desai-Durrett, Mukhoti, Keith-O'Connor | Several code links | ACL/Paper pages provided Keith-O'Connor code links; Guo code found through author GitHub. |
| Author pages | Gary King, SLANG Lab, Katherine Keith, PPI paper page | 5 relevant links | Important for ReadMe and `freq-e`. |

## Papers With No Package Found

- Platt (1999), *Probabilistic Outputs for SVMs* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. Author website not deeply inspected. No replication archive found; method is implemented in `sklearn.calibration.CalibratedClassifierCV` and `netcal.scaling.LogisticCalibration`.
- Zadrozny & Elkan (2002), *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No original replication archive found; isotonic available in scikit-learn/netcal.
- Forman (2005, 2008), quantification papers — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No author replication archive found; methods implemented in QuaPy/QuantificationLib.
- Saerens et al. (2002), EM prior adjustment — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No original replication archive found; EMQ/SLD implemented in QuaPy.
- Bella et al. (2010), PCC/PACC — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No original replication archive found; PACC implemented in QuaPy/QuantificationLib.
- Gonzalez et al. (2017), quantification survey — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No replication archive expected/found; survey points to method implementations.
- Silva Filho et al. (2023), calibration survey — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No replication archive found; use survey as literature guide.
- Desai & Durrett (2020), transformer calibration — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No standalone official replication archive located in this pass; transformer calibration methods are implementable via standard calibration libraries.
- Hausman, Abrevaya & Scott-Morton (1998) and Meyer & Mittag (2017) — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No packages found; use as econometric theory references.

## Recommended Reuse for This Project

1. **Calibration metrics and curves**: start with in-project lightweight implementation using `sklearn.metrics.brier_score_loss`, `log_loss`, and `calibration_curve`; add `netcal` for ECE/MCE and reliability diagrams if Python 3.13 compatibility is acceptable. For publication-grade reliability diagrams, evaluate Apple `relplot`/SmoothECE.
2. **Post-hoc calibration**: compare no calibration, Platt/logistic scaling, temperature scaling, and isotonic regression. Use scikit-learn for Platt/isotonic on probabilities; use `gpleiss/temperature_scaling` pattern or `netcal.scaling.TemperatureScaling` for logits.
3. **Vector/matrix/Dirichlet calibration**: do not use initially for binary mission classification. If multiclass/multilabel labels are introduced, test `probmetrics`, PyHealth Dirichlet patterns, or PyCalib.
4. **PCC/ACC/PACC and EM prior shift**: use **QuaPy** as the main benchmark library because it directly implements CC/ACC/PCC/PACC and EMQ/SLD, with evaluation protocols and LeQua loaders.
5. **Bootstrapped prevalence CIs**: implement a project-native bootstrap over documents plus calibration/audit strata; compare against `freq-e` for uncertainty-aware document-class prevalence.
6. **Prediction-powered inference / audit-label correction**: use **`ppi_py`** for mean/prevalence confidence intervals and downstream regression correction where you have a gold audit sample plus predictions on the population.
7. **Misclassification correction**: for hard-label prevalence, implement ACC/Rogan-Gladen with validation-estimated TPR/FPR and bootstrap. For regressions with predicted labels, prefer PPI or explicit sensitivity analysis over plug-in hard labels.
8. **Shift/subgroup calibration**: use the 2026 multicalibrated LLM prevalence repository as a design reference for subgroup/multicalibration audits, especially if prevalence is reported by state/NTEE/year/source. Do not adopt MCGrad until simpler calibration and audit-label workflows are stable.

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| LeQua 2022 datasets | Zenodo | Learning to Quantify / LeQua 2022 | ZIP + ReadMe | T1A/T1B/T2A/T2B train/dev/test/prevalence files | Open benchmark dataset | https://zenodo.org/api/records/6546188/files/ReadMe.txt/content |
| LeQua 2024 datasets | Zenodo | Learning to Quantify / LeQua 2024 | ZIP + ReadMe | T1 binary, T2 multiclass, T3 ordinal, T4 covariate-shift binary | Open benchmark dataset | https://zenodo.org/api/records/11661820/files/ReadMe.txt/content |
| PPI datasets | Zenodo | Prediction-Powered Inference | `.npz` | census, plankton, alphafold, ballots, gene expression, galaxies, forest | Open replication dataset | https://zenodo.org/api/records/8397451/files/census_income.npz/content |
| Hopkins-King corpora / replication files | Murray Research Archive | Hopkins & King (2010) | Legacy archive files | Blog/opinion/text corpora and replication materials per article metadata | Public legacy / web-only | http://hdl.handle.net/1902.1/12898 |
| Yelp review groups workflow | GitHub + external Yelp dataset | Keith & O'Connor (2018) | JSON/scripts; raw Yelp external | Business-level sentiment prevalence groups | Partly open; raw Yelp external | https://github.com/slanglab/doc_prevalence/tree/master/yelp_data |
| CAP inference output and sample | GitHub | Unbiased Prevalence Estimation with Multicalibrated LLMs (2026) | CSV/inference outputs | 30K CAP sample with Claude Opus scores per README | Open GitHub, CC BY-NC; raw CAP external | https://github.com/facebookresearch/multicalibrated_llm_measurement/tree/main/cap_analysis/data |
| ACS via folktables | GitHub workflow | Unbiased Prevalence Estimation with Multicalibrated LLMs (2026) | Downloaded by Python package | Employment prevalence by U.S. state | Public data via folktables/Census | https://github.com/facebookresearch/multicalibrated_llm_measurement/tree/main/acs_analysis |
| freq-e example data | GitHub | Keith & O'Connor software | Example data | Tutorial-scale prevalence examples | Open | https://github.com/slanglab/freq-e/tree/master/example_data |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| Guo et al. (2017) | Python/PyTorch | `temperature_scaling.py`, `demo.py`, `train.py` | Temperature scaling, ECE loss | https://github.com/gpleiss/temperature_scaling | Open | https://raw.githubusercontent.com/gpleiss/temperature_scaling/master/temperature_scaling.py |
| Mukhoti et al. (2020) | Python/PyTorch | `Losses/`, `Metrics/`, `train.py`, `evaluate.py`, `temperature_scaling.py` | Focal loss calibration, ECE, temperature scaling | https://github.com/torrvision/focal_calibration | Open | https://raw.githubusercontent.com/torrvision/focal_calibration/main/temperature_scaling.py |
| netcal | Python | `netcal/`, `examples/`, `README.md` | Logistic/temperature/beta calibration; ECE/MCE/reliability diagrams | https://github.com/EFS-OpenSource/calibration-framework | Open | https://raw.githubusercontent.com/EFS-OpenSource/calibration-framework/main/README.md |
| Apple relplot | Python/Jupyter | `src/`, `notebooks/paper_experiments.ipynb` | SmoothECE, reliability diagrams | https://github.com/apple/ml-calibration | Open | https://github.com/apple/ml-calibration/tree/main/src |
| QuaPy | Python | `quapy/`, `examples/`, `setup.py` | CC, ACC, PCC, PACC, EMQ/SLD, quantification metrics | https://github.com/HLT-ISTI/QuaPy | Open | https://github.com/HLT-ISTI/QuaPy/tree/master/quapy |
| QuantificationLib | Python | `quantificationlib/`, `examples/`, `tests/` | AC and other binary/multiclass quantifiers | https://github.com/AICGijon/quantificationlib | Open (GPL-3.0) | https://github.com/AICGijon/quantificationlib/tree/main/quantificationlib |
| Keith & O'Connor (2018) | Python/shell/Jupyter | `code/train_all/`, `code/eval/`, `graphs/*.ipynb` | Prevalence intervals and generative prevalence experiments | https://github.com/slanglab/doc_prevalence | Open | https://github.com/slanglab/doc_prevalence/tree/master/code |
| freq-e | Python/Jupyter | `py/`, `py_tutorial/`, `tests/` | Class frequency/prevalence estimation from predictions | https://github.com/slanglab/freq-e | Open | https://github.com/slanglab/freq-e/tree/master/py |
| Hopkins & King ReadMe | R/Python | `R/`, `demo/`, `inst/` | Direct nonparametric content-analysis prevalence estimator | https://github.com/iqss-research/ReadMeV1 | Open | https://github.com/iqss-research/ReadMeV1/tree/master/R |
| readme2 | R | `readme/`, `data/`, `results/`, `useInstructions.R` | Improved ReadMe direct prevalence estimator | https://github.com/iqss-research/readme-software | Open | https://raw.githubusercontent.com/iqss-research/readme-software/master/useInstructions.R |
| Angelopoulos et al. (2023) | Python | `ppi_py/ppi.py`, `cross_ppi.py`, `baselines.py`, `examples/` | Prediction-powered inference CIs and point estimates | https://github.com/aangelopoulos/ppi_py | Open | https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/ppi.py |
| Multicalibrated LLM prevalence (2026) | Python | `simulation/run_simulation.py`, `acs_analysis/run_acs.py`, `cap_analysis/opus/run_cap_opus.py`, `cap_analysis/llama/run_llama.py` | Multicalibrated prevalence under covariate shift | https://github.com/facebookresearch/multicalibrated_llm_measurement | Open (CC BY-NC) | https://github.com/facebookresearch/multicalibrated_llm_measurement |
| Binary quantification and dataset shift | Python/Jupyter | `reviews_prior_shift.py`, `reviews_covariate_local.py`, analysis notebooks | Quantification under prior/covariate/concept shift | https://github.com/pglez82/quant_datasetshift | Open | https://raw.githubusercontent.com/pglez82/quant_datasetshift/master/reviews_prior_shift.py |
