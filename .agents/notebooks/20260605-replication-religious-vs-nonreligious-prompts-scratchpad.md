# Replication Scratchpad: Religious vs Nonreligious Prompt and Codebook Artifacts

Created: 20260605
Topic Keywords: religious mission prompts; faith-based codebooks; survey religion wording; nonprofit religious classifiers; religion lexicons

---

## Problem Definition

### Research Questions

- Primary: Where are reusable replication packages, prompt repositories, codebooks, survey instruments, annotation guidelines, datasets, dictionaries, and reproducibility materials for classifying religious vs non-religious text/narratives?
- Secondary: Which artifacts contain exact wording or lexical triggers usable for this project's LLM prompt engineering, weak-supervision label functions, or human annotation instructions?

### Input Mode

- [x] DOI/URL/citations from literature-seeker
- [x] Freeform paper citations
- [x] Topic string (bounded bootstrapping)

### Success Criteria

- [x] Use upstream reports as context and keep artifact search centered on their scoped sources plus user priority sources.
- [ ] Check Dataverse, OpenICPSR, OSF, Zenodo, and GitHub/Hugging Face/web repositories before marking priority paper/artifact families as missing.
- [ ] Inspect accessible package contents and capture file manifests or stable artifact locations where feasible.
- [ ] Extract directly reusable wording/triggers from accessible codebooks/model cards/dataset cards.

## Research Log

### Entry 1: Upstream context and bounded scope

THOUGHT: I need to avoid an unbounded web survey and first use existing project reports that already mapped relevant literature and artifact leads.
ACTION: Read `.agents/docs/20260605-literature-synthesis-map.md`, `.agents/docs/20260605-literature-religious-identity-prompts.md`, `.agents/docs/20260605-literature-religious-vs-nonreligious-mission-prompts.md`; extracted handoff citation lists.
OBSERVATION:

- Source: local literature reports.
- Repository: Local project docs.
- Key Facts: Priority leads include GivingTuesday `religious_org_v1` and `religious_orgs_training`, Sider & Unruh, Bielefeld & Cleveland, Smith & Sosin, Ebaugh/Chafetz/Pipes, Ma, Fyall et al., Santamarina et al./Nonprofit Mission Classifiers, Pew/GSS/WVS/EVS/NCS/ARDA, and adjacent LLM/prompt-method papers.
- Confidence: High because local reports are complete and user explicitly requested them as context.
- New Questions: Which of these have public replication packages or reusable machine-readable codebooks/prompts?

### Entry 2: GivingTuesday religious organization model/dataset

THOUGHT: The closest artifact is a prompt-labeled religious nonprofit classifier and training dataset, so I need model card, dataset card, file list, prompt wording, and update/license signals.
ACTION: Fetched Hugging Face model and dataset pages/API for `GivingTuesday/religious_org_v1` and `GivingTuesday/religious_orgs_training`.
OBSERVATION:

- Source: https://huggingface.co/GivingTuesday/religious_org_v1; https://huggingface.co/datasets/GivingTuesday/religious_orgs_training
- Repository: Hugging Face / model + dataset repository.
- Key Facts: Open Apache-2.0; model lastModified 2025-12-17; dataset lastModified 2025-11-18. Model files include `config.json`, `model.safetensors`, tokenizer files, MLflow metadata, requirements/conda files. Dataset contains `training_data/more_fine_tune_df.parquet`, 498 visible rows, columns `ein`, `name`, `mission`, `activity_1`, `activity_2`, `activity_3`, `classification`. Card states GPT-4 was prompted with name, mission, and key activities to identify wording/terminology revealing religious affiliations; final BERT weighted F1 0.93, macro F1 0.76.
- Confidence: High for repository contents and public model/dataset; Medium for exact original GPT-4 prompt because only intent is published, not the full hidden prompt.
- New Questions: Internal Databricks notebook and Google Docs technical specification may be partially inaccessible.

### Entry 3: Archive search coverage

THOUGHT: To avoid false negatives, I need repository-complete searches across Dataverse, OSF, Zenodo, OpenICPSR, and GitHub for the scoped artifact families.
ACTION: Searched Dataverse/OSF/Zenodo APIs with queries: `religious organization classification prompt codebook`; `faith-based organization codebook religious mission`; `religious nonprofit mission classification`; `Nonprofit Mission Classifiers`; `GivingTuesday religious_org_v1`; `National Congregations Study codebook religion`; `GSS religion codebook`; `World Values Survey religion questionnaire`; `religion lexicon dictionary text classification`; `Codebook LLMs replication`; `Language Models in the Loop weak supervision prompt`; `What's in a Prompt Atreja replication`.
OBSERVATION:

- Source: Dataverse API, OSF API, Zenodo API.
- Repository: Dataverse/OpenICPSR/OSF/Zenodo/GitHub.
- Key Facts: Dataverse/OSF returned no direct topic hits from generic API queries. Zenodo returned broad LLM/codebook and religion items but no direct religious-vs-nonreligious mission classifier except adjacent LLM-codebook records. OpenICPSR web search found Pew RLS public-use file, NYU Science & Religion Survey, NaNDA religious organization datasets, and economics religion packages; no direct prompt/codebook package for religious mission text classification.
- Confidence: Medium-High; APIs/searches are broad but exact titles may not be indexed uniformly.
- New Questions: Author websites may host FBO survey instruments for Ebaugh/Chafetz/Pipes.

### Entry 4: Santamarina / Nonprofit Mission Classifiers / Paxton materials

THOUGHT: Nonprofit mission classifiers are likely GitHub/Dataverse packages, not generic archive hits; I need manifests and direct religion-specific files.
ACTION: Fetched Nonprofit Mission Classifiers site, `fjsantam/bespoke-npo-taxonomies` GitHub/Pages, websearched Dataverse DOI, and inspected GitHub recursive tree.
OBSERVATION:

- Source: https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/; https://github.com/fjsantam/bespoke-npo-taxonomies; https://fjsantam.github.io/bespoke-npo-taxonomies/; Dataverse DOIs `10.7910/DVN/BL6XLW`, `10.7910/DVN/EO2HIM`, `10.7910/DVN/4GZJSK`.
- Repository: GitHub + Harvard Dataverse.
- Key Facts: Package includes raw/cleaned IRS 1023-EZ data, crosswalks, Paxton mission glossary/stemmer (`glossaryv1.csv`, `Mission_Stemmer_v1.csv`, codebooks), RMarkdown preprocessing/classification docs, classification-ready `.RData`, bootstrap `.rds` outputs including `Orgpurposereligious` and `ntmaj10rel`. Dataverse file API returned 403; GitHub mirror exposed file tree and sizes.
- Confidence: High for GitHub mirror; Medium for Dataverse file details because API manifest blocked.
- New Questions: Paxton original ICPSR/distributor record should be cited via author page if needed.

### Entry 5: Survey/codebook instruments

THOUGHT: The user needs exact religion-question wording and reusable codebooks; official survey providers are authoritative even if not replication packages.
ACTION: Fetched NCS data/documentation, Pew measurement FAQ, ARDA Measurements, WVS/EVS/GESIS search results, and GSS/WVS PDFs where possible.
OBSERVATION:

- Source: NCS, Pew, NORC GSS, WVS, GESIS EVS, ARDA.
- Repository: Official survey/codebook providers; ICPSR/OpenICPSR for selected public-use files.
- Key Facts: NCS page links cumulative codebook, ARDA/ICPSR data, questionnaires by wave, and restricted NORC data. Pew exposes exact wording for present religion, born-again/evangelical, denomination follow-ups, and category construction. ARDA Measurements provides question bank and single-item measures across religious identification, salience/scripture/prayer, spirituality, attendance, denomination, etc. EVS/GESIS provides variable reports and religion appendices with exact question text and harmonized denomination variables.
- Confidence: High for provider availability and Pew wording; Medium for GSS/WVS exact text due PDF extraction noise, supplemented by prior local reports.
- New Questions: none critical.

### Entry 6: NLP/codebook/lexicon repositories

THOUGHT: Religious text classification may be adjacent (hate speech, religiolects, identity labels) and useful for annotation rules/lexicons even if not nonprofit-specific.
ACTION: Fetched GitHub pages for `dhfbk/religious-hate-speech`, `dansachs/indo-religiolects`, `guymorlan/hebid`; searched Codebook LLMs, Atreja, Smith et al., Törnberg.
OBSERVATION:

- Source: GitHub pages and publication pages.
- Repository: GitHub, Hugging Face, journal pages, arXiv.
- Key Facts: `dhfbk/religious-hate-speech` has MIT-licensed code/data with English/Italian tweet IDs and labels; Zenodo DOI badge. `dansachs/indo-religiolects` has Python crawler/training/inference pipeline and Hugging Face dataset/model for Islam/Catholic/Protestant Indonesian religiolects. `guymorlan/hebid` has CC-BY-4.0 CSV train/val/test/addendum and label definitions including Ultra-orthodox, Liberal, Conservative, Zionist. Codebook LLMs provides five-component codebook format and behavioral tests; Törnberg provides structured prompt template with JSON and Uncertain option; Smith et al. provides prompted-labeling-function/label-map framework. GitHub API rate-limited after initial requests.
- Confidence: High for repository landing pages; Medium for full manifests where API was rate-limited.
- New Questions: none critical.

## Draft Output

### Packages Found

GivingTuesday HF; Santamarina/Bespoke NPO Taxonomies GitHub+Dataverse; Nonprofit Mission Classifiers docs; Paxton mission glossary/stemmer; NCS/Pew/GSS/WVS/EVS/ARDA instruments; OpenICPSR Pew RLS/NYU/NaNDA; dhfbk religious hate; indo-religiolects; HebID; Codebook LLMs/Törnberg/Smith et al. prompt-method materials.

### Repository Coverage

Dataverse: generic queries 0; specific Santamarina Dataverse DOIs found via GitHub/Pages. OSF: generic queries 0. Zenodo: broad hits; religious hate repo has Zenodo badge; no direct nonprofit religious prompt package. OpenICPSR: Pew RLS, NYU Science & Religion, NaNDA and economics religion packages found; no direct mission-text prompt codebook. GitHub/Hugging Face: multiple verified packages; GitHub API rate-limited, web landing pages used as fallback.

### Not Found

No full original GPT-4 prompt for GivingTuesday found publicly; no public replication package located for Sider & Unruh, Bielefeld & Cleveland, Smith & Sosin, or Ebaugh/Chafetz/Pipes beyond article/codebook wording and author/project pages; no direct public prompt/code repository found for Ma (2021) or Fyall et al. (2018); no direct religion-specific prompt assets found in OSF or generic Dataverse API searches.
