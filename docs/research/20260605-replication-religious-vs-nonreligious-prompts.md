---
created: 20260605
agent: replication-seeker
scratchpad: docs/research/notebooks/20260605-replication-religious-vs-nonreligious-prompts-scratchpad.md
status: complete
title: Replication - Religious vs Nonreligious Prompts
topic: religious mission prompts; faith-based codebooks; survey religion wording; nonprofit religious classifiers; religion lexicons
---

# Replication: Religious vs Nonreligious Prompt and Codebook Artifacts

## Search Strategy and Exact Queries Used

### Local upstream context

Read:

- `docs/research/20260605-literature-synthesis-map.md`
- `docs/research/20260605-literature-religious-identity-prompts.md`
- `docs/research/20260605-literature-religious-vs-nonreligious-mission-prompts.md`
- `docs/research/20260605-literature-religious-nonprofit-classification.md`

### Archive/API queries

Searched Dataverse, OSF, and Zenodo APIs with:

1. `religious organization classification prompt codebook`
2. `faith-based organization codebook religious mission`
3. `religious nonprofit mission classification`
4. `Nonprofit Mission Classifiers`
5. `GivingTuesday religious_org_v1`
6. `National Congregations Study codebook religion`
7. `GSS religion codebook`
8. `World Values Survey religion questionnaire`
9. `religion lexicon dictionary text classification`
10. `Codebook LLMs replication`
11. `Language Models in the Loop weak supervision prompt`
12. `What's in a Prompt Atreja replication`

OpenICPSR/web queries:

- `site:openicpsr.org "religious" "codebook" "replication" "text classification"`
- `site:openicpsr.org "990 Mission" Paxton Velasco Ressler glossary stemmer`
- `site:openicpsr.org "National Congregations Study" ICPSR 3471 codebook`
- `site:openicpsr.org "faith-based organizations" replication data codebook`

GitHub/Hugging Face/web queries/actions:

- Fetch `https://huggingface.co/GivingTuesday/religious_org_v1`
- Fetch `https://huggingface.co/datasets/GivingTuesday/religious_orgs_training`
- Fetch `https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/`
- Fetch `https://github.com/fjsantam/bespoke-npo-taxonomies`
- Search `"Replication Data for: Bespoke NPO Taxonomies" Harvard Dataverse DOI`
- Search `"Codebook LLMs" GitHub replication prompt codebook Halterman Keith`
- Search `"What's in a Prompt" Atreja GitHub prompts dataset ICWSM 2025 replication`
- Search `"Language Models in the Loop" "weak supervision" GitHub prompt labeling functions`
- Search `"Best Practices for Text Annotation with Large Language Models" Tornberg prompt JSON uncertain annotator example`

## Verified Packages

| Paper / Artifact | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|---|---|---|---|---|---|---|
| GivingTuesday Religious Orgs Segmentation Model | Hugging Face model repo | https://huggingface.co/GivingTuesday/religious_org_v1 | Python / Transformers / MLflow metadata | Model weights/config/tokenizer; linked training dataset | Open, Apache-2.0 | 14 model files listed |
| GivingTuesday Religious Orgs Training Dataset | Hugging Face dataset repo | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Parquet / HF Datasets | 498 visible rows; EIN/name/mission/activity/classification | Open, Apache-2.0 | 3 files listed |
| Santamarina, Lecy & van Holm, *How to Code a Million Missions* / Bespoke NPO Taxonomies | GitHub + Harvard Dataverse | https://github.com/fjsantam/bespoke-npo-taxonomies; https://fjsantam.github.io/bespoke-npo-taxonomies/ | R/RMarkdown, Python utility files | IRS 1023-EZ raw/cleaned data, DFM/RData, bootstrap outputs, Paxton dictionaries | Open; Dataverse API file listing blocked by 403, GitHub mirror accessible | 100+ files in recursive GitHub tree |
| Nonprofit Mission Classifiers | GitHub Pages / NODC docs | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | R/quanteda vignettes | Training dataset description, NTEE/IRS purpose code classifiers, model assessment | Open docs | Web docs + linked GitHub/data |
| Paxton, Velasco & Ressler 990 Mission Glossary/Stemmer | GitHub mirror in Bespoke NPO Taxonomies; author page cited | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/data/step-01-raw-data/paxton | Python preprocessing; CSV/XLSX/DOCX | `glossaryv1.csv`, `Mission_Stemmer_v1.csv`, codebook DOCX | Open via GitHub mirror | 8 duplicated Paxton-path files visible |
| National Congregations Study cumulative data/codebooks | NCS / ARDA / ICPSR | https://www.nationalcongregationsstudy.org/data-documentation; https://www.icpsr.umich.edu/web/ICPSR/studies/3471 | Survey/codebook files | Cumulative codebook, wave questionnaires, ARDA/ICPSR data | Open + restricted NORC sensitive variants | Multiple questionnaires/codebook/data links |
| Pew 2023-24 Religious Landscape Study Public Use File | OpenICPSR | https://www.openicpsr.org/openicpsr/project/221062/version/V1/view | Survey data package | Public-use ZIP, license terms | OpenICPSR web-only | 2 listed key files |
| NYU Science & Religion Survey | OpenICPSR | https://www.openicpsr.org/openicpsr/project/208021/version/V2/view | Stata / codebook | Codebook PDF, Stata data | OpenICPSR web-only | 2 files |
| NaNDA Civic, Social, and Religious Organizations 1990-2021 | OpenICPSR | https://www.openicpsr.org/openicpsr/project/207966/view | CSV, Stata, PDF | Religious/civic/social org counts by tract/ZCTA | OpenICPSR web-only | 6 listed key files |
| Religious hate speech taxonomy/data | GitHub + Zenodo badge | https://github.com/dhfbk/religious-hate-speech | Python, notebooks, shell | English/Italian tweet IDs and labels; aggregated/disaggregated | Open, MIT | Landing page lists code/data directories + 4 TSV data files |
| Indonesian religiolect classifier | GitHub + Hugging Face | https://github.com/dansachs/indo-religiolects; https://huggingface.co/datasets/dansachs/indonesian-religious-corpus | Python, notebooks, Transformers | ~3M sentences; Islam/Catholic/Protestant labels; model | Open for academic research; external HF dataset/model | Repo landing page lists 6 dirs + HF assets |
| HebID social identity corpus | GitHub | https://github.com/guymorlan/hebid | CSV dataset | Hebrew political sentences with identity labels incl. Ultra-orthodox | Open, CC-BY-4.0 | 4 CSV files + README/LICENSE |
| Codebook LLMs | Cambridge/Political Analysis; planned encrypted Dataverse/code | https://www.cambridge.org/core/journals/political-analysis/article/codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts/7B323A0E47F782F2698A0AE849EA00DE | LLM eval code (replication materials mention `behavioral_tests.py`) | Three political-science codebook datasets | Publication says materials released/replication; access not fully verified | Not fully listed |
| Language Models in the Loop / prompted weak supervision | ACM / arXiv; Alfred adjacent GitHub | https://doi.org/10.1145/3617130; https://github.com/BatsResearch/alfred | Python | Prompted labeling-function framework; WRENCH benchmark usage | Article open text; code via adjacent Alfred system | Article, not full package manifest |

## Prompt / Codebook Artifacts

| Artifact | Stable URL / DOI | Contents | Reusable wording / criteria | Access / update signal |
|---|---|---|---|---|
| GivingTuesday model prompt intent | https://huggingface.co/GivingTuesday/religious_org_v1 | Model card, examples, linked notebooks | “name, mission statement and key activities” are given to GPT-4 to find “mentions/wording/terminology that reveal an org’s religious affiliations.” Definition: “Religious organizations are organizations whose identity and mission are derived from a religious or spiritual tradition and which operate as registered or unregistered, nonprofit, voluntary entities.” | Open; last modified 2025-12-17 |
| GivingTuesday dataset examples | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Nonprofit name/mission/activity/classification examples | Positive examples include “Bible studies,” “Music Ministry, promoting the Gospel,” “Christian worldview,” “religious, cultural and charitable activities,” “Jewish values,” “Catholic health care system,” “pastoral care.” Negative boundary example: boilerplate 501(c)(3) “charitable, religious, educational...” not necessarily positive. | Open; 498 visible rows; updated 2025-11-18 |
| Sider & Unruh typology | https://doi.org/10.1177/0899764003257494 | FBO construct typology | Faith-permeated; faith-centered; faith-affiliated; faith-background; faith-secular partnership; secular. Use “tangibly expressive ways that religion may be manifest” across mission, founding, affiliation, board, staff, support, practices, environment, program content, integration, outcomes. | Article; no public replication package found |
| Smith & Sosin faith-related agencies | https://doi.org/10.1111/0033-3352.00137 | FBO construct theory | Coupling to faith via resources, authority, and culture. | Article; no public package found |
| Ebaugh/Chafetz/Pipes FBO indicators | https://doi.org/10.1111/1468-5906.00191; https://doi.org/10.1353/sof.2006.0086 | Religiosity indicators | Visible religious symbols; religiously explicit mission/materials; prayer or religious material with clients/staff; proselytizing; staff understand work as religious purpose. | Article; author/project materials not fully archived |
| Codebook LLM format | https://www.cambridge.org/core/journals/political-analysis/article/codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts/7B323A0E47F782F2698A0AE849EA00DE | Codebook-to-prompt design | Structure each label with label, definition, clarification, negative clarification, positive/negative examples, task instruction, output reminder. Run behavioral tests for valid labels, definition recall, example-label recall, and order sensitivity. | Article; replication materials referenced |
| Törnberg LLM annotation template | https://doi.org/10.6092/issn.1971-8853/19461 | Prompt best practices | Prompt should contain context, question, constraints; give “Uncertain”; use JSON. Example: “As an expert annotator... Does the message contain misinformation...? Provide your response in JSON... Yes/No/Uncertain... justification...” | Open article / arXiv |
| Prompted weak supervision | https://doi.org/10.1145/3617130 | LLM prompts as labeling functions | Prompt template + label map. Example prompt type: “Does the following comment ask the user to click a link?” Map “Yes/True” to class, other outputs to abstain. | Article; Alfred adjacent repo |

## Survey Instruments and Codebooks

| Source | Stable URL / DOI | Contents | Exact wording / variables | Access Type |
|---|---|---|---|---|
| Pew religion measurement FAQ / RLS | https://www.pewresearch.org/religion/2018/07/05/how-does-pew-research-center-measure-the-religious-composition-of-the-u-s-answers-to-frequently-asked-questions/; https://doi.org/10.3886/E221062V1 | Question wording, denominational categories, public-use file | “What is your present religion, if any? Are you Protestant, Roman Catholic, Mormon, Orthodox such as Greek or Russian Orthodox, Jewish, Muslim, Buddhist, Hindu, atheist, agnostic, something else, or nothing in particular?” Follow-up: “Would you describe yourself as a born-again or evangelical Christian, or not?” Denomination: “As far as your present religion, what denomination or church, if any, do you identify with most closely?” | Open web + OpenICPSR |
| GSS religion codebook | https://gss.norc.org/ | Codebooks and variables | Core item: “What is your religious preference? Is it Protestant, Catholic, Jewish, some other religion, or no religion?” Variables include RELIG, DENOM, ATTEND, PRAY, REBORN, RELPERSN, SPRTPRSN. | Open docs / data access via NORC |
| WVS Wave 8 | https://www.worldvaluessurvey.org/documents/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf | Questionnaire | Items include importance of religion/God, attendance, prayer, denomination, religious person: “Independently of whether you attend religious services or not, would you say you are: a religious person, not a religious person, an atheist?” | Open PDF |
| EVS / GESIS | https://www.gesis.org/en/european-values-study/data-and-documentation/5th-wave-2017; https://access.gesis.org/dbk/65190 | Variable reports and religion appendices | Variables include `v52_cs` / `v52`: “which religious denomination do you belong to (Q13a)” with country-specific and harmonized categories. | Open documentation; data via GESIS |
| National Congregations Study | https://www.nationalcongregationsstudy.org/data-documentation; https://www.icpsr.umich.edu/web/ICPSR/studies/3471 | Cumulative codebook, questionnaires, ARDA/ICPSR data | “Is your congregation formally affiliated with a denomination, convention, or some similar kind of association?” “Please tell me the name of your denomination or other association.” | Open docs/data; restricted variants via NORC |
| ARDA Measurements / Question Bank | https://www.thearda.com/data-archive/measurements | Religion question bank and single-item measures | Concepts: Denomination/Affiliation; Religious Tradition; Religious Self-Identification; Church Attendance; Prayer Frequency; Religion Importance; Spirituality Strength; Scripture Reading; Proselytization. Provides full question wording/category labels when selected. | Open web tool |

## Dictionaries / Lexicons / Trigger Lists

| Source | URL | Artifact type | Extracted triggers / reusable lexical domains | Access |
|---|---|---|---|---|
| IRS obsolete religious activity codes / NTEE | https://github.com/Nonprofit-Open-Data-Collective/irs-exempt-org-business-master-file/blob/master/README.md; https://nccs.urban.org/nccs/resources/ntee/ | Administrative dictionary | 001 Church/synagogue/etc.; 002 Association or convention of churches; 003 Religious order; 004 Church auxiliary; 005 Mission; 006 Missionary activities; 007 Evangelism; 008 Religious publishing activities; 029 Other religious activities. NTEE X = Religion Related / Spiritual Development. | Open |
| Paxton mission glossary/stemmer | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/data/step-01-raw-data/paxton | Mission text preprocessing dictionary | `glossaryv1.csv`, `Mission_Stemmer_v1.csv`, DOCX codebooks; used to correct and stem nonprofit mission vocabulary before classification. | Open mirror |
| GivingTuesday training examples | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Empirical trigger examples | Gospel, Bible studies, Christian worldview, discipleship, Catholic, pastoral care, Jewish values, Islam, religious/cultural/charitable activities, yeshiva, church, ministry. | Open |
| NCS/Pew denomination lists | NCS/Pew URLs above | Denominational taxonomy | Catholic, Baptist, Methodist, Lutheran, Presbyterian/Reformed, Pentecostal, Episcopal/Anglican, Church of Christ/Disciples, Congregational/UCC, Holiness, Reformed, Church of God, nondenominational; Jewish, Muslim, Buddhist, Hindu, Mormon/LDS, Orthodox. | Open docs |
| Religious hate speech taxonomy | https://github.com/dhfbk/religious-hate-speech | Annotated religious-hate labels | Not a generic religion lexicon, but useful for religion-target references and taxonomy creation. | Open MIT |
| Indonesian religiolect corpus | https://github.com/dansachs/indo-religiolects | Corpus/model-derived lexical signal | Islam/Catholic/Protestant institutional language; useful for non-English religious vocabulary and denominational discourse. | Open / academic |

### Consolidated trigger domains for weak supervision

- **Institution/identity**: church, congregation, mosque, masjid, synagogue, temple, chapel, cathedral, parish, diocese, religious order, yeshiva, monastery, seminary, ministry, faith-based, Catholic, Protestant, Baptist, Methodist, Lutheran, Presbyterian, Episcopal, Anglican, Pentecostal, Orthodox, LDS/Mormon, Jewish, Muslim/Islamic, Hindu, Buddhist, Sikh, interfaith.
- **Purpose/activity**: worship, prayer, Bible/Quran/Torah study, Gospel, scripture, discipleship, evangelism, missionary, chaplaincy, pastoral care, sacraments, religious education, faith formation, spiritual formation, religious publishing, religious counseling.
- **Theological/spiritual terms**: God, Jesus, Christ, Holy Spirit, Allah, Quran, Torah, Talmud, Dharma, Buddha, sacred, divine, holy, faith, spiritual/spirituality.
- **Negative/ambiguous filters**: boilerplate “charitable, religious, educational...” purpose clauses; “without regard to religion”; “mission” alone; “ministry” as government department in non-US contexts; saint/place names without current religious evidence; values words like service, compassion, stewardship without religious anchor.

## Datasets

| Dataset | Source Repo | Coverage / Format | Stable URL | Access Type | Update Signal |
|---|---|---|---|---|---|
| GivingTuesday Religious Orgs Training | Hugging Face | 498 visible nonprofit rows; Parquet; fields: EIN/name/mission/activity_1-3/classification | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Open | Updated 2025-11-18 |
| IRS 1023-EZ mission corpus / Bespoke NPO Taxonomies | GitHub + Dataverse | 104,072 cleaned mission docs; CSV, RData, RDS | https://github.com/fjsantam/bespoke-npo-taxonomies; https://doi.org/10.7910/DVN/BL6XLW | Open | GitHub 45 commits; Dataverse V1 |
| NCS cumulative dataset | NCS/ARDA/ICPSR | Congregations; denomination/formal affiliation/worship/program items | https://www.nationalcongregationsstudy.org/data-documentation | Open + restricted variants | Waves I-IV cumulative codebook 2020 |
| Pew RLS 2023-24 Public Use File | OpenICPSR | 36,000+ respondents; ZIP 208.6 MB | https://doi.org/10.3886/E221062V1 | OpenICPSR | V1 2025-02-27 |
| NaNDA religious/civic/social orgs 1990-2021 | OpenICPSR | Tract/ZCTA org counts/densities; CSV/Stata | https://doi.org/10.3886/E207966V1 | OpenICPSR | V1 2024-07-24 |
| Religious hate speech | GitHub | English/Italian tweet IDs + labels | https://github.com/dhfbk/religious-hate-speech | Open MIT | Release v1.0 2022-09-15 |
| Indonesian Religious Corpus | Hugging Face/GitHub | ~3M clean sentences from 100+ religious websites | https://huggingface.co/datasets/dansachs/indonesian-religious-corpus | Open/academic | GitHub current 2026 page |
| HebID | GitHub | 5,536 Hebrew political sentences + Knesset addendum; CSV | https://github.com/guymorlan/hebid | Open CC-BY-4.0 | EMNLP 2025 repo |

## Code / Model Repositories

| Repo | URL | Contents | Method | Access |
|---|---|---|---|---|
| GivingTuesday `religious_org_v1` | https://huggingface.co/GivingTuesday/religious_org_v1 | BERT classifier, tokenizer, MLflow metadata, example inference | GPT-4-labeled training + BERT fine-tuning | Open Apache-2.0 |
| Bespoke NPO Taxonomies | https://github.com/fjsantam/bespoke-npo-taxonomies | RMarkdown docs, data, bootstrap classifier outputs | Naive Bayes / quanteda mission-text classifiers | Open |
| Nonprofit Mission Classifiers | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Tutorials/vignettes, accuracy tables | NTEE/IRS purpose classifiers from mission/activity text | Open docs |
| Religious hate speech | https://github.com/dhfbk/religious-hate-speech | Baselines, ML, PLM notebooks, dataset IDs | Taxonomy-to-automated detection | Open MIT |
| Indo-religiolects | https://github.com/dansachs/indo-religiolects | Crawler, training, inference, interactive scripts | IndoBERT denominational classifier | Open/academic |
| HebID | https://github.com/guymorlan/hebid | CSV corpus and label definitions | Multi-label identity detection | Open CC-BY-4.0 |
| Alfred prompted weak supervision | https://github.com/BatsResearch/alfred | Prompted weak supervision system | Prompted LFs + label model | Open GitHub |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | 12 API queries listed above; specific DOI lookup for `10.7910/DVN/BL6XLW`, `10.7910/DVN/EO2HIM`, `10.7910/DVN/4GZJSK` | Generic query hits: 0; specific Santamarina Dataverse DOIs found via GitHub/Pages | File API returned 403; GitHub mirror used for file manifest. |
| OpenICPSR | `site:openicpsr.org` queries listed above | Pew RLS, NYU Science & Religion, NaNDA, economics religion packages found | No direct religious mission prompt/codebook package found. Web-only inspection. |
| OSF | 12 API queries listed above | 0 direct hits | No scoped packages found. |
| Zenodo | 12 API queries listed above | Broad LLM/codebook/religion hits; no direct nonprofit religious classifier | Religious-hate GitHub has Zenodo DOI badge. |
| GitHub | Hugging Face/GitHub page fetches; code searches for NODC and religion classification terms | GivingTuesday, Santamarina, NODC, dhfbk, indo-religiolects, HebID, Alfred found | GitHub API became rate-limited; landing pages and webfetch used as fallback. |

## Papers With No Package Found

- Sider & Unruh (2004), *Typology of Religious Characteristics...* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. Author website not exhaustively checked. Public article/construct wording found; no replication package located.
- Bielefeld & Cleveland (2013), *Defining Faith-Based Organizations...* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No replication package located.
- Smith & Sosin (2001/2002), *The Varieties of Faith-Related Agencies* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No replication package located.
- Ebaugh / Chafetz / Pipes FBO indicator papers — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. CMACS/project references found in upstream reports, but no reusable public replication package verified.
- Ma (2021), *Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No direct replication package found; adjacent NODC/Bespoke packages found.
- Fyall, Moore & Gugerty (2018), *Beyond NTEE Codes* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub/web. No direct replication package found; related mission coding framework and IRS/NTEE data sources found.
- Atreja et al. (2025), *What's in a Prompt?* — Searched GitHub/web/Zenodo/OSF. Article and DOI found; no public GitHub replication repo verified in this pass.
- Full original GivingTuesday GPT-4 prompt — Searched Hugging Face model/dataset cards and public docs. Prompt intent found, not exact system/user prompt text.

## Not-Found / Limited-Access Artifacts

| Artifact | Status | Repositories checked | Caveat |
|---|---|---|---|
| GivingTuesday internal Databricks notebook | Link present | Hugging Face card | Databricks workspace likely internal; not accessible as public replication. |
| GivingTuesday exact GPT-4 prompt | Partial | Hugging Face, web | Only operational intent published. |
| Sider/Unruh annotation instrument | Not found | Dataverse, OSF, Zenodo, OpenICPSR, GitHub/web | Article has typology; not a package. |
| Bielefeld/Cleveland codebook/data | Not found | Dataverse, OSF, Zenodo, OpenICPSR, GitHub/web | Construct review article. |
| Ebaugh/Chafetz/Pipes survey instrument/codebook | Partial/not verified | Web/OpenICPSR/archives | Indicator wording found in literature; no open manifest verified. |
| Ma/Fyall direct replication materials | Not found | Dataverse, OSF, Zenodo, OpenICPSR, GitHub/web | Adjacent mission-classifier packages are usable substitutes. |

## Recommended Prompt / Codebook Domains for This Project

Use a multi-label codebook and derive the binary religious/nonreligious label only after evidence capture.

| Domain | Positive rule | Negative / ambiguous rule | Evidence-span requirement |
|---|---|---|---|
| `religious_purpose_explicit` | Mission/purpose states worship, ministry, evangelism, religious education, faith formation, discipleship, Gospel, scripture, prayer, God/Christ/Allah/Torah/Quran/Dharma-linked purpose. | Do not count generic “mission,” “serve,” “values,” “compassion,” “community,” or boilerplate “charitable, religious, educational” without a religious anchor. | Quote exact phrase from name/mission/activity. |
| `religious_identity_or_affiliation` | Name/text identifies church, mosque, synagogue, temple, congregation, denomination, diocese, religious order, Catholic/Lutheran/Baptist/Islamic/Jewish/Hindu/Buddhist/Sikh/etc. | Saint names, historic founders, hospitals/schools with religious names are review unless current affiliation or purpose is explicit. | Quote name phrase and any affiliation phrase. |
| `religious_service_content` | Prayer, Bible/Quran/Torah study, chaplaincy, pastoral care, worship, sacraments, religious counseling, missionary activities, evangelism, religious publishing/schooling. | Secular services by a historically religious organization should not be activity-positive unless service content itself is religious. | Quote program/activity text. |
| `religious_governance_or_authority` | Board/staff/clergy selected by faith body; sponsored by diocese/order/congregation; formal denomination/convention affiliation. | If absent, code unknown, not negative. | Quote governance/affiliation text if available. |
| `spiritual_or_faith_inspired_ambiguous` | Spirituality, sacred, faith-inspired, meditation, healing, Mother Earth, “faith in action,” historic roots without named tradition or practice. | Keep separate from strict binary positive unless co-occurring with explicit religious identity/activity. | Quote ambiguous term and context. |
| `administrative_religion_prior` | NTEE X/REL, IRS religious purpose, old IRS activity codes 001-008/029, church filing exemption/status. | Treat as weak prior, not ground truth. | Record code/source field. |
| `negative_or_nonreligious_evidence` | “No religion,” atheist, agnostic, secular, humanist, freethought; “without regard to religion”; purely civic/education/health/housing mission with no religious evidence. | Nonreligious identity can be a positive class only if the research question is nonreligion; for religious-org binary it is negative. | Quote negative/secular phrase or state “no explicit religious evidence.” |

### Derived binary rule

- **Positive** if `religious_purpose_explicit`, `religious_service_content`, or high-confidence `religious_identity_or_affiliation` has an evidence span.
- **Probable positive / review** if only governance/network/administrative prior is present.
- **Ambiguous / audit** if only spiritual language, historical roots, saint names, generic faith/values, or faith-based partner references appear.
- **Negative** if no explicit religious/spiritual evidence appears or text explicitly says secular/nonreligious/without regard to religion.

### LLM prompt skeleton to adapt

```text
You are an expert nonprofit-text annotator. Classify only observable religious or spiritual mission/expression in the provided organization text. Do not infer latent religiosity, morality, or founder intent.

Use only these fields: organization name, mission/purpose, program/activity descriptions, history/affiliation text, and administrative codes if provided.

Return JSON with:
{
  "binary_religious": "positive|negative|probable_positive|ambiguous|uncertain",
  "domains": [list of domain labels],
  "evidence_spans": [exact quoted phrases],
  "negative_or_ambiguity_notes": "brief note",
  "confidence": "high|medium|low"
}

If evidence is absent or unclear, choose negative or uncertain; do not guess. Boilerplate legal purpose clauses and generic values words are insufficient without a religious anchor.
```

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| GivingTuesday Religious Orgs Training | Hugging Face | GivingTuesday `religious_org_v1` | Parquet | 498 visible nonprofit examples; U.S. nonprofit name/mission/activity/classification | Open | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training/tree/main/training_data |
| IRS 1023-EZ raw/cleaned mission data | GitHub / Dataverse | Santamarina et al. | CSV/XLSX | 2018-2019 1023-EZ approvals with mission text | Open | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/data/step-01-raw-data |
| Paxton mission glossary/stemmer | GitHub mirror | Paxton, Velasco & Ressler | CSV/XLSX/DOCX | Nonprofit mission glossary/stemmer | Open | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/data/step-01-raw-data/paxton |
| NCS cumulative dataset | NCS / ARDA / ICPSR | Chaves et al. | Survey data/codebook | Congregations, denomination/formal affiliation/worship/program variables | Open + restricted variants | https://www.nationalcongregationsstudy.org/data-documentation |
| Pew RLS 2023-24 PUF | OpenICPSR | Pew RLS | ZIP | 36,000+ U.S. adults; religion identity/practice/belief | OpenICPSR | https://doi.org/10.3886/E221062V1 |
| NaNDA Religious/Civic/Social Organizations | OpenICPSR | NaNDA | CSV/Stata/PDF | U.S. tract/ZCTA organization counts, 1990-2021 | OpenICPSR | https://doi.org/10.3886/E207966V1 |
| Religious hate speech dataset | GitHub | Ramponi et al. 2022 | TSV tweet IDs/labels | English/Italian religious hate labels | Open MIT | https://github.com/dhfbk/religious-hate-speech/tree/main/data |
| Indonesian Religious Corpus | Hugging Face | Sachs et al. / Indo-religiolects | CSV | ~3M sentences from Indonesian religious websites | Open/academic | https://huggingface.co/datasets/dansachs/indonesian-religious-corpus |
| HebID | GitHub | Morlan et al. 2025 | CSV | Hebrew political text identity labels | Open CC-BY-4.0 | https://github.com/guymorlan/hebid |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| GivingTuesday `religious_org_v1` | Python / Transformers | `config.json`, `model.safetensors`, tokenizer files, `metadata/MLmodel`, `metadata/requirements.txt` | BERT sequence classification | https://huggingface.co/GivingTuesday/religious_org_v1 | Open Apache-2.0 | https://huggingface.co/GivingTuesday/religious_org_v1/tree/main |
| Santamarina et al. / Bespoke NPO Taxonomies | R/RMarkdown/Python | `docs/step-01-data-preprocessing.Rmd`, `docs/old/Classification_Bootstrapping_Replication.Rmd`, `data/model-accuracy.csv` | quanteda Naive Bayes and bootstrap classification | https://github.com/fjsantam/bespoke-npo-taxonomies | Open | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/docs |
| Paxton mission preprocessing | Python/CSV | `00.data_prep_glossary.py`, `glossaryv1.csv`, `Mission_Stemmer_v1.csv` | Dictionary/stemmer preprocessing | https://github.com/fjsantam/bespoke-npo-taxonomies | Open | https://github.com/fjsantam/bespoke-npo-taxonomies/tree/main/data/step-01-raw-data/paxton |
| Religious hate speech | Python/Jupyter | `code/baselines-and-ml-models/`, `code/pretrained-language-models/` | Baselines + PLM classifiers | https://github.com/dhfbk/religious-hate-speech | Open MIT | https://github.com/dhfbk/religious-hate-speech/tree/main/code |
| Indo-religiolects | Python/Jupyter | `run_crawler.py`, `src/`, `training/train_model.py`, `interactive/predict.py` | Crawling + IndoBERT fine-tuning/inference | https://github.com/dansachs/indo-religiolects | Open/academic | https://github.com/dansachs/indo-religiolects |
| HebID | CSV/README | `train.csv`, `val.csv`, `test.csv`, `addendum_knesset.csv`, README label definitions | Multi-label identity dataset | https://github.com/guymorlan/hebid | Open CC-BY-4.0 | https://github.com/guymorlan/hebid |
| Alfred prompted weak supervision | Python | Prompted LF and label-model system files | Prompted weak supervision | https://github.com/BatsResearch/alfred | Open | https://github.com/BatsResearch/alfred |
