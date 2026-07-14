---
created: 20260606
agent: tech-seeker
scratchpad: docs/research/notebooks/20260605-tech-religious-identity-prompts-scratchpad.md
status: complete
title: Tech - Religious Identity Prompts and Textual Triggers
topic: religious identity; prompts; survey wording; nonprofit mission; lexical triggers
---

# Tech: Religious Identity Prompts, Survey Wording, and Textual Triggers

## 1. Search Strategy and Exact Queries Used

Prioritized official instruments/codebooks, peer-reviewed nonprofit/FBO typologies, mission-text classification work, and recent CSS/NLP artifacts. Seeded from Sider & Unruh, Bielefeld & Cleveland, Smith & Sosin, NCS, Pew, GSS, ARDA, WVS/EVS, and nonprofit mission classification studies.

Exact queries used:

1. `2026 GSS 2024 codebook religion variables RELIG DENOM ATTEND PRAY REBORN exact question wording`
2. `2026 Pew Research Center religious affiliation question wording born-again evangelical denomination attendance prayer present religion if any`
3. `2026 World Values Survey Wave 8 questionnaire religion religious person denomination attendance prayer exact wording`
4. `2026 National Congregations Study cumulative codebook denomination formal affiliation religious tradition question wording`
5. `Sider Unruh 2004 Typology Religious Characteristics Social Service Educational Organizations Programs mission affiliation staff prayer DOI`
6. `Bielefeld Cleveland 2013 Defining Faith-Based Organizations Understanding Them Through Research nonprofit classification religious mission DOI`
7. `Ebaugh Chafetz Pipes Where's the Religion faith-based organizations measures visible religious symbols mission prayer clients staff 2003 2006 DOI`
8. `2026 nonprofit mission statement classification religious nonprofits Form 990 NTEE religion text machine learning Ma Fyall Gugerty Moore Santamarina`
9. `computational social science NLP religious identity text classification keywords lexicon annotation religion 2020 2026`
10. GitHub/content queries: `religious_org_v1`, `Orgpurposereligious`, `religion_terms`, `(church|synagogue|mosque|temple).*religion`.

## 2. Key Academic Papers, Reports, Codebooks, and Datasets

| # | Source | Year | Venue / source | DOI / link | Why it matters |
|---|---|---:|---|---|---|
| 1 | Sider & Unruh, “Typology of Religious Characteristics…” | 2004 | Nonprofit and Voluntary Sector Quarterly | https://doi.org/10.1177/0899764003257494 | Best organization/program typology: faith-permeated, faith-centered, faith-affiliated, faith-background, faith-secular partnership, secular. |
| 2 | Smith & Sosin, “The Varieties of Faith-Related Agencies” | 2001 | Public Administration Review | https://doi.org/10.1111/0033-3352.00137 | Distinguishes faith-related auspice, resources, culture, and service technology. |
| 3 | Bielefeld & Cleveland, “Defining Faith-Based Organizations…” | 2013 | NVSQ | https://doi.org/10.1177/0899764013484090 | Reviews 600+ FBO works and definitional/methodological pitfalls. |
| 4 | Ebaugh, Pipes, Chafetz & Daniels, “Where’s the Religion?” | 2003 | JSSR | https://doi.org/10.1111/1468-5906.00191 | Mission/public-face religious imagery and org-characteristic measures. |
| 5 | Ebaugh, Chafetz & Pipes, “Where’s the Faith…?” | 2006 | Social Forces | https://doi.org/10.1353/sof.2006.0086 | Measures coalition religiosity: mission, symbols, prayer, staff/client practices. |
| 6 | Pew, “How Does Pew Measure Religious Composition?” | 2018 | Official methodology | https://www.pewresearch.org/religion/2018/07/05/how-does-pew-research-center-measure-the-religious-composition-of-the-u-s-answers-to-frequently-asked-questions/ | Canonical “present religion, if any” and born-again/denomination follow-ups. |
| 7 | Pew NPORS / RLS methodology and questionnaires | 2021–2025 | Official survey documentation | https://www.pewresearch.org/religion/2021/12/14/methodology-46-4/ | Current wording, mode changes, switching/raised religion items. |
| 8 | NORC GSS 2024 Codebook R3 | 2026 | Official codebook | https://gss.norc.org/ | RELIG, DENOM, ATTEND, PRAY, REBORN; notes 2024 web-mode wording changes. |
| 9 | WVS Wave 8 Questionnaire v11 | 2024 | Official questionnaire | https://www.worldvaluessurvey.org/documents/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf | Cross-national religion salience, God importance, attendance, prayer, denomination. |
| 10 | National Congregations Study I-IV Cumulative Codebook | 2020 | NCS/ARDA/ICPSR | https://www.nationalcongregationsstudy.org/data-documentation | Congregation affiliation, denomination, religious tradition coding. |
| 11 | Bradburn et al., ARDA Measurement Wizard | 2014 | JSSR | https://doi.org/10.1111/jssr.12131 | Repository logic for improving religion survey questions. |
| 12 | Smith & Kim, “Counting Religious Nones…” | 2007 | GSS Methodological Report | https://gss.norc.org/ | Shows wording/order effects for no religion/nones. |
| 13 | Pickel et al., “Functional Equivalence…” | 2016 | Methodological Innovations | https://doi.org/10.1177/2059799115622756 | Cross-cultural construct-validity warning. |
| 14 | Hackett & Conrad, “Changing Survey Measures…” | 2026 | Survey Practice | https://www.surveypractice.org/article/159509-changing-survey-measures-and-measuring-change-a-call-for-religion-measurement-experiments | Recent call for religion measurement experiments. |
| 15 | Fyall, Gugerty & Moore, “Beyond NTEE Codes…” | 2018 | NVSQ | https://doi.org/10.1177/0899764018768019 | Demonstrates mission statements can outperform administrative codes for activity identification. |
| 16 | Ma, “Automated Coding Using Machine Learning…” | 2021 | NVSQ | https://doi.org/10.1177/0899764020968153 | BERT/ML nonprofit NTEE classifier; NTEE X as religion-related benchmark. |
| 17 | Santamarina, Lecy & van Holm, “How to Code a Million Missions” | 2021 | VOLUNTAS / replication | https://fjsantam.github.io/bespoke-npo-taxonomies/ | Quanteda Naive Bayes over IRS 1023-EZ mission text; includes religious purpose code. |
| 18 | Nonprofit Open Data Collective Mission Classifiers | 2019–2026 | Open docs/code | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Benchmarks religious purpose: ICR .97, ML accuracy .92; NTEE religion accuracy .95. |
| 19 | GivingTuesday `religious_org_v1` | 2025–2026 | Hugging Face model card | https://huggingface.co/GivingTuesday/religious_org_v1 | Public BERT religious nonprofit classifier over name/mission/activity; useful teacher, not ground truth. |
| 20 | Chen, Weber & Okulicz-Kozaryn, “U.S. Religious Landscape on Twitter” | 2014 | SocInfo | https://ingmarweber.de/wp-content/uploads/2014/11/U.S._Religious_Landscape_on_Twitter.pdf | Self-declared religion cues in social media profiles/text. |
| 21 | Sircar et al., “It’s All in the Name” | 2023 | Political Analysis | https://www.cambridge.org/core/journals/political-analysis/article/its-all-in-the-name-a-characterbased-approach-to-infer-religion/B6A8AEE0AB1DA607B0AE1A57D869C641 | Name-based religion inference; useful but high fairness/construct caveats. |
| 22 | Manerba et al., “Addressing Religious Hate Online…” | 2023 | PeerJ Computer Science | https://pmc.ncbi.nlm.nih.gov/articles/PMC10280248/ | Annotation taxonomy and religious-group terms for NLP. |
| 23 | Sachs et al., “Indonesian Religiolect Corpus” | 2026 | ACL LoResLM | https://aclanthology.org/2026.loreslm-1.36.pdf | Institutional religious-language corpus across Muslim/Catholic/Protestant sources. |
| 24 | Morlan et al., HEBID | 2025 | Findings of EMNLP | https://github.com/guymorlan/hebid | Multilabel identity detection including religious/political identity overlaps. |

## 3. Concrete Question Wordings, Prompt Phrasings, Codebook Items, and Lexical Triggers

| Construct | Source | Concrete wording / trigger | Use in prompts, codebooks, or weak supervision |
|---|---|---|---|
| Present affiliation | Pew | “What is your present religion, if any?” with Protestant, Catholic, LDS, Orthodox, Jewish, Muslim, Buddhist, Hindu, atheist, agnostic, something else, nothing in particular. | Use “if any”; include nonreligious options. |
| Religious preference | GSS | “What is your religious preference?” | Compact U.S. affiliation item; less granular than Pew. |
| Denomination | Pew / NCS | “What denomination or church, if any, do you identify with most closely?” / “Please tell me the name of your denomination or other association.” | Require named affiliation for fine-grained tradition coding. |
| Born-again/evangelical | Pew / GSS | “Would you describe yourself as a ‘born-again’ or evangelical Christian, or not?” | Self-ID is distinct from denominational evangelical coding. |
| Congregational auspice | NCS | “Is your congregation formally affiliated with a denomination, convention, or some similar kind of association?” | Model formal control/affiliation separately from local name. |
| Attendance | Pew / WVS | “Aside from weddings and funerals, how often do you attend religious services?” | Practice/activity cue; not equivalent to affiliation. |
| Prayer | Pew / WVS | “Outside of religious services, how often do you pray?” | Practice/salience cue. |
| Salience | Pew / WVS | “How important is religion/God in your life?” | Intensity; not a label by itself. |
| Religious person | WVS | “Independently of whether you attend religious services or not, would you say you are a religious person, not a religious person, an atheist?” | Useful cross-national identity wording. |
| FBO identity dimensions | Sider & Unruh | Mission/vision, founding history, affiliation, control structures, management, staff, support, staff religious practices. | Human codebook dimensions. |
| Program religiosity | Sider & Unruh | Religious environment, program content, integration, expected religious-outcome connection. | Separate organization identity from program activity. |
| Public-face religiosity | Ebaugh et al. | Religious imagery in mission/name/materials; visible symbols; prayer/religious materials with clients/staff. | Evidence span checklist. |
| Administrative priors | NCCS/IRS/NODC | NTEE major group `X` Religion Related; IRS 1023-EZ `Religious Purpose`. | Weak labels/priors; incomplete for affiliated hospitals/schools/social services. |
| High-precision organizational names | NCS/Pew/FBO lit | church, congregation, synagogue, mosque/masjid, temple, parish, diocese, cathedral, chapel, mission, ministry when paired with religious co-term. | Positive only with context for ambiguous terms. |
| Christian terms | Pew/NCS/FBO lit | Christian, Jesus, Christ, Gospel, Bible, Scripture, Holy Spirit, evangelical, discipleship, missionary, Catholic, Baptist, Methodist, Lutheran, Presbyterian, Episcopal/Anglican, Pentecostal, LDS/Mormon. | Strong cues; include denominations. |
| Jewish terms | Pew/NCS/WVS | Jewish, Judaism, Torah, Talmud, synagogue, rabbi, kosher, Sabbath/Shabbat, Reform, Conservative, Orthodox. | Distinguish religious, ethnic, cultural identity where relevant. |
| Islamic terms | Pew/WVS/NLP taxonomies | Islam, Muslim, Allah, Quran/Qur’an, mosque/masjid, imam, Ramadan, Eid, zakat, halal, Sunni, Shia, Sufi. | Avoid ethnicity/nationality proxies. |
| Hindu/Buddhist/Sikh terms | WVS/Pew | Hindu, mandir, Veda, dharma, Buddhist, Buddha, sangha, Theravada, Mahayana, Vajrayana, Sikh, gurdwara, Guru Granth Sahib. | Reduce Christian/English bias. |
| Activity terms | FBO lit | worship, prayer, scripture study, religious education, pastoral care, chaplaincy, sacraments, evangelism, proselytizing, pilgrimage. | Indicates religious activity/program content. |
| Ambiguous/review terms | FBO/survey caveats | faith, spiritual, sacred, holy, mission, ministry, service, grace, mercy, saint/St., temple, fellowship. | Require co-occurring explicit tradition/practice or mark review. |
| Negative/exclusion patterns | coding caveats | “without regard to religion,” nondiscrimination lists, street/place names, medical cross, government ministry, secular mission statement. | Prevent false positives. |

## 4. Synthesis: What Best Distinguishes Constructs

- **Religious identity / affiliation:** strongest evidence is self-identification or organizational self-presentation in name, mission, “about/history,” statement of faith, or explicit tradition.
- **Denomination / tradition:** requires named denomination, congregation, diocese/order/convention, or official association; generic “Christian” is often too coarse.
- **Practice / religiosity:** attendance, prayer, worship, sacraments, scripture study, chaplaincy, religious education, and proselytizing signal activity, not necessarily organizational control.
- **Mission:** best indicated by a mission/vision tying purpose to God, faith, scripture, church, denomination, religious teaching, salvation, worship, or explicitly spiritual calling.
- **Auspice / control:** formal affiliation, sponsorship, board appointment, staff faith requirements, denominational authority, diocese/order/congregational control, or faith-community funding/volunteers.
- **Faith-inspired ambiguity:** historical religious roots, broad spirituality, “values,” “service,” or “mission” language may reflect ethos but not current religious identity; use middle labels or human review.

## 5. Practical Recommendations for LLM Prompts, Weak-Supervision Labeling Functions, and Human Annotation

### LLM prompt template

> Classify whether the organization expresses a current religious or spiritual identity, mission, affiliation, or religious activity. Use only evidence from the organization name, mission statement, program/activity descriptions, history, affiliation/governance, staff/volunteer requirements, religious practices, or explicit religious language. Separately assess: (A) religious identity/affiliation, (B) religious program activity, (C) formal religious auspice/control, (D) historical religious roots only, and (E) ambiguous spiritual/value language. Do **not** count generic words such as “mission,” “ministry,” “service,” “community,” “grace,” or “spiritual” unless tied to a named religious tradition, worship/prayer/scripture, clergy/congregation, denomination, or faith-based affiliation. Return label, confidence, evidence spans, and caveats.

### Weak-supervision labeling functions

- **Positive/high precision:** named tradition/denomination + organization cue: `Catholic Charities`, `Lutheran`, `Episcopal`, `Baptist`, `United Methodist`, `Jewish`, `Islamic`, `Muslim`, `synagogue`, `mosque`, `masjid`, `temple`, `church`, `parish`, `diocese`, `Sisters of`, `congregation`.
- **Positive/activity:** `worship`, `prayer`, `Bible study`, `scripture`, `Gospel`, `Torah`, `Quran`, `chaplain`, `pastoral care`, `religious education`, `sacrament`, `evangelism`, `discipleship`, `missionary`, `zakat`, `halal`, `kosher`.
- **Review/ambiguous:** `faith`, `spiritual`, `sacred`, `ministry`, `mission`, `saint/St.`, `temple`, `grace`, `mercy`, `fellowship`, historically religious universities/hospitals, Salvation Army-like cases.
- **Negative/exclusion:** `without regard to religion`, anti-discrimination clauses, addresses/place names, government ministries, generic corporate mission statements, medical/legal/cultural uses of religious-looking words.
- **Administrative priors:** NTEE `X`, IRS `Religious Purpose`, church/convention filing status, denominational directories; use as weak labels, not ground truth.

### Human annotation instructions

1. Code **identity**, **activity**, and **auspice/control** separately.
2. Require an exact evidence span for every positive label.
3. Mark **historical roots only** when religion appears only in founding/name legacy and not current mission/activity.
4. Use **ambiguous/review** when only weak terms appear without named tradition/practice.
5. Do not assume Christianity; include Jewish, Islamic, Hindu, Buddhist, Sikh, Indigenous, interfaith, spiritual-but-not-religious, and nonreligious options.
6. Document false-positive rationales for ambiguous tokens (`mission`, `ministry`, `saint`, `temple`, `service`).

## 6. Gaps, Caveats, and Read-First List

Major caveats:

- **Christian/English bias:** U.S. nonprofit and survey instruments overrepresent Christian denominations and English lexical cues.
- **Ambiguous terms:** `mission`, `ministry`, `service`, `saint`, `temple`, `faith`, and `spiritual` require context.
- **Identity/practice/belief are distinct:** survey sources consistently separate affiliation, attendance, prayer, salience, belief, and denomination.
- **Administrative labels are incomplete:** NTEE X misses many faith-affiliated schools, hospitals, relief agencies, foundations, and international NGOs.
- **Cross-national wording:** “religion,” “denomination,” and “religious person” may not be functionally equivalent across countries.
- **Social desirability/nonresponse:** attendance/prayer/religion questions are mode-sensitive; Pew and GSS document wording/mode shifts.

Read first:

1. Sider & Unruh (2004)
2. Pew religious composition methodology + 2025 RLS questionnaire/topline
3. NCS cumulative codebook
4. Bielefeld & Cleveland (2013)
5. Ebaugh/Chafetz/Pipes (2003, 2006)
6. GSS 2024 R3 and WVS Wave 8 questionnaires
7. Ma (2021), Fyall/Gugerty/Moore (2018), Santamarina/Lecy/van Holm (2021), and NODC mission classifiers

## From Replication Packages

| Paper / artifact | Language | Key Files | Method | Source URL |
|---|---|---|---|---|
| GivingTuesday `religious_org_v1` | Python / Transformers | model card, config/tokenizer/model files | BERT classifier over name + mission + activity, GPT-labeled/curated data | https://huggingface.co/GivingTuesday/religious_org_v1 |
| Santamarina et al. bespoke NPO taxonomies | R / quanteda | preprocessing and bootstrapped classification replication docs | Naive Bayes over IRS 1023-EZ mission text, religious purpose labels | https://fjsantam.github.io/bespoke-npo-taxonomies/ |
| NODC mission classifiers | R | vignettes/data docs | NTEE and IRS purpose classifiers; benchmark ICR and ML accuracy | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ |
| UK-CAT | Python / regex + ML | `data/ukcat.csv`, `src/ukcat`, notebooks | Rule-based UK charity tags plus logistic regression ICNP/TSO classifier | https://github.com/charity-classification/ukcat |

## Package Recommendations

| Language | Package / artifact | Version | Key Function | Source |
|---|---|---|---|---|
| Python | `transformers` + GivingTuesday model | model updated 2025; BERT base | Fast benchmark/teacher for religious nonprofit classification | https://huggingface.co/GivingTuesday/religious_org_v1 |
| R | `quanteda` | Santamarina replication cites Benoit et al. 2018 / quanteda | DFM, Naive Bayes, dictionary preprocessing | https://fjsantam.github.io/bespoke-npo-taxonomies/ |
| Python | UK-CAT regex scripts | public release v0.3; README notes Python 3.13 | Regex tagger design for charity activities | https://github.com/charity-classification/ukcat |
| Administrative | NTEE / IRS 1023-EZ Religious Purpose | current as available in IRS/NCCS datasets | Weak labels and validation priors | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/data/ |

## Implementation Examples

### Example 1: Multi-field evidence prompt

**Source:** Pew/NCS/Sider & Unruh/Ebaugh synthesis; GivingTuesday model card uses name, mission, and key activities.

```text
Input fields: organization_name, mission_statement, program_activity_text, history, affiliations.
Task: classify current religious mission/identity.
Return JSON: label, confidence, identity_evidence, activity_evidence, auspice_evidence,
historical_roots_only, ambiguous_terms, false_positive_risks.
Rule: generic service/value language is insufficient without religious tradition/practice/affiliation.
```

### Example 2: Weak-supervision trigger tiers

**Source:** NCS/Pew denomination lists; Sider & Unruh; Ebaugh et al.; NODC/Santamarina classifier design.

```text
POS_HIGH = named denomination/tradition OR religious institution term with context
POS_ACTIVITY = worship/prayer/scripture/chaplaincy/religious education/proselytizing
AMBIG_REVIEW = faith/spiritual/ministry/mission/saint/temple/grace/mercy without explicit tradition
NEG_EXCLUDE = without regard to religion OR nondiscrimination list OR address/place/government/corporate sense
ADMIN_PRIOR = NTEE X OR IRS Religious Purpose OR church/convention status
```

## Version and Compatibility Notes

- GSS 2024 R3 (2026 release) flags web-mode wording changes, especially in Religion variables; cite exact release/version.
- Pew 2025 RLS uses bridge surveys to assess comparability after mode changes; do not combine older telephone and newer ABS/web estimates blindly.
- WVS Wave 8 began 2024 and runs 2024–2026; country-specific denomination lists vary.
- GivingTuesday `religious_org_v1` is public Apache-2.0 and useful, but model-card metrics are not peer-reviewed; audit for synthetic-data and Christian/English bias.
- `quanteda`/Naive Bayes replication workflows are transparent baselines; for production, compare against transformer/LLM labels and human evidence-span validation.
