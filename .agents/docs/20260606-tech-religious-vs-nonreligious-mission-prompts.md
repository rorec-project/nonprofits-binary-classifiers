---
created: 20260606
agent: tech-seeker
scratchpad: .agents/notebooks/20260606-tech-religious-mission-prompts-scratchpad.md
status: complete
title: Tech - Religious vs Nonreligious Mission Prompts
topic: religious mission prompts; faith-based organization codebooks; nonprofit mission classification; religion survey wording; weak supervision triggers
---

# Tech: Religious vs Nonreligious Mission Prompts

## Search Strategy and Exact Queries Used

I used the local synthesis map and related reports as background, then ran targeted searches for direct prompt/codebook wording, survey items, official taxonomies, nonprofit mission classifiers, and faith-based organization definitions.

**Local background read**

- `.agents/docs/20260605-literature-synthesis-map.md`
- `.agents/docs/20260605-literature-religious-identity-prompts.md`
- `.agents/docs/20260605-literature-religious-nonprofit-classification.md`
- `.agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md`
- `.agents/docs/20260605-replication-religious-vs-nonreligious-prompts.md`
- `.agents/docs/20260605-replication-religious-nonprofit-mission.md`

**Exact web queries/actions used**

1. `2026 GivingTuesday religious_org_v1 GPT-4 religious organizations name mission activities prompt religious affiliations`
2. `2026 Codebook LLMs evaluating LLMs measurement tools political science concepts prompt codebook label definition clarifications examples`
3. `2026 "What's in a Prompt" prompt design text annotation large language models social science ICWSM 2025`
4. `2026 "religious organization" "mission statement" "GPT-4" nonprofit classifier religious affiliation mission activities`
5. `2026 "faith-based organization" codebook mission religious activity identity affiliation governance staff program content indicators`
6. Fetch `https://huggingface.co/GivingTuesday/religious_org_v1`
7. Fetch `https://huggingface.co/datasets/GivingTuesday/religious_orgs_training`
8. Fetch `https://scholarworks.indianapolis.iu.edu/bitstreams/90959d19-3e24-457a-b205-ff6cf708e4c2/download`
9. Fetch `https://digitalcollections.lipscomb.edu/cgi/viewcontent.cgi?article=1002&context=hpp_fac`
10. Fetch `https://www.pamelapaxton.com/religious-dictionary-holding`

## Key Sources

| # | Source | Authors/Org | Year | Venue/source | DOI/link | Why it matters |
|---|---|---|---|---|---|---|
| 1 | Typology of Religious Characteristics of Social Service and Educational Organizations and Programs | Sider & Unruh | 2004 | NVSQ | https://doi.org/10.1177/0899764003257494 | Core FBO typology: faith-permeated through secular; separates organization/program dimensions. |
| 2 | Defining Faith-Based Organizations and Understanding Them Through Research | Bielefeld & Cleveland | 2013 | NVSQ | https://doi.org/10.1177/0899764013484090 | Warns that FBO status is multidimensional, not a simple binary. |
| 3 | The Varieties of Faith-Related Agencies | Smith & Sosin | 2001 | Public Administration Review | https://doi.org/10.1111/0033-3352.00137 | Defines coupling to faith via resources, authority, and culture. |
| 4 | Where's the Religion? | Ebaugh, Pipes, Chafetz & Daniels | 2003 | JSSR | https://doi.org/10.1111/1468-5906.00191 | Operational indicators: symbols, mission/materials, prayer, proselytizing. |
| 5 | Where's the Faith in Faith-Based Organizations? | Ebaugh, Chafetz & Pipes | 2006 | Social Forces | https://doi.org/10.1353/sof.2006.0086 | Measures organizational religiosity in social service coalitions. |
| 6 | Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector | Ji Ma | 2021 | NVSQ | https://doi.org/10.1177/0899764020968153 | Closest large-scale NTEE nonprofit text classifier. |
| 7 | Beyond NTEE Codes | Fyall, Moore & Gugerty | 2018 | NVSQ | https://doi.org/10.1177/0899764018768019 | Shows mission-statement dictionaries can outperform NTEE for activity detection. |
| 8 | How to Code a Million Missions | Santamarina, Lecy & van Holm | 2023 | Voluntas | https://doi.org/10.1007/s11266-021-00420-z | Replicable mission classifier with IRS purpose codes, including religious purpose. |
| 9 | Nonprofit Mission Classifiers | NODC | 2019-2026 docs | Project docs | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Direct methods and benchmarks for mission/activity text and IRS purpose/NTEE labels. |
| 10 | Religious Orgs Segmentation Model | GivingTuesday Data Commons | 2025/2026 | Hugging Face | https://huggingface.co/GivingTuesday/religious_org_v1 | Closest direct artifact: GPT-4-labeled name/mission/activity data and BERT classifier for religious orgs. |
| 11 | Religious Orgs Training Dataset | GivingTuesday | 2025 | Hugging Face dataset | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Concrete positive/negative examples from nonprofit name, mission, activities. |
| 12 | Religious Dictionary | Ressler, Fulton & Paxton / Paxton | 2019 | Public dictionary | https://www.pamelapaxton.com/religious-dictionary-holding | 900-word dictionary for identifying nonprofits with religious missions in Form 990/990EZ text. |
| 13 | Organizational Religious Expression codebook/report | IU/ORE report | n.d. | ScholarWorks Indianapolis | https://scholarworks.indianapolis.iu.edu/bitstreams/90959d19-3e24-457a-b205-ff6cf708e4c2/download | Strong codebook: conservative default, identity/activity distinction, explicit evidence and examples. |
| 14 | Faith-Based Organizations in Foreign Aid (FORFA) Codebook | Susan Turner Haynes | 2026 | Lipscomb faculty works | https://digitalcollections.lipscomb.edu/hpp_fac/3 | Direct human-coding instructions for secular, religious roots, traditions, ambiguous/indigenous, multireligious. |
| 15 | Faith Data: Methodology | NPC | 2016 | Methodology report | https://www.thinknpc.org/wp-content/uploads/2018/07/NPC-faith-data_methodology_April16.pdf | Shows combined self-classification, taxonomy, keyword scoring, manual override approach. |
| 16 | National Congregations Study cumulative codebook | Chaves et al. | 2020 | Official codebook | https://www.nationalcongregationsstudy.org/data-documentation | Exact congregation affiliation and denomination wording. |
| 17 | GSS 2024 Codebook | NORC | 2024 | Official codebook | https://gss.norc.org/ | Classic religion preference, denomination, attendance, prayer, born-again items. |
| 18 | Pew Religious Landscape Study / question wording | Pew Research Center | 2018-2025 | Official survey docs | https://www.pewresearch.org/religion/ | Best U.S. affiliation wording: "present religion, if any" plus nonreligion categories. |
| 19 | WVS Wave 8 Questionnaire | WVS Association | 2024 | Official questionnaire | https://www.worldvaluessurvey.org/documents/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf | Identity/practice/salience wording including "religious person / not religious / atheist." |
| 20 | ARDA Measurement Wizard | Bradburn et al. | 2014 | JSSR / ARDA | https://doi.org/10.1111/jssr.12131 | Survey question bank for religion/spirituality constructs. |
| 21 | Codebook LLMs | Halterman & Keith | 2025/2026 | Political Analysis | https://www.cambridge.org/core/journals/political-analysis/article/codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts/7B323A0E47F782F2698A0AE849EA00DE | Best current codebook-to-LLM measurement framework. |
| 22 | What's in a Prompt? | Atreja et al. | 2025 | ICWSM | https://doi.org/10.1609/icwsm.v19i1.35807 | Prompt design changes compliance, accuracy, and label distributions. |
| 23 | Language Models in the Loop | Smith, Fries, Hancock & Bach | 2024 | ACM/IMS JDS | https://doi.org/10.1145/3617130 | Incorporates LLM prompts as weak-supervision labeling functions with abstention. |
| 24 | Faith-based organisations in multilateral humanitarian aid | Journal of International Humanitarian Action | 2026 | Springer | https://link.springer.com/article/10.1186/s41018-025-00188-7 | Recent quantitative FBO/secular NGO operationalization; conservative evidence rule. |
| 25 | Concept of Religion | Stanford Encyclopedia of Philosophy | current | Reference | https://plato.stanford.edu/entries/concept-religion | Authoritative warning: no universal definition; use stipulative measurement definition. |

## Prompt-Like Language Extracted

| Type | Source | Actual wording / criteria / trigger | Use for this project |
|---|---|---|---|
| Direct LLM classifier intent | GivingTuesday | Provides name, mission statement, and key activities to GPT-4 and asks it to find "mentions/wording/terminology that reveal an org's religious affiliations." | Closest direct prompt precedent; add explicit evidence spans, uncertainty, and boundary rules. |
| Direct definition | GivingTuesday | "Religious organizations are organizations whose identity and mission are derived from a religious or spiritual tradition and which operate as registered or unregistered, nonprofit, voluntary entities." | Adopt as broad definition, then narrow operational binary to observable text evidence. |
| Direct training examples | GivingTuesday dataset | "Bible studies," "Music Ministry, promoting the Gospel," "Christian worldview," "Jewish values," "Catholic health care system," "pastoral care," "religious, cultural and charitable activities." | Seed high-precision positive examples and audit boundary cases. |
| Conservative evidence default | ORE/IU | "No known religious expression" until evidence suggests otherwise; code identity OR activity and identity AND activity. | Make absence-of-evidence negative/unknown, not inferred religiosity. |
| ORE identity | ORE/IU | Identity includes organization name, religious imagery, mission statement, history, purpose statement, logo, tagline, branding, statement of belief/confession. | Domain: `religious_identity_or_affiliation`. |
| ORE activity | ORE/IU | Activity includes religious practice supported through funding, program goals, products, services; religious expression in program goals/activities/donations/products/services. | Domain: `religious_service_content`. |
| ORE lexical categories | ORE/IU | Name of religious faith; religious title; religious community; religious actions/activity; religion; religious building; religious practice; doctrine/belief; sacred writings. | Use as weak-supervision trigger families, not flat keywords. |
| ORE spiritual caveat | ORE/IU | "Spirit/spiritual/spirituality" counted only when directly modified by a religious tradition/word or when other explicit religious identity/activity evidence exists. | Prevent broad spiritual false positives. |
| ORE partner caveat | ORE/IU | Working with religious groups is not evidence of religious identity; activity-positive only if activity itself is explicitly religious. | Separate network/partner evidence from organization identity. |
| FBO typology | Sider & Unruh | Faith-permeated; faith-centered; faith-affiliated; faith-background; faith-secular partnership; secular; religion manifests tangibly through mission, founding, affiliation, board, staff, support, practices, environment, content, integration, outcomes. | Multi-domain labels plus ambiguous/historical class. |
| Faith-related coupling | Smith & Sosin | Faith linkage via resources, authority, and culture. | Add governance/resources/culture domains where text allows. |
| Visible religiosity | Ebaugh/Chafetz/Pipes | Visible religious symbols; religiously explicit mission/materials; prayer or religious material with clients/staff; proselytizing; staff understand work as religious purpose. | Human annotation checklist. |
| FORFA code 30/40 | Haynes FORFA | Code 30 if mission/name indicates secular; code 40 if secular with religious roots. | Keep `secular_with_religious_roots` separate from positive religious mission. |
| FORFA tradition codes | Haynes FORFA | Jewish, Buddhist, Islamic, Christian, Hindu, multireligious, spiritually ambiguous/indigenous; secular defined as absence or negative qualifying reference. | Tradition-specific output optional; useful for bias audit. |
| FORFA lexical triggers | Haynes FORFA | Jewish: Torah, Halakha, synagogue; Islamic: Allah, Quran, Muslim, Sharia, Sunni, Imam, Sufi, zakat; Christian: Bible, Gospels, Christ, Jesus, Cross, Catholic, church, Holy Spirit; Spiritual ambiguous: God, divine, Spirit, meditation, holy, faith-based. | Expand beyond Christian terms; flag ambiguous spiritual terms. |
| FORFA negative rule | Haynes FORFA | "Without regard to...religion" should be coded secular, despite containing "religion." | Add explicit negative filter. |
| FORFA imagery caveat | Haynes FORFA | Logo/homepage religious images can count; distinguish Christian cross from medical cross. | Useful if future pipeline includes websites/images; not for text-only. |
| Paxton dictionary | Ressler/Fulton/Paxton | 900 words to identify nonprofits with religious missions in Form 990/990EZ mission and primary exempt-purpose fields. | Candidate dictionary baseline and LF source; validate before use. |
| IRS/NTEE triggers | IRS/NCCS/NODC | Church/synagogue/etc.; association/convention of churches; religious order; church auxiliary; mission; missionary activities; evangelism; religious publishing; other religious activities; NTEE X Religion Related. | Administrative priors; mission alone ambiguous. |
| Pew affiliation | Pew | "What is your present religion, if any? Are you Protestant, Roman Catholic, Mormon, Orthodox..., Jewish, Muslim, Buddhist, Hindu, atheist, agnostic, something else, or nothing in particular?" | Use "if any"; explicit nonreligious categories. |
| GSS preference | GSS | "What is your religious preference? Is it Protestant, Catholic, Jewish, some other religion, or no religion?" | Compact classic wording; expand for non-Christian traditions. |
| NCS affiliation | NCS | "Is your congregation formally affiliated with a denomination, convention, or some similar kind of association?" | Formal affiliation domain. |
| NCS denomination | NCS | "Please tell me the name of your denomination or other association." | Named denomination as high-precision evidence. |
| WVS identity | WVS/EVS | "Independently of whether you attend religious services or not, would you say you are: a religious person, not a religious person, an atheist?" | Identity, practice, and belief must stay separate. |
| Codebook LLM format | Halterman & Keith | Codebook components: Label; Label Definition; Clarification; Negative Clarification; Positive & Negative Examples; task instruction; Output reminder. | Use a semi-structured codebook before prompting. |
| Codebook LLM prompt style | Halterman & Keith | "You're an expert... Carefully read the definitions below, read the story, and write the Label that best matches the story. Use only the provided labels." | Use expert role + definitions + constrained labels + output reminder. |
| Prompt sensitivity | Atreja et al. | Definition inclusion, output type, explanation, and prompt length affect compliance, accuracy, and label distribution. | Freeze prompts; run prompt-variant validation; do not tune on final test set. |
| Prompted weak supervision | Smith et al. | LLM prompts can be labeling functions with label maps; nonmatching outputs abstain. | Treat prompts as noisy LFs, not ground truth. |

## Proposed Prompt/Codebook Architecture

### Core construct

Classify **observable religious or spiritual mission/expression in short organization text**, not latent religiosity, virtue, morality, founder intent, or respondent belief. Use only the provided fields: organization name, mission/purpose, program/activity text, history/affiliation text if available, and administrative labels if explicitly provided.

### Recommended domains/dimensions

| Domain | Positive evidence | Negative/ambiguous rule |
|---|---|---|
| `religious_purpose_explicit` | Worship, ministry tied to a faith, evangelism, religious education, faith formation, discipleship, Gospel/scripture/prayer-linked mission, God/Christ/Allah/Torah/Quran/Dharma-linked purpose. | Do not count generic "mission," "serve," "values," "compassion," "community," "stewardship" without religious anchor. |
| `religious_identity_or_affiliation` | Church, mosque, synagogue, temple, congregation; denomination/order/diocese; Catholic, Lutheran, Baptist, Islamic, Jewish, Hindu, Buddhist, Sikh, etc.; statement of faith/confession; formal faith-based auspice. | Saint/geographic/founder names alone are ambiguous for hospitals, schools, places, and memorials. |
| `religious_service_content` | Prayer, Bible/Quran/Torah study, chaplaincy, pastoral care, worship, sacraments, religious counseling, missionary activity, evangelism, religious publishing/schooling/spiritual formation. | Secular services by a faith-founded org are not activity-positive unless service content is religious. |
| `religious_governance_or_authority` | Board/staff/clergy selected by faith body; sponsorship by diocese/order/congregation; formal denomination/convention affiliation. | Often absent from short text; code unknown rather than negative. |
| `religious_resources_or_networks` | Funding, volunteers, facilities, coalition membership, or support from churches/faith communities/religious orders. | Faith-based partners indicate network, not necessarily organization identity. |
| `spiritual_or_faith_inspired_ambiguous` | Spirituality, sacred, divine, faith-inspired, meditation, healing, Mother Earth, "faith in action," historic roots, religious values without named tradition/practice. | Keep separate; do not automatically map to strict religious positive. |
| `secular_with_religious_roots` | Founded by missionaries, former church affiliation, religious founder history, but current mission/activity is secular. | Review/sensitivity class, not positive unless current evidence exists. |
| `administrative_religion_prior` | NTEE X/REL; IRS religious purpose; old IRS activity codes; church filing exemption/status. | Weak prior only; administrative labels are incomplete, stale, and single-purpose. |
| `negative_or_secular_evidence` | Explicit secular/humanist/atheist/nonreligious identity; anti-discrimination "without regard to religion"; no religious evidence in adequate text. | "No evidence" should remain `insufficient_information` if text is too short. |

### Derived binary rules

- **Religious positive**: explicit evidence span for `religious_purpose_explicit`, `religious_service_content`, or high-confidence `religious_identity_or_affiliation`.
- **Probable/review**: only governance, resources/network, or administrative prior is present.
- **Ambiguous/review**: only spiritual language, historical roots, saint names, generic faith/values language, faith-based partners, or boilerplate legal purposes appear.
- **Nonreligious/negative**: no evidence in sufficient text, or explicit secular/nonreligious/no religious affiliation.
- **Insufficient information**: text too short or only organization name with weak/ambiguous cue.

### Recommended LLM prompt skeleton

```text
You are an expert social-science annotator coding nonprofit organization text.

Task: Classify whether the organization expresses an observable religious or spiritual identity, mission, affiliation, governance tie, resource/network tie, or religious program/activity. Use only the provided text fields. Do not infer hidden beliefs or religiosity.

Definitions:
- Religious positive: explicit evidence that the organization's identity, mission, affiliation, governance, or program content is derived from or tied to a religious/spiritual tradition or religious practice.
- Nonreligious/negative: no explicit religious/spiritual evidence in sufficient provided text, or the text is purely secular/civic/service-oriented.
- Ambiguous/review: weak or context-dependent evidence only, such as saint name, generic faith/spiritual language, historical roots, faith-based partner mention, or ministry/mission without religious co-term.
- Insufficient information: text is too sparse to evaluate.

Code domains separately: religious_purpose_explicit, religious_identity_or_affiliation, religious_service_content, religious_governance_or_authority, religious_resources_or_networks, spiritual_or_faith_inspired_ambiguous, secular_with_religious_roots, administrative_religion_prior, negative_or_secular_evidence.

Rules:
1. Require short evidence spans for every positive or ambiguous domain.
2. Do not count generic words like mission, ministry, serve, compassion, community, values, grace, mercy, fellowship, or stewardship unless tied to a religious tradition, practice, scripture, worship, denomination, congregation, or faith body.
3. Separate organization-level identity from program-level religious content.
4. Mark historical religious roots or religiously founded but currently secular service text as ambiguous/review unless current mission/activity/affiliation is explicit.
5. Include non-Christian evidence: Jewish, Muslim/Islamic, Hindu, Buddhist, Sikh, Indigenous, interfaith, spiritual, humanist/atheist/nonreligious.
6. Treat boilerplate legal clauses like "charitable, religious, educational..." and phrases like "without regard to religion" as not positive by themselves.

Return JSON:
{
  "binary_label": "religious" | "nonreligious" | "ambiguous_review" | "insufficient_information",
  "confidence": 0-1,
  "domains_present": [...],
  "evidence_spans": [{"domain": "...", "field": "name|mission|activity|admin", "quote": "..."}],
  "boundary_notes": "..."
}
```

## Synthesis of Authoritative Definitions

| Construct | Synthesis for classifier |
|---|---|
| Religion | No universal scholarly definition; use a stipulative measurement definition: observable ties to religious/spiritual traditions, practices, symbols, institutions, or mission in text. |
| Religious identity | Organizational self-presentation with a religion, tradition, denomination, congregation, order, statement of faith, religious imagery/name, or formal affiliation. |
| Religious mission | Purpose or program goal explicitly connected to worship, religious formation, evangelism/missionary activity, scripture, prayer, sacred obligations, or service framed as religious vocation. |
| Faith-based organization | Multidimensional: may be faith-permeated, faith-centered, faith-affiliated, faith-background, faith-secular partnership, or secular. Code dimensions first. |
| Religious organization | Organization whose identity and mission are derived from a religious/spiritual tradition and that operates as a nonprofit/voluntary entity; operationalize only observable evidence. |
| Faith community/congregation | Community organized around worship/practice/teaching; NCS operationalizes formal affiliation, denomination/association, worship services, clergy, and religious life. |
| Spirituality | Sacred/transcendent/meaning-oriented language; in organization text, ambiguous unless tied to tradition, practice, governance, or program content. |
| Nonreligion | Survey categories include atheist, agnostic, nothing in particular/no religion; in organization text: absence of religious evidence or explicit secular/humanist/atheist/freethought identity. |
| Secular/nonreligious organization | Mission/activity/identity not religiously grounded in observable text; secular does not mean anti-religious. |

## Pitfalls and Boundary Cases

- **Christian/English bias**: Add Jewish, Muslim, Hindu, Buddhist, Sikh, Indigenous, interfaith, and nonreligious examples; audit tradition-level false negatives.
- **Saint names**: `St.`, `Saint`, `San`, `Santa`, and religious founders may indicate geography, hospital/school naming, or historical roots only.
- **Ministry/mission ambiguity**: `mission`, `ministry`, `fellowship`, `stewardship`, `grace`, `mercy`, `spirit`, `healing`, `chapel`, `temple` require co-text.
- **Religiously founded but secularized institutions**: Hospitals, universities, schools, and relief agencies may retain names/history but not religious mission/activity.
- **Spiritual but nonreligious language**: Meditation, wellness, sacred ecology, higher power, and spiritual healing need a separate ambiguous/spiritual class.
- **Ethnic/religious overlap**: Jewish, Sikh, Hindu, Muslim, Arab, Indian, Pakistani, Tibetan, etc. can signal ethnicity/culture as well as religion; require context.
- **Nonresponse/social desirability analog**: Survey literature shows identity, practice, belief, salience, and affiliation differ; do not collapse them conceptually.
- **Organization vs program level**: A religious organization may run secular programs; a secular organization may run chaplaincy/spiritual-care programs.
- **Administrative gaps**: NTEE X misses faith-affiliated schools/hospitals/social services; Form 990 excludes many churches; old IRS codes may be stale.
- **LLM prompt drift**: Preserve prompt text, model version, parameters, raw outputs, evidence spans, and prompt-disagreement flags.

## Read-First List and Direct Implications

1. **GivingTuesday model/dataset** — closest direct GPT-4 + nonprofit name/mission/activity classifier; use examples and failure cases.
2. **ORE/IU codebook** — best identity/activity evidence structure and conservative default.
3. **Sider & Unruh; Bielefeld & Cleveland; Smith & Sosin** — prevents over-binary construct design.
4. **FORFA codebook + Paxton/Ressler/Fulton dictionary** — concrete tradition-specific triggers and secular/roots/spiritual ambiguous rules.
5. **Pew/GSS/WVS/NCS/ARDA** — wording for affiliation, nonreligion, denomination, practice, salience.
6. **Ma/Fyall/Santamarina/NODC** — treat mission and program text as classification fields; administrative labels are noisy priors.
7. **Codebook LLMs; Atreja et al.; Smith et al.** — create a semi-structured codebook, freeze prompts, use uncertainty/abstention, validate prompt variants.

**Direct weak-supervision implications**

- High-precision positives: named traditions/denominations + organization terms; worship/prayer/scripture/evangelism/chaplaincy/missionary/religious education.
- Ambiguous-review LFs: saint names, spiritual/faith/grace/mercy/ministry/mission alone, religious roots, faith-based partner mentions, boilerplate 501(c)(3) purposes.
- Negative filters: "without regard to religion," purely legal "charitable, religious, educational" clauses, medical cross, geographic saint names, government "ministry."
- Administrative priors: NTEE X, IRS religious purpose/activity codes, church status, Candid/PCS religion subjects if licensed.

## From Replication Packages

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| GivingTuesday Religious Orgs Segmentation | Python/Transformers | HF model card and dataset | GPT-4 labeling + BERT classifier over name/mission/activity | https://huggingface.co/GivingTuesday/religious_org_v1 |
| Santamarina et al. (2023) / Bespoke NPO Taxonomies | R/Python | Dataverse/GitHub mission corpora and classifiers | Quanteda/Naive Bayes mission classifiers incl. religious purpose | https://github.com/fjsantam/bespoke-npo-taxonomies |
| Nonprofit Mission Classifiers | R | DATA/MISSION, docs, taxonomies | Mission/activity classifier benchmarks for NTEE/IRS purpose | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ |
| Ma (2021) | Python/Jupyter | `ma-ji/npo_classifier` | NTEE classifiers over nonprofit text | https://github.com/ma-ji/npo_classifier |

## Package Recommendations

Not applicable — this report is a literature/codebook search, not a software-package selection task.

## Implementation Examples

Not applicable — no code implementation requested. Prompt skeleton and weak-supervision triggers are provided above.

## Version and Compatibility Notes

- GivingTuesday model/dataset are current public artifacts with Apache-2.0 license; public prompt is partial, not a full reproducible GPT-4 prompt.
- Codebook LLMs and prompt-design evidence are 2024-2026 frontier sources; use validation-first workflow because model behavior changes quickly.
- Classic FBO sources remain construct-authoritative despite age; use them for dimensions, not performance claims.
- Dictionaries/taxonomies should be audited for Christian/English bias and false positives before production use.
