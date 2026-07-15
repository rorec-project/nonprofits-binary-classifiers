---
created: 20260605
agent: literature-seeker
scratchpad: docs/research/notebooks/20260605-literature-religious-vs-nonreligious-mission-prompts-scratchpad.md
status: complete
title: Literature - Religious vs Nonreligious Mission Prompts
topic: religious mission prompts; faith-based organization codebooks; nonprofit mission classification; religion survey wording; weak supervision triggers
---

# Literature: Religious vs Nonreligious Mission Prompts

## Search Strategy

I used the existing synthesis map and related reports as background, then searched breadth-first across official survey/codebook documentation, nonprofit/FBO construct literature, nonprofit mission-text classifiers, direct model/dataset cards, LLM codebook-prompt methodology, and definitional reference sources.

**Exact searches/actions used**

1. Read local synthesis/background: `docs/research/20260605-literature-synthesis-map.md`; `docs/research/20260605-literature-religious-identity-prompts.md`; `docs/research/20260605-literature-religious-nonprofit-classification.md`; `docs/research/20260605-literature-llm-weak-supervision-noisy-labels.md`.
2. `2026 Pew GSS WVS National Congregations Study religion question wording codebook religious affiliation denomination religious person attendance prayer`
3. `2026 WVS EVS ARDA Measurement Wizard religion survey question wording religious person spiritual person God importance affiliation denomination`
4. `2026 Sider Unruh faith-based organization typology religious characteristics mission affiliation staff support program content indicators Bielefeld Cleveland Smith Sosin Ebaugh Chafetz Pipes`
5. `2026 nonprofit mission statement classification NTEE religious purpose IRS Form 990 Ma Fyall Santamarina Lecy codebook classifiers religious organization model mission activity text`
6. Fetch `https://huggingface.co/GivingTuesday/religious_org_v1`
7. Fetch `https://huggingface.co/datasets/GivingTuesday/religious_orgs_training`
8. `2026 LLM as annotator computational social science prompt wording codebook text annotation Gilardi Pangakis Ziems Heseltine Törnberg Mellon codebook prompt`
9. `2026 authoritative definitions religion religious organization religious identity spirituality nonreligion secular organization faith community sociology of religion nonprofit studies`

## Papers

| Title | Authors/Org | Year | Venue/Source | DOI/URL | Confidence |
|-------|-------------|------|--------------|---------|------------|
| Typology of Religious Characteristics of Social Service and Educational Organizations and Programs | Ronald J. Sider; Heidi Rolland Unruh | 2004 | Nonprofit and Voluntary Sector Quarterly | https://doi.org/10.1177/0899764003257494 | High |
| Defining Faith-Based Organizations and Understanding Them Through Research | Wolfgang Bielefeld; William Suhs Cleveland | 2013 | Nonprofit and Voluntary Sector Quarterly | https://doi.org/10.1177/0899764013484090 | High |
| The Varieties of Faith-Related Agencies | Steven Rathgeb Smith; Michael R. Sosin | 2001/2002 | Public Administration Review | https://doi.org/10.1111/0033-3352.00137 | High |
| Where’s the Religion? Distinguishing Faith-Based from Secular Social Service Agencies | Penny Edgell Becker / Ebaugh, Pipes, Chafetz, Daniels | 2003 | Journal for the Scientific Study of Religion | https://doi.org/10.1111/1468-5906.00191 | High |
| Where’s the Faith in Faith-Based Organizations? Measures and Correlates of Religiosity in Faith-Based Social Service Coalitions | Helen Rose Ebaugh; Janet Saltzman Chafetz; Paula Pipes | 2006 | Social Forces | https://doi.org/10.1353/sof.2006.0086 | High |
| Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector | Ji Ma | 2021 | Nonprofit and Voluntary Sector Quarterly | https://doi.org/10.1177/0899764020968153 | High |
| Beyond NTEE Codes: Opportunities to Understand Nonprofit Activity Through Mission Statement Content Coding | Rachel Fyall; Mary Kay Gugerty; Megan/Rachel Moore | 2018 | Nonprofit and Voluntary Sector Quarterly | https://doi.org/10.1177/0899764018768019 | High |
| How to Code a Million Missions: Developing Bespoke Nonprofit Activity Codes Using Machine Learning Algorithms | Francisco J. Santamarina; Jesse D. Lecy; Eric J. van Holm | 2023 | Voluntas | https://doi.org/10.1007/s11266-021-00420-z | High |
| Nonprofit Mission Classifiers | Nonprofit Open Data Collective / Lecy, van Holm, Santamarina | 2019-2023 | Replication docs | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Medium-High |
| National Taxonomy of Exempt Entities (NTEE) Codes | NCCS / Urban Institute | 2026 | Official taxonomy documentation | https://nccs.urban.org/nccs/resources/ntee/ | High |
| Exempt Organizations Business Master File documentation | IRS | 2026 | Official documentation | https://www.irs.gov/pub/irs-soi/eo-info.pdf | High |
| Religious Orgs Segmentation Model | GivingTuesday Data Commons | 2025/2026 | Hugging Face model card/dataset | https://huggingface.co/GivingTuesday/religious_org_v1 | Medium-High |
| National Congregations Study Cumulative Codebook | Mark Chaves et al.; NCS/ARDA | 2020 | Official codebook/dataset | https://sites.duke.edu/ncsweb/files/2020/09/NCS-I-IV-Cumulative-Codebook_FINAL_8Sept2020.pdf | High |
| GSS 2024 Codebook | NORC at the University of Chicago | 2024 | Official codebook | https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook%20R2.pdf | High |
| Pew religion survey questionnaires / Religious Landscape Study wording | Pew Research Center | 2021-2025 | Official methodology/questionnaires | https://www.pewresearch.org/religion/ | High |
| WVS Wave 8 Questionnaire | World Values Survey Association | 2024 | Official questionnaire | https://www.worldvaluessurvey.org/documents/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf | High |
| Toward Assessing and Improving Survey Questions on Religion: The ARDA's Measurement Wizard | Norman Bradburn et al. | 2014 | Journal for the Scientific Study of Religion | https://doi.org/10.1111/jssr.12131 | High |
| Codebook LLMs: Evaluating LLMs as Measurement Tools for Political Science Concepts | Halterman & Keith / Codebook LLMs authors | 2024/2026 | Political Analysis / arXiv | https://www.cambridge.org/core/journals/political-analysis/article/codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts/7B323A0E47F782F2698A0AE849EA00DE | High |
| Best Practices for Text Annotation with Large Language Models | Petter Törnberg | 2024 | Sociologica | https://doi.org/10.6092/issn.1971-8853/19461 | High |
| What’s in a Prompt? | Atreja et al. | 2025 | ICWSM | https://doi.org/10.1609/icwsm.v19i1.35807 | High |
| Can Large Language Models Transform Computational Social Science? | Caleb Ziems et al. | 2024 | Computational Linguistics | https://doi.org/10.1162/coli_a_00502 | High |
| Language Models in the Loop: Incorporating Prompting into Weak Supervision | Ryan Smith; Jason Fries; Braden Hancock; Stephen Bach | 2024 | ACM/IMS Journal of Data Science | https://doi.org/10.1145/3617130 | High |
| The Concept of Religion | Stanford Encyclopedia of Philosophy | Current | Reference | https://plato.stanford.edu/entries/concept-religion | High |
| Religious Organizations | Springer Encyclopedia of Global Religion | Current | Reference | https://link.springer.com/rwe/10.1007/978-3-319-31816-5_2514-1 | High |

## Why Key Sources Matter

- **Closest direct source**: GivingTuesday `religious_org_v1` uses GPT-4 labels and BERT on nonprofit name, mission, and program activity text to classify religion/nonreligion and religious affiliation. Public prompt details are partial but directly aligned.
- **Best construct sources**: Sider & Unruh; Bielefeld & Cleveland; Smith & Sosin; Ebaugh/Chafetz/Pipes define religiousness as multidimensional: identity, affiliation, authority, culture, resources, staff/governance, service content, proselytizing, and program integration.
- **Best mission-text classifier sources**: Ma; Fyall et al.; Santamarina et al.; Nonprofit Mission Classifiers; NCCS/IRS show how name/mission/program fields, NTEE, and IRS purpose/activity codes can become noisy labels.
- **Best exact survey wording**: Pew, GSS, WVS/EVS, NCS, ARDA provide tested wording for affiliation, nonreligion, denomination, attendance, prayer, spirituality, and salience.
- **Best LLM/codebook prompt sources**: Codebook LLMs, Törnberg, Atreja et al., Ziems et al., Smith et al. show why prompts must be codebook-grounded, structured, validated, and tested for prompt sensitivity.

## Prompt-Like Language Extracted

| Type | Source | Actual wording / criteria / trigger | Direct implication |
|---|---|---|---|
| Religious affiliation survey item | Pew | “What is your present religion, if any? Are you Protestant, Roman Catholic, Mormon, Orthodox..., Jewish, Muslim, Buddhist, Hindu, atheist, agnostic, something else, or nothing in particular?” | Use “if any”; explicitly include nonreligious classes instead of treating absence as residual. |
| Religious preference survey item | GSS | “What is your religious preference? Is it Protestant, Catholic, Jewish, some other religion, or no religion?” | Compact binary/nominal affiliation template; too Christian-centered unless expanded. |
| Congregation affiliation | NCS | “Is your congregation formally affiliated with a denomination, convention, or some similar kind of association?” | Code formal affiliation separately from religious content. |
| Denomination follow-up | NCS | “Please tell me the name of your denomination or other association.” | Use named denomination/association as high-precision identity evidence. |
| NCS broad tradition categories | NCS | Catholic; Baptist; Methodist; Lutheran; Presbyterian/Reformed; Pentecostal; Other Christian; Non-Christian; TRAD3: Roman Catholic, White conservative/evangelical/fundamentalist, Black Protestant, White liberal/moderate, Non-Christian. | Useful grouping but avoid importing race-specific tradition labels unless needed and justified. |
| Religion importance | WVS | “For each of the following, indicate how important it is in your life... Religion: Very important / Rather important / Not very important / Not at all important.” | Separate salience from identity; salience is not enough for organization positive class. |
| God importance | WVS | “How important is God in your life? ... 10 means ‘very important’ and 1 means ‘not at all important.’” | Captures theistic salience; not applicable to all traditions. |
| Attendance | Pew/WVS | “Aside from weddings and funerals, how often do you attend religious services?” | Exclude ceremonial/social mentions unless mission/activity includes religious services. |
| Prayer | WVS/Pew | “Outside of religious services, how often do you pray?” / “Apart from weddings and funerals... how often do you pray?” | Prayer is strong religious practice trigger in mission/activity text. |
| Religious person identity | WVS/EVS | “Independently of whether you attend religious services or not, would you say you are: a religious person, not a religious person, an atheist?” | Identity, practice, and belief are distinct; allow nonreligious identity. |
| Denomination belonging | WVS | “Do you belong to a religion or religious denomination? If yes, which one?” | Use belonging/affiliation separately from service content. |
| FBO typology | Sider & Unruh | Faith-permeated; faith-centered; faith-affiliated; faith-background; faith-secular partnership; secular. Focus on “tangibly expressive ways that religion may be manifest” and separate organizations from programs. | Use ambiguous/historical roots category; do not force every faith-linked org into binary positive. |
| FBO dimensions | Sider & Unruh / reviews | Mission; founding; affiliation; board; management; staff; support/resources; practices; environment; program content; integration; outcomes. | Label dimensions, then derive binary. |
| Faith-related agency coupling | Smith & Sosin | Coupling to faith through resources, authority, and culture. | A school/hospital may be religious by authority/culture even if program text is secular; needs separate evidence type. |
| Visible religiosity indicators | Ebaugh/Chafetz/Pipes | Visible religious symbols; religiously explicit mission/materials; prayer or religious material with clients/staff; proselytizing; staff understand work as religious purpose. | Strong annotation checklist; map each indicator to evidence span/field. |
| Mission-statement public face | Becker/Ebaugh et al. | Faith-based agencies used religious imagery in the “public face” to communicate religiousness. | Include name/logo/tagline/history/mission, not just program services. |
| ORE codebook identity | ORE/IU report | Identity: organization’s name, religious imagery, mission statement, history, purpose statement, logo, tagline, branding, statement of belief/confession. | Define `religious_identity_or_affiliation`. |
| ORE codebook activity | ORE/IU report | Activity: religious practice supported through funding, program goals, products, services; evidence of religious expression in program goals/activities/donations/products/services. | Define `religious_service_content`. |
| ORE coding assumption | ORE/IU report | “No known religious expression” until evidence suggests otherwise; code “identity OR activity” and “identity AND activity.” | Conservative default negative/unknown; require evidence. |
| IRS activity code triggers | IRS BMF | Religious Activities: Church/synagogue/etc.; association or convention of churches; religious order; church auxiliary; mission; missionary activities; evangelism; religious publishing; other religious activities. | High-precision weak-supervision trigger list; note “mission” alone is ambiguous. |
| NTEE major group | NCCS/IRS | NTEE `X` / `REL` = Religion Related; NTEE is descriptive, single-category, and often imprecise. | Treat NTEE X as a prior/weak label, not ground truth. |
| IRS 1023-EZ purpose code | Nonprofit Mission Classifiers | Binary “Religious Purpose” among charitable, religious, educational, scientific, literary, etc. | Multi-purpose binary code fits multi-label architecture better than NTEE. |
| Direct GPT-4 prompt intent | GivingTuesday | Prompt GPT-4 with name, mission, key activities and ask it to find “mentions/wording/terminology that reveal an org’s religious affiliations.” | Closest direct LLM precedent; add explicit evidence-span and ambiguity rules. |
| Direct training examples | GivingTuesday dataset | Positives include “Bible studies,” “Music Ministry, promoting the Gospel,” “Christian worldview,” “Jewish values,” “Catholic health care system,” “religious, cultural and charitable activities.” | Use as seed examples and weak-trigger calibration cases. |
| LLM codebook format | Codebook LLMs | Structured components: Label; label definition; clarifications; negative clarifications; positive and negative examples; task instruction; output reminder. | Build machine/human-readable codebook before prompting. |
| LLM prompt example | Törnberg | “As an expert annotator... Does the message contain misinformation...? Provide your response in JSON... Yes/No/Uncertain... justification...” | Use role + question + constraints + uncertain option + JSON. |
| Prompt sensitivity | Atreja et al. / Prompt selection papers | Definition inclusion, output type, explanation, and prompt length change compliance/accuracy and label distributions. | Run multiple prompt variants and aggregate/disagreement-audit. |

## Proposed Prompt/Codebook Architecture for This Project

### Core construct

Classify **observable religious or spiritual mission/expression in short organization text**, not latent religiosity, virtue, morality, or founder intent. Use only evidence in the provided fields: organization name, mission/purpose statement, program/activity description, history/affiliation text if available, and administrative labels if explicitly provided.

### Recommended label domains

| Domain | Positive evidence | Negative/ambiguous rule |
|---|---|---|
| `religious_purpose_explicit` | Mission/purpose says worship, ministry, evangelism, religious education, faith formation, discipleship, Gospel, scripture, prayer, religious charitable mission, God/Christ/Allah/Torah/Quran/Dharma-linked purpose. | Do not count generic “mission,” “serve,” “values,” “compassion,” “community,” “stewardship” without religious anchor. |
| `religious_identity_or_affiliation` | Church, mosque, synagogue, temple, congregation; denomination/religious order/diocese; Catholic, Lutheran, Baptist, Islamic, Jewish, Hindu, Buddhist, Sikh, etc.; statement of faith/confession; faith-based coalition/auspice. | Saint or tradition name alone can be ambiguous for hospitals/schools/geography; mark as review if no current religious evidence. |
| `religious_service_content` | Prayer, Bible/Quran/Torah study, chaplaincy/pastoral care, worship, sacraments, religious counseling, missionary activities, evangelism, religious publishing, religious schooling, spiritual formation. | Secular services delivered by religiously founded org should not be coded activity-positive unless service content is religious. |
| `religious_governance_or_authority` | Board/staff/clergy selected by faith body; sponsored by diocese/order/congregation; formal denomination/convention affiliation; religious order controls institution. | Often absent from short text; use `unknown` not negative if no evidence. |
| `religious_resources_or_networks` | Funding, volunteers, facilities, coalition membership, or support from churches/faith communities/religious orders. | “Partnered with faith-based organizations” may indicate network, not necessarily org identity; do not force binary positive without other evidence. |
| `spiritual_or_faith_inspired_ambiguous` | Spirituality, sacred, faith-inspired, meditation, healing, Mother Earth, “faith in action,” historic faith roots, religious values without named tradition or practice. | Keep separate; audit separately; do not automatically positive for strict binary. |
| `administrative_religion_prior` | NTEE X/REL, IRS religious purpose, old IRS activity codes 001-029, church filing exemption/status. | Weak prior only; administrative codes are incomplete, single-purpose, and may be stale. |
| `negative_or_secular_evidence` | “No religion,” secular, humanist/atheist/freethought identity if classifying religious vs nonreligious; “without regard to religion” anti-discrimination language; purely civic/education/health/housing mission with no religious evidence. | Nonreligious identity can be positive for “nonreligion” studies but negative for “religious organization” binary. |

### Derived binary rule

- **Religious positive** if any of the following have explicit evidence spans: `religious_purpose_explicit`, `religious_service_content`, or high-confidence `religious_identity_or_affiliation`.
- **Probable positive / review** if only `religious_governance_or_authority`, `religious_resources_or_networks`, or `administrative_religion_prior` is present.
- **Ambiguous / audit** if only spiritual language, historical roots, saint names, generic faith/values language, or faith-based partner references appear.
- **Nonreligious/negative** if no evidence is found or text explicitly states secular/nonreligious/no religious affiliation, while preserving “insufficient information” when text is too short.

### Recommended LLM prompt skeleton

```text
You are an expert social-science annotator coding nonprofit organization text.

Task: Classify whether the organization expresses an observable religious or spiritual identity, mission, affiliation, governance tie, or religious program/activity. Use only the provided text fields. Do not infer hidden beliefs or religiosity.

Definitions:
- Religious positive: explicit evidence that the organization’s identity, mission, affiliation, governance, or program content is derived from or tied to a religious/spiritual tradition or religious practice.
- Nonreligious/negative: no explicit religious/spiritual evidence in the provided text, or the text is purely secular/civic/service-oriented.
- Ambiguous/review: weak or context-dependent evidence only (e.g., saint name, generic faith/spiritual language, historical roots, faith-based partner mention, ministry/mission without religious co-term).

Code these domains separately: religious_purpose_explicit, religious_identity_or_affiliation, religious_service_content, religious_governance_or_authority, religious_resources_or_networks, spiritual_or_faith_inspired_ambiguous, administrative_religion_prior, negative_or_secular_evidence.

Rules:
1. Require short evidence spans for every positive or ambiguous domain.
2. Do not count generic words like mission, ministry, serve, compassion, community, values, grace, mercy, fellowship, or stewardship unless tied to a religious tradition, practice, scripture, worship, denomination, congregation, or faith body.
3. Separate organization-level identity from program-level religious content.
4. Mark historical religious roots or religiously founded but currently secular service text as ambiguous/review unless current mission/activity/affiliation is explicit.
5. Include non-Christian evidence: Jewish, Muslim/Islamic, Hindu, Buddhist, Sikh, Indigenous, interfaith, spiritual, humanist/atheist/nonreligious.

Return JSON with:
{
  "binary_label": "religious" | "nonreligious" | "ambiguous_review" | "insufficient_information",
  "confidence": 0-1,
  "domains_present": [...],
  "evidence_spans": [{"domain": "...", "field": "name|mission|activity|admin", "quote": "..."}],
  "boundary_notes": "..."
}
```

## Synthesis of Authoritative Definitions

| Construct | Synthesis for this classifier |
|---|---|
| Religion | No universal definition; scholarship distinguishes substantive, functional, mixed, and polythetic definitions. For classification, use a stipulative measurement definition: observable ties to religious/spiritual tradition, practice, symbols, institutions, or mission in text. |
| Religious identity | Self-presentation or organizational identity with a religion/tradition/denomination/congregation/order/faith community; may appear in name, mission, tagline, history, statement of belief, affiliation, or administrative records. |
| Religious mission | Purpose or program goal explicitly connected to worship, religious formation, evangelism/missionary activity, scripture, prayer, sacred obligations, religious values with named tradition, or service framed as religious vocation. |
| Faith-based organization | Not binary in classic literature; an organization may be faith-permeated, faith-centered, faith-affiliated, faith-background, faith-secular partnership, or secular. Code degrees/domains first. |
| Religious organization | A useful organizational definition: identity and mission derived from a religious or spiritual tradition and operating as a nonprofit/voluntary entity. |
| Faith community/congregation | A community organized around worship/practice/teaching; NCS operationalizes it through attendance nominations, formal affiliation, denomination/association, clergy, worship services, and religious life. |
| Spirituality | Often personal, experiential, sacred/transcendent, or meaning-oriented; in organization text, spiritual language is ambiguous unless tied to concrete tradition, practice, or program. |
| Nonreligion | Survey sources operationalize as atheist, agnostic, or nothing in particular/no religion; for organization text, nonreligious usually means absence of religious evidence or explicit secular/humanist/atheist/freethought identity. |
| Secular/nonreligious organization | Organization whose mission/activity/identity is not religiously grounded in observable text. “Secular” can also mean church-state separation or a positive worldview; do not assume anti-religion. |

## Pitfalls and Boundary Cases

- **Christian/English bias**: U.S. nonprofit triggers over-detect Christian terms and under-detect Jewish, Muslim, Hindu, Buddhist, Sikh, Indigenous, interfaith, and embedded/nonverbal religious expression.
- **Saint names**: `St.`, `Saint`, `San`, `Santa`, named saints, and religious founders may signal geography, hospital/school naming, or historic origin rather than current religious mission.
- **Ministry/mission ambiguity**: `mission`, `ministry`, `fellowship`, `stewardship`, `grace`, `mercy`, `spirit`, `healing`, `chapel`, `temple` can be religious or secular depending on co-text.
- **Religiously founded but secularized institutions**: Hospitals, universities, schools, and relief agencies may have faith origins but secular operations; code identity/history separately from current service content.
- **Spiritual but nonreligious language**: Meditation, wellness, sacred ecology, spiritual healing, and higher power can be religious/spiritual but not denomination-based; keep separate unless the binary definition includes spirituality.
- **Ethnic/religious overlap**: Jewish, Sikh, Hindu, Muslim, Arab, Indian, Pakistani, Tibetan, etc. can signal ethnicity/culture as well as religion; require mission/activity/identity context.
- **Nonresponse/social desirability**: Survey wording shows affiliation, practice, belief, and salience are distinct and sensitive to mode/order; do not collapse them conceptually.
- **Organization vs program level**: A religious organization may run secular programs; a secular organization may run chaplaincy/spiritual-care programs. Code both levels.
- **Administrative label incompleteness**: NTEE X misses many faith-affiliated schools/hospitals/social services; Form 990 excludes many churches; IRS/NCCS codes can be stale or single-purpose.
- **LLM prompt drift**: Model version and wording can change labels. Preserve prompts, parameters, raw outputs, evidence spans, and prompt-disagreement flags.

## Read-First List and Direct Implications

1. **Sider & Unruh (2004)** — use the sixfold typology and organization/program separation.
2. **GivingTuesday `religious_org_v1` + dataset** — closest direct nonprofit mission/activity religious classifier; inspect examples for triggers and errors.
3. **NCS cumulative codebook** — model denomination/affiliation and congregation-level wording.
4. **Pew + GSS + WVS + ARDA** — borrow “if any,” nonreligious categories, and distinct identity/practice/salience/belief wording.
5. **Ebaugh/Chafetz/Pipes and ORE/IU coding schema** — use visible religiosity and identity/activity evidence criteria.
6. **Ma, Fyall, Santamarina, Nonprofit Mission Classifiers, NCCS/IRS** — use name/mission/program text and administrative labels as noisy priors, not truth.
7. **Codebook LLMs + Törnberg + Atreja et al. + Smith et al.** — make a semi-structured prompt codebook, use JSON, include uncertainty, run prompt variants, and aggregate with weak supervision.

**Immediate prompt/codebook recommendations**

- Build a human-readable and machine-readable codebook with definitions, inclusion/exclusion rules, positive/negative examples, and boundary examples.
- Require evidence spans and a field source for every positive/ambiguous label.
- Use multi-domain labels first; derive binary later with documented thresholds.
- Run at least 3 prompt variants and log prompt disagreement as uncertainty.
- Add weak-supervision labeling functions for high-precision tradition/practice triggers, administrative priors, ambiguous-review triggers, and negative anti-discrimination/secular filters.

## Coverage Notes

- **Databases searched**: local prior reports; web search over SAGE, Wiley, Springer, Cambridge Core, ACL Anthology, ICWSM/AAAI, Sociologica, arXiv, Pew, GSS/NORC, WVS/EVS/GESIS, NCS/ARDA/ICPSR, NCCS/Urban, IRS, Nonprofit Mission Classifiers, GitHub/Hugging Face, SEP, OSCE/ODIHR.
- **Date range**: classic definitional roots through 2026; main operational sources 2001-2026; main LLM/CSS sources 2023-2026.
- **Search queries used**: listed in Search Strategy.
- **Gaps identified**: Very few peer-reviewed sources expose a full LLM prompt for religious-vs-nonreligious nonprofit mission classification. GivingTuesday is close but public prompt text is partial. Non-Christian organizational religious-expression codebooks are thinner than Christian/U.S. FBO sources. Proprietary Candid/GuideStar/Cause IQ materials may contain useful coding but are not fully open.

## Handoff: Citation List

| Citation | DOI/URL | Short Title |
|---|---|---|
| Sider, R. J., & Unruh, H. R. (2004). Typology of Religious Characteristics of Social Service and Educational Organizations and Programs. | https://doi.org/10.1177/0899764003257494 | Sider-Unruh Typology |
| Bielefeld, W., & Cleveland, W. S. (2013). Defining Faith-Based Organizations and Understanding Them Through Research. | https://doi.org/10.1177/0899764013484090 | Defining FBOs |
| Smith, S. R., & Sosin, M. R. (2001/2002). The Varieties of Faith-Related Agencies. | https://doi.org/10.1111/0033-3352.00137 | Faith-Related Agencies |
| Ebaugh, H. R., Pipes, P. F., Chafetz, J. S., & Daniels, M. (2003). Where’s the Religion? | https://doi.org/10.1111/1468-5906.00191 | Where’s the Religion |
| Ebaugh, H. R., Chafetz, J. S., & Pipes, P. (2006). Where’s the Faith in Faith-Based Organizations? | https://doi.org/10.1353/sof.2006.0086 | FBO Religiosity Measures |
| Ma, J. (2021). Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector. | https://doi.org/10.1177/0899764020968153 | Ma NTEE Classifier |
| Fyall, R., Moore, M. K., & Gugerty, M. K. (2018). Beyond NTEE Codes. | https://doi.org/10.1177/0899764018768019 | Mission Statement Coding |
| Santamarina, F. J., Lecy, J. D., & van Holm, E. J. (2023). How to Code a Million Missions. | https://doi.org/10.1007/s11266-021-00420-z | Million Missions |
| Nonprofit Open Data Collective. Nonprofit Mission Classifiers. | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Mission Classifiers |
| NCCS/Urban Institute. National Taxonomy of Exempt Entities (NTEE) Codes. | https://nccs.urban.org/nccs/resources/ntee/ | NTEE Docs |
| IRS. Exempt Organizations Business Master File documentation. | https://www.irs.gov/pub/irs-soi/eo-info.pdf | IRS BMF |
| GivingTuesday. Religious Orgs Segmentation Model. | https://huggingface.co/GivingTuesday/religious_org_v1 | Religious Org Model |
| GivingTuesday. Religious Orgs Training Dataset. | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | Religious Org Dataset |
| Chaves, M., et al. National Congregations Study Cumulative Codebook. | https://sites.duke.edu/ncsweb/files/2020/09/NCS-I-IV-Cumulative-Codebook_FINAL_8Sept2020.pdf | NCS Codebook |
| NORC. GSS 2024 Codebook. | https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook%20R2.pdf | GSS Codebook |
| Pew Research Center. Religion survey methodology/questionnaires. | https://www.pewresearch.org/religion/ | Pew Religion Wording |
| World Values Survey Association. WVS Wave 8 Questionnaire. | https://www.worldvaluessurvey.org/documents/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf | WVS Questionnaire |
| Bradburn, N., et al. (2014). The ARDA's Measurement Wizard. | https://doi.org/10.1111/jssr.12131 | ARDA Measurement Wizard |
| Halterman & Keith / Codebook LLMs authors. Codebook LLMs. | https://www.cambridge.org/core/journals/political-analysis/article/codebook-llms-evaluating-llms-as-measurement-tools-for-political-science-concepts/7B323A0E47F782F2698A0AE849EA00DE | Codebook LLMs |
| Törnberg, P. (2024). Best Practices for Text Annotation with Large Language Models. | https://doi.org/10.6092/issn.1971-8853/19461 | LLM Annotation Best Practices |
| Atreja, S., et al. (2025). What’s in a Prompt? | https://doi.org/10.1609/icwsm.v19i1.35807 | Prompt Design Experiment |
| Ziems, C., et al. (2024). Can Large Language Models Transform Computational Social Science? | https://doi.org/10.1162/coli_a_00502 | LLMs for CSS |
| Smith, R., Fries, J. A., Hancock, B., & Bach, S. (2024). Language Models in the Loop. | https://doi.org/10.1145/3617130 | Prompted Weak Supervision |
| Stanford Encyclopedia of Philosophy. The Concept of Religion. | https://plato.stanford.edu/entries/concept-religion | Concept of Religion |
| Springer Encyclopedia of Global Religion. Religious Organizations. | https://link.springer.com/rwe/10.1007/978-3-319-31816-5_2514-1 | Religious Organizations Definition |

## Handoff: Datasets Mentioned

| Dataset Name | Paper Reference | Source URL (if found) | Notes |
|---|---|---|---|
| GivingTuesday Religious Orgs Training | GivingTuesday model card | https://huggingface.co/datasets/GivingTuesday/religious_orgs_training | 498 visible rows; fields include EIN, name, mission, activity_1-3, classification; useful direct examples and boundary cases. |
| IRS Form 990 e-file XML | Ma; Fyall; Santamarina; Nonprofit Mission Classifiers | https://www.irs.gov/charities-non-profits/form-990-series-downloads | Mission and program service text; many churches exempt from annual filing. |
| IRS Business Master File | IRS; NCCS; Ma | https://www.irs.gov/pub/irs-soi/eo-info.pdf | NTEE, subsection, filing requirements, old activity codes including religious activities. |
| NCCS BMF/Core Files and NTEE | NCCS/Urban; Ma | https://nccs.urban.org/ | Research-ready nonprofit classifications; NTEE_IRS vs NTEE_NCCS distinctions. |
| IRS 1023-EZ purpose codes | Santamarina; Nonprofit Mission Classifiers | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/data/ | Binary religious purpose and other purpose codes; useful weak labels. |
| National Congregations Study | Chaves et al.; NCS/ARDA/ICPSR | https://www.nationalcongregationsstudy.org/data-documentation | Congregation benchmark; denomination, formal affiliation, worship/program activity items. |
| General Social Survey | NORC | https://gss.norc.org/ | Religion variables: affiliation/preference, denomination, attendance, prayer, spirituality, religious person, born-again. |
| Pew Religious Landscape Study / ATP / NPORS | Pew Research Center | https://www.pewresearch.org/religion/ | Exact affiliation/nonreligion/denomination/born-again/practice wording. |
| World Values Survey / EVS | WVS/EVS/GESIS | https://www.worldvaluessurvey.org/ | Cross-national religion, God importance, attendance, prayer, denomination, religious person wording. |
| ARDA Measurement Wizard | Bradburn et al. | https://www.thearda.com/data-archive/measurements | Bank of full religion/spirituality question wording and categories. |
| Candid/GuideStar Philanthropy Classification System | NCCS docs; prior reports | https://taxonomy.candid.org/ | Includes subject/population/organization-type/auspice concepts; access/licensing may constrain use. |
| Nonprofit Mission Classifiers benchmark data | Lecy/van Holm/Santamarina | https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/ | Replication docs for mission/activity text classification using NTEE and purpose taxonomies. |
