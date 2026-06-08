## Annex A — Full research: Mission Quality Characterization (sub-agent `aa42836beb4803b01`)

_Verbatim, complete output of the read-only quality-characterization data pass. This is the empirical basis for the §2.1 sample-construction design._

**Data:** `missions_cross_section.parquet` — 560,351 EIN2 (one row each, no nulls, lowercased ASCII). Target = `LONGEST_MISSION` (word count: median 22, p90 66, max 204). Joined to `bmf_unified_processed.parquet` on EIN2 via `NTEE_IRS[0]` = major group. **Join rate 99.4%** (3,238 EIN2 unmatched + non-alpha codes → 6,805-row `?` bucket, reported as a footnote, never a 27th group).

### 1. Quality rubric (operational, computable)

Score **Q (max 6.0)** = weighted sum of regex/heuristic features on the lowercased text:

| Component                           | Points              | Rule                                                                                                                |
| ----------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `s_len` (length, capped sub-linear) | 0 / 0.5 / 1.0 / 1.5 | <4w=0, 4–7=0.5, 8–14=1.0, ≥15w=1.5                                                                                  |
| `f_purpose`                         | +1.0                | action verb: provide/support/promote/serve/educate/deliver/fund/preserve…                                           |
| `f_benef`                           | +1.0                | named beneficiary: children, veterans, low-income, patients, homeless, animals…                                     |
| `f_acts`                            | +1.0                | named activity/program: scholarships, food bank, housing, research, counseling, camps…                              |
| `f_specific`                        | +1.0                | **specificity, length-independent**: beneficiary AND activity both present                                          |
| `s_clause`                          | +0.5                | ≥2 clauses/sentences (`[.;,]`)                                                                                      |
| **Penalties**                       |                     | boilerplate 501(c)/IRC enumeration −1.5; vague-only template ("to promote the general welfare") −2.0; <4 words −1.0 |

The length cap (1.5 max) is deliberate so a long boilerplate PTA charter cannot reach HIGH on length alone, while `f_specific` rewards a concise, concrete mission. Thresholds were **locked only after reading boundary examples** at each cut.

**Tiers:** HIGH `Q ≥ 5.0` · MEDIUM `3.0–4.5` · LOW `< 3.0`.

### 2. Tier quantification

**Whole corpus (560,351):** HIGH **178,863 (31.9%)** · MEDIUM **216,231 (38.6%)** · LOW **165,257 (29.5%)**.

#### HIGH-quality pool per NTEE major group (26 rows, A–Z)

| MG  | Name           |  Total |    HIGH | HIGH% |
| --- | -------------- | -----: | ------: | ----: |
| A   | Arts           | 56,170 |  12,211 | 21.7% |
| B   | Education      | 94,366 |  31,960 | 33.9% |
| C   | Environment    | 12,711 |   2,393 | 18.8% |
| D   | Animal         | 16,363 |   4,992 | 30.5% |
| E   | Health         | 32,140 |  14,244 | 44.3% |
| F   | Mental Health  | 13,457 |   6,062 | 45.0% |
| G   | Diseases       |  9,768 |   4,086 | 41.8% |
| H   | Med Research   |  3,935 |   1,589 | 40.4% |
| I   | Crime/Legal    |  8,401 |   3,188 | 37.9% |
| J   | Employment     |  5,245 |   2,127 | 40.6% |
| K   | Food/Agri      |  7,915 |   2,474 | 31.3% |
| L   | Housing        | 22,652 |  12,989 | 57.3% |
| M   | Public Safety  | 10,575 |   2,044 | 19.3% |
| N   | Recreation     | 43,292 |   9,905 | 22.9% |
| O   | Youth Dev      | 18,696 |   6,680 | 35.7% |
| P   | Human Services | 73,777 |  30,068 | 40.8% |
| Q   | International  | 13,348 |   4,209 | 31.5% |
| R   | Civil Rights   |  4,207 |   1,164 | 27.7% |
| S   | Community      | 21,916 |   5,903 | 26.9% |
| T   | Philanthropy   | 26,429 |   7,402 | 28.0% |
| U   | Sci/Tech       |  3,081 | **581** | 18.9% |
| V   | Social Sci     |  1,036 | **257** | 24.8% |
| W   | Public Benefit |  7,952 |   2,132 | 26.8% |
| X   | Religion       | 40,749 |   6,679 | 16.4% |
| Y   | Membership     |  1,040 | **200** | 19.2% |
| Z   | Unknown        |  4,325 |   1,372 | 31.7% |

**Thin pools (<500 HIGH):** **V (257)** and **Y (200)**; **U (581)** borderline. (Footnote: the `?` non-group bucket has 1,952 HIGH.) X has the lowest HIGH% (16.4%) — most religious orgs file bare labels — but still yields **6,679 HIGH** religious missions, a healthy positive pool.

### 3. Example bank (verbatim, full text)

**HIGH** (one per group, label = NTEE letter):

- **[A]** we are a non-profit concert production company that raises money for music education and gives opportunities for youth to perform on the big stage
- **[B]** private school satellite program (psp) for 35 families. program provides support for home educating families through record keeping, counseling, testing, transcript services etc
- **[C]** the purpose is to reconnect individuals to their human and natural environment through educational outreach, environmental activism and the creation of sustainable community processes.
- **[D]** animal control and adoption - serving surrounding cities with animal impounding, vaccinations, licensing, adoption, spaying & neutering.....areas services include ontario, montclair and rancho cucamonga.
- **[E]** to be the premier healthcare institution serving our greater community by providing excellence in clinical care and patient safety, education, clinical research, and service.
- **[F]** recovery program for adults, housing, food, transportation, medical, counseling provided to recovering alcoholics & substance abuse adults to provide new start on life
- **[I]** promote gender equality through leadership education and research; provide scholarship and awards to women; provide networking opportunities for all women in federal law enforcement.
- **[L]** to provide low income elderly persons with housing designed to meet their needs, to promote their health, security and happiness.
- **[N]** develop hockey programs and provide hockey training to underserved teams. maintain a state-wide hockey newsletter via the internet. manage an ice arena.
- **[O]** to promote the growth and development of children in the community by providing activities that build leadership while promoting fellowship and community service.
- **[P]** to provide scholarships to high school seniors in the charlotte mecklenburg school system (north carolina) to further education at institution of higher learning.
- **[Q]** charitable and educational purposes to enhance the level of education of public students by providing computer and information technology to public schools in mexico
- **[R]** promoting and defending the rights of lesbian, gay, bisexual. transgender, and intersex (""lgbti"") individuals by awarding grants to support grassroots lgbti organizations around the world.
- **[S]** advocacy throughout the legal process for children in foster care and the provision of supervised visitation services for these children while in care.
- **[T]** we partner with people and organizations to provide resources for the community focused on education, income, and health to improve lives in utah county.

**MEDIUM** (decent but thinner on one feature, vaguer, or single-clause):

- **[X]** manage and operate a christian camp and conference center.
- **[Q]** to raise funds for uganda's poor people for deep-water wells, assist with purchases of prescribed meds for the ugandian poor, and assist orphans.
- **[B]** assisting clients in learning how to deal with life conflicts, crises, behavior problems and substance abuse.
- **[O]** provide opportunities for boya and girls aged 4 to 18 to participate in organized baseball and softball activities.
- **[B]** support organization for benton public school football program
- **[N]** to teach baseball to young men and have a spiritual and moral influence on them.
- **[N]** we are a 100% volunteer organization serving area youth. we use baseball to teach on teamwork, fair play, sportsmanship. we give youth a structured athletic outlet.
- **[B]** to support louisiana state university at alexandria and any affiliated intercollegiate athletics program.

**LOW** (fragments, bare labels, no purpose+beneficiary+activity):

- **[A]** preservation and documentation of artifacts of the community
- **[C]** stewardship of dam, wildlife and habitat on the pond
- **[B]** community service projects
- **[N]** train competitive youth soccer
- **[E]** emergency transport
- **[B]** before and after school care at the school
- **[Z]** encourage, educate and inspire in every area of daily life.
- **[I]** the networks's mission is to create a more effictive way to assist low-risk juvenile offenders who do not belong in court.

### 4. Religious signal across the NTEE spectrum

**Lexicon used** (broad prior, case-insensitive word boundaries): christ/christian/jesus/gospel/bible/biblical/scripture/church/congregation/parish/ministry/ministries/missionary/evangel/catholic/baptist/methodist/lutheran/presbyterian/episcopal/pentecostal/jewish/synagogue/torah/rabbi/islam/muslim/mosque/quran/buddhist/hindu/temple/salvation army/worship/prayer/holy/sacred/seminary/chapel/diocese/orthodox/spiritual/faith(-based). Overall hit rate **11.2%** of the corpus; by group it ranges from X=67.5% down to D=0.4%.

**Strong religious signal exists in HIGH-quality missions across every sector tested.** Religious HIGH count per sector: B=2,974, P=2,908, E=1,311, Q=648, **A=302**, L=952. Representative verbatim contrasts:

- **B Education** — RELIGIOUS: _"the king's university is a higher education and ministry leadership training center that exists to provide biblically and theologically sound preparation that equips men and women to become the next generation of christ-like, spirit-empowered, transformational leaders for the global church."_ · SECULAR: _"the mission of the early learning coalition of florida's heartland, inc. is to support families and children in accessing high quality early care and education services via school readiness and voluntary prekindergarten programs."_
- **P Human Services** — RELIGIOUS: _"present the gospel of jesus christ and provide education, relief, refuge and love to burmese refugees. bring awareness to the genocide that is taking place in burma..."_ · SECULAR: _"to help families and individuals improve their money management skills through educational workshops, financial, budget and credit counseling, and debt management."_
- **E Health** — RELIGIOUS: _"to glorify christ by providing quality, charitable health care to low income uninsured and underinsured residents of northeast tennessee and southwest virginia."_ · SECULAR: _"our mission is to assist individuals with intellectual and/or developmental disabilities to get the supports and services they need to live a quality life."_
- **Q International** — RELIGIOUS: _"villages in partnership (vip) is a faith-based christian organization that works together with churches, schools, businesses and community organizations to devise and implement sustainable development projects (e.g wells, schools, agriculture and health programs)"_ · SECULAR: _"to provide for the support and education of aids orphans or other disadvantaged youth in tanzania."_
- **A Arts** — RELIGIOUS: _"to provide kindergarten and pre-school education for preschool children in a church oriented, christian, environment."_ · SECULAR: _"to provide artistic and educational programming of classical string quartet chamber music, with performances around the united states, educational workshops with university music students..."_
- **L Housing** — RELIGIOUS: _"to be the body of christ in our community providing food, shelter, clothing and other basic needs to those are struggling and offer rehabilitative services in the name of jesus christ"_ · SECULAR: _"provide individuals who have developmental disabilities with low income housing and services designed to meet their individual needs."_

### 5. Hard / boundary cases among HIGH-quality missions (verbatim, with prevalence)

- **Saint-named, faith-founded-but-often-now-secular institutions — 685 (0.38% of HIGH):** _"st joseph's ancillary services, inc. will improve the health of all we serve through community-owned health care services that set the standard for high-quality, compassionate care."_ · _"providing multi-specialty medical services to a medically underserved community for st. luke's cornwall hospital."_ These read fully secular apart from the saint name — true coin-flips.
- **"Spiritual but not religious" / wellness — 946 (0.53% of HIGH):** _"the society leads women and men to join together to grow spiritually by offering person-to-person service to those who are needy and suffering."_ · _"compassion that compels offers encouraging and uplifting studies for the purpose of healing women spiritually..."_ "Spiritual" with no named tradition.
- **Generic "ministry/mission/compassion" without a named tradition — ~1,268 (0.71%):** e.g. _"the mission is to provide ministry services to the community"_ — "ministry" is the only cue. (Note: the broader words _mission/serve/compassion_ appear in ~22% of HIGH but are overwhelmingly secular usage, so they are NOT a religious signal on their own.)
- **Boilerplate religious legal clause — 191 (0.11%):** _"the organization mission is exclusively for religious, charitable and educational purposes to promote healthy biblical relationships through (but not limited to) teaching, coaching and training individuals."_ — 501(c)(3) enumeration; "religious" is legal scaffolding, not necessarily the org's substance.
- **Faith-heritage/values framing (culturally religious, possibly secular ops) — ~684 (0.38%):** _"…infused with values emphasizing judeo-christian principles and traditions"_ (Jewish/Catholic-heritage hospitals and schools).

These ambiguous-but-clean-text patterns are exactly what the **human gold test set must retain**.

### 6. Synthesis for sampling

- **Threshold recommendation:** Cherry-pick from **HIGH (Q ≥ 5.0)** as the silver pool — boundary reading confirms these are uniformly concrete (purpose + beneficiary + activity). Top-MEDIUM (Q = 4.5) is a clean overflow reservoir for thin strata.
- **Positive-class interaction (decisive number, at the locked Q ≥ 5.0 threshold):** strong-tradition religious missions have a **30.0% HIGH-rate vs 31.9% for clearly-secular** missions — a 1.9pp gap, **not** a disproportionate strip. The gap is **fully explained by 6,827 bare-label churches** (≤6 words: "church", "christian ministries") that correctly land in LOW — useless training text. Descriptive positives are in fact slightly _longer_ than the corpus (median 28 vs 22 words). **Recommendation:** add a light positive-protective rule — rescue MEDIUM rows that carry a strong-tradition lexicon hit **and** a concrete purpose verb — to top up the X positive pool without re-admitting bare labels.
- **Feasibility:**
  - **~400 gold (~15/group):** comfortably supported — every one of the 26 groups has ≥200 HIGH.
  - **~20k silver:** covered in aggregate (178k HIGH) but an equal ~769/group is impossible for **V (257), Y (200), U (581)**. Use proportional/capped HIGH-only allocation for fat strata; for V/Y/U take the entire HIGH pool and top up from top-MEDIUM (Q = 4.5).
