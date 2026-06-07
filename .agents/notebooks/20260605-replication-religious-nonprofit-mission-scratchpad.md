# Replication Scratchpad: Religious Nonprofit Mission Classification

Created: 20260605
Topic Keywords: religious nonprofits; mission classification; nonprofit taxonomies; Form 990; faith-based organizations

---

## Problem Definition

### Research Questions

- Primary: Where are replication packages, datasets, code repositories, taxonomies, model artifacts, codebooks, and reproducibility materials for academic and official work on religious organization classification, nonprofit mission classification, charity/organization text classification, faith-based organizations, and nonprofit taxonomies?
- Secondary: Which artifacts are official administrative data vs research replication packages vs proprietary products, and which are reusable for a binary religious mission classifier?

### Input Mode

- [x] DOI/URL/citations from literature-seeker
- [x] Freeform paper citations
- [x] Topic string (bounded bootstrapping)

### Upstream Context Used

- `.agents/docs/20260605-literature-synthesis-map.md`
- `.agents/docs/20260605-literature-religious-nonprofit-classification.md`
- `.agents/docs/20260605-literature-religious-identity-prompts.md`

### Success Criteria

- [x] Use upstream citation lists as bounded scope for named academic papers and official artifacts.
- [ ] Search Dataverse, OpenICPSR, OSF, Zenodo, and GitHub for priority papers/artifacts before marking missing.
- [ ] Inspect open package contents and record file manifests where APIs permit.
- [ ] Produce output report under `.agents/docs/` with required handoff blocks and explicit empty/not-found markers.

## Research Log

### Entry 1: Upstream handoff discovery

THOUGHT: I need to bound the package search around the existing same-topic literature reports so I do not silently expand beyond the project scope.
ACTION: Read upstream reports and handoff citation lists from `.agents/docs/20260605-literature-religious-nonprofit-classification.md`, `.agents/docs/20260605-literature-synthesis-map.md`, and `.agents/docs/20260605-literature-religious-identity-prompts.md`.
OBSERVATION:

- Source: local `.agents/docs/` reports.
- Repository: Upstream handoff discovery.
- Key Facts: Priority papers include Ma (2021), Fyall et al. (2018), Litofcenko et al. (2020), Santamarina et al. (2023), Sider & Unruh (2004), Bielefeld & Cleveland (2013), Smith & Sosin (2001/2002), Becker/Ebaugh et al., NTEE/NCCS, IRS, NCS/ARDA, UK Charity Classification, and Nonprofit Open Data Collective materials.
- Confidence: High because these are complete same-day literature-seeker reports.
- New Questions: Which of these have public replication packages or only article PDFs/codebooks?

### Entry 2: Structured archive API pass

THOUGHT: I need to search Dataverse, OSF, and Zenodo for exact priority titles before using web fallback.
ACTION: Search[Dataverse `/api/v1/search`, Zenodo `/api/records`, OSF `/v2/nodes` for Ma, Fyall, Litofcenko, Santamarina, NODC, ANGEL, UK-CAT, NCS, Sider/Unruh, Bielefeld/Cleveland].
OBSERVATION:

- Source: Dataverse, Zenodo, OSF APIs.
- Repository: Dataverse/Zenodo/OSF.
- Key Facts: OSF title-filter returned zero for priority artifacts; Zenodo exact/fielded title queries returned zero target matches; Dataverse broad title queries were noisy but no top hits except Santamarina found by web discovery with DOIs.
- Confidence: Medium; Dataverse search is noisy and Zenodo rate-limited once, but exact fielded follow-up returned zero target title matches.
- New Questions: Need web fallback and package-specific DOI inspection.

### Entry 3: Ma (2021) package

THOUGHT: I need to verify Ma's working directory and contents because the paper claims GitHub availability.
ACTION: Search[`"Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector" Ji Ma data code`] and Verify[GitHub repo page].
OBSERVATION:

- Source: https://github.com/ma-ji/npo_classifier
- Repository: GitHub.
- Key Facts: Public repo with API, dataset/UCF, output, reference, script folders; README links OSF preprint DOI and NTEE classifier materials; GitHub API was rate-limited, so file manifest comes from repository page and search snippets.
- Confidence: High for existence and top-level contents; Medium for file-level completeness because API inspection was blocked.
- New Questions: Need caution that some data are author-hosted/remapped via jima.me rather than archived DOI.

### Entry 4: Santamarina et al. / Bespoke NPO Taxonomies

THOUGHT: I need file-level manifests for the replication Dataverse datasets.
ACTION: Fetch[replication site and GitHub repo] + List files[Dataverse persistent IDs 10.7910/DVN/BL6XLW, EO2HIM, 4GZJSK].
OBSERVATION:

- Source: https://fjsantam.github.io/bespoke-npo-taxonomies/; https://github.com/fjsantam/bespoke-npo-taxonomies; Harvard Dataverse DOIs.
- Repository: Dataverse + GitHub.
- Key Facts: Raw/cleaned IRS 1023-EZ data DOI has 12 files; preprocessed DOI has 3 large CSVs; classification DOI has 70 RData/RDS files including `Orgpurposereligious` and `ntmaj10rel` results. GitHub hosts replication steps/docs/data.
- Confidence: High; verified by Dataverse API file listings.
- New Questions: None critical.

### Entry 5: Official/admin datasets and taxonomies

THOUGHT: I need distinguish official public data from research replication packages and proprietary products.
ACTION: Fetch[IRS Form 990 downloads, NCS/ICPSR, NCCS/NTEE fallback, Candid PCS, Cause IQ, UK-CAT].
OBSERVATION:

- Source: IRS, ICPSR/NCS, NCCS/Urban, Candid, Cause IQ, UK-CAT.
- Repository: Official docs, ICPSR, GitHub.
- Key Facts: IRS Form 990 e-file XML is official and updated May 20 2026; ICPSR NCS v6 DOI updated Jun 3 2025; NTEE X religion-related taxonomy available via NCCS widgets; PCS is CC BY 4.0; Cause IQ is proprietary with Types/Issues/Characteristics, NTEE, NAICS; UK-CAT provides CC BY 4.0 code/data including religion tags.
- Confidence: High for official pages and access caveats.
- New Questions: None critical.

### Entry 6: Not-found and restricted artifacts

THOUGHT: I need identify papers with no public package after target repository coverage.
ACTION: Search[OpenICPSR site-scoped exact title query plus web searches for Fyall, Litofcenko, Sider/Unruh, Bielefeld/Cleveland, Smith/Sosin, Becker/Ebaugh].
OBSERVATION:

- Source: OpenICPSR web search, journal pages, author/institution pages.
- Repository: OpenICPSR/GitHub/web.
- Key Facts: No public replication package found for Fyall et al. (2018), Litofcenko et al. (2020), Sider & Unruh (2004), Bielefeld & Cleveland (2013), Smith & Sosin (2001/2002), Becker/Ebaugh et al. ANGEL has a GitHub repo but README says code “Coming soon”; only paper PDF is present.
- Confidence: Medium; package absence is hard to prove, but all target repositories and web fallback were checked.
- New Questions: For classics, codebooks/scales are in articles rather than public machine-readable archives.

## Draft Output

### Packages Found

Ma GitHub; Santamarina Dataverse/GitHub; NODC mission codes/taxonomies/IRS e-file builder; UK-CAT GitHub/data; NCS ICPSR; GivingTuesday HF model/dataset; Candid PCS; NCCS NTEE; IRS Form 990/BMF/1023-EZ official pages; Cause IQ proprietary docs.

### Repository Coverage

Dataverse, OSF, Zenodo APIs searched; OpenICPSR site-scoped web searched; GitHub and project pages searched/fetched; official documentation fetched.

### Not Found

No package found for Fyall et al. (2018), Litofcenko et al. (2020), Sider & Unruh (2004), Bielefeld & Cleveland (2013), Smith & Sosin (2001/2002), Becker/Ebaugh et al. ANGEL code not yet released despite public placeholder repo.
