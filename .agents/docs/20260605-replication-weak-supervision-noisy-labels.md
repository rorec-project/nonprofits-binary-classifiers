---
created: 20260605
agent: replication-seeker
scratchpad: .agents/notebooks/20260605-replication-weak-supervision-noisy-labels-scratchpad.md
status: complete
title: Replication - Weak Supervision Noisy Labels
topic: LLM annotation; weak supervision; noisy labels; transformer fine-tuning; reproducibility artifacts
---

# Replication: LLM Annotation, Weak Supervision, Noisy Labels, and Transformer Fine-Tuning

## Search Strategy and Exact Queries Used

Scope was bounded by `.agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md`, the synthesis map, and the user priority list. I searched Dataverse, OSF, Zenodo, OpenICPSR, GitHub, ACL/OpenReview/PapersWithCode-style pages, and author/project pages.

### Repository/API queries

- Dataverse API: `q="<paper title> <authors>"&type=dataset&per_page=5` for each priority paper; examples:
  - `Gilardi Alizadeh Kubli ChatGPT Outperforms Crowd Workers text annotation`
  - `Heseltine Clemm von Hohenberg Large language models substitute human experts annotating political text`
  - `WRENCH Comprehensive Benchmark Weak Supervision Zhang Yu Li`
  - `Noise-Robust Fine-Tuning Pretrained Language Models External Guidance LAFT Wang Tan Guo Li`
  - `Feeding LLM Annotations to BERT Classifiers at Your Own Risk Lu Smith`
- Dataverse file manifests:
  - `https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/PQYF6M`
  - `https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/V2P6YL`
- OSF API: `https://api.osf.io/v2/nodes/?filter[title]=<paper title/authors>&filter[public]=true&page[size]=5` for each priority paper.
- Zenodo API: `https://zenodo.org/api/records?q=<paper title/authors>&size=5&sort=bestmatch` for each priority paper.
- GitHub Contents API / fallback landing pages:
  - `npangakis/gpt_annotate`, `JieyuZ2/wrench`, `snorkel-team/snorkel`, `snorkel-team/snorkel-tutorials`, `uds-lsv/BERT-LNL`, `uds-lsv/transfer-distant-transformer-african`, `SongW-SW/LAFT`, `gtfintechlab/SiDyP`, `Zhen-Tan-dmml/LLM4Annotation`.
- OpenICPSR web queries:
  - `site:openicpsr.org "ChatGPT outperforms crowd" OR "Gilardi" "text-annotation"`
  - `site:openicpsr.org "Automated Annotation with Generative AI Requires Validation" OR "Pangakis"`
  - `site:openicpsr.org "Large Language Models as a Substitute for Human Experts" "Heseltine"`
  - `site:openicpsr.org "WRENCH" "Weak Supervision" OR "Is BERT Robust to Label Noise"`
- Web/project discovery queries:
  - `"Automated Annotation with Generative AI Requires Validation" data code GitHub OSF replication`
  - `"Large Language Models Outperform Expert Coders" Törnberg replication data GitHub OSF`
  - `"WRENCH: A Comprehensive Benchmark for Weak Supervision" GitHub benchmark data`
  - `"Language Models in the Loop: Incorporating Prompting into Weak Supervision" code GitHub data prompts`
  - `"Is BERT Robust to Label Noise" GitHub code data "Hausa" "Yoruba"`
  - `"SaFER" "Robust" "Fine-tuning BERT" "Noisy Labels" GitHub ACL Industry 2023`
  - `"Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance" LAFT GitHub code`
  - `"Feeding LLM Annotations to BERT Classifiers at Your Own Risk" GitHub code data arxiv 2504.15432`
  - `"Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement" GitHub code data`
  - `"BOXWRENCH" weak supervision benchmark GitHub code dataset`

## Verified Packages

| Paper | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|-------|-----------|-----|----------------|---------------|-------------|------------|
| Gilardi, Alizadeh & Kubli (2023), *ChatGPT outperforms crowd-workers for text-annotation tasks* | Harvard Dataverse | https://doi.org/10.7910/DVN/PQYF6M | R, Python | Yes: annotation spreadsheets, ChatGPT outputs, MTurk/evaluation data; tweet text partly limited to IDs | Open, with Twitter data caveat | 62 |
| Heseltine & Clemm von Hohenberg (2024), *LLMs as a substitute for human experts in annotating political text* | Harvard Dataverse | https://doi.org/10.7910/DVN/V2P6YL | R, Python archive | Yes: tweet/candidate CSVs, GPT/human coding result tables | Open | 8 |
| Pangakis, Wolken & Fasching (2023/2024), *Automated Annotation with Generative AI Requires Validation* / human-centered follow-on | GitHub | https://github.com/npangakis/gpt_annotate | Python, notebook | No main validation datasets; software and sample notebook only; paper says source datasets were password-protected or author-obtained | Open code-only / prompt-workflow | 5 top-level files |
| Ratner et al. (2016/2017), Data Programming / Snorkel | GitHub / PyPI | https://github.com/snorkel-team/snorkel | Python | Library tests/docs; not a paper-specific replication dataset | Open code framework | 20+ top-level entries |
| Snorkel tutorials | GitHub | https://github.com/snorkel-team/snorkel-tutorials | Python notebooks/scripts | Yes: tutorial loaders/examples for spam, spouse, crowdsourcing, etc. | Open code/tutorial artifacts | 20+ top-level entries |
| Zhang et al. (2021), *WRENCH* | GitHub + Hugging Face | https://github.com/JieyuZ2/wrench ; https://huggingface.co/datasets/jieyuz2/WRENCH | Python | Yes: 22 weak-supervision benchmark tasks, weak labels/LFs in dataset format | Open benchmark/code | 9 top-level entries; dataset hosted separately |
| BoxWRENCH / *Stronger Than You Think* (2024/2025 adjacent) | GitHub + Google Drive/HF-linked sources | https://github.com/jeffreywpli/stronger-than-you-think | Python | Yes for most datasets; Amazon31 explicitly not released/retracted | Open code; mixed data availability | Not fully API-inspected |
| Zhu et al. (2022), *Is BERT Robust to Label Noise?* | GitHub | https://github.com/uds-lsv/BERT-LNL | Python | No full datasets in repo; points to AG News, IMDB, Hausa/Yorùbá sources | Open code-only plus external data links | 9 top-level entries |
| Hedderich et al. African transfer/distant supervision datasets used by Zhu et al. | GitHub | https://github.com/uds-lsv/transfer-distant-transformer-african/tree/master/data | Python/data TSV | Yes: Hausa/Yorùbá clean/noisy train/dev/test files per README | Open data/code | 4 top-level entries; data subdirs |
| Wang et al. (2023), LAFT | GitHub | https://github.com/SongW-SW/LAFT | None visible beyond README | No usable code observed; repository has only `README.md` | Open but effectively empty / caveat | 1 |
| Ye et al. (2025), SiDyP | GitHub | https://github.com/gtfintechlab/SiDyP | Python, Shell | Yes: includes `datasets/llm/fewshot/mixtral822`; README says Mixtral-labeled data supplied because model deprecated | Open code + LLM-labeled data subset | 7 top-level entries |
| Ziems et al. (2024), *Can LLMs Transform CSS?* | Paper-referenced GitHub data directory; exact repo not surfaced in accessible search | ACL/MIT Press page: https://aclanthology.org/2024.cl-1.8/ | Unknown | Paper states datasets, prompts, and model outputs were released in a GitHub data directory | Claimed open, not verified | Unknown |
| Zhen Tan et al. survey, LLM annotation/synthesis resources | GitHub curated list | https://github.com/Zhen-Tan-dmml/LLM4Annotation | Markdown | Curated paper/dataset links; no primary replication data | Open bibliography/prompt-artifact map | 2 top-level entries |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | Exact-title/author API queries for 14 priority papers | 2 verified priority hits | Gilardi `10.7910/DVN/PQYF6M`; Heseltine `10.7910/DVN/V2P6YL`. Other API hits were broad false positives/noisy. |
| OpenICPSR | `site:openicpsr.org` exact-title/author queries | 0 verified priority hits | Search returned unrelated AI/economics deposits; no priority package found. OpenICPSR remains web-only. |
| OSF | OSF API title filters for 14 priority papers; web query for LLM text annotation replication | 0 priority hits; 1 adjacent OSF artifact (`osf.io/ctgqx`) | Priority packages not found. Adjacent open-source LLM text-annotation guide has OSF replication files but is outside priority list. |
| Zenodo | Zenodo API exact-title/author queries for 14 priority papers | 0 verified priority hits | Many broad false positives; no exact priority replication package. |
| GitHub | Exact title/project searches and Contents API/fallback webfetch | 9 relevant repositories | Verified: `gpt_annotate`, `wrench`, `snorkel`, `snorkel-tutorials`, `BERT-LNL`, African data repo, `LAFT` empty shell, `SiDyP`, `LLM4Annotation`; BoxWRENCH adjacent. |
| ACL/OpenReview/PapersWithCode/author pages | Title searches for WRENCH, SaFER, LAFT, Ziems, Zhu, Ye, Lu | Multiple artifact links | ACL/OpenReview surfaced GitHub links for WRENCH, Zhu, LAFT, Ye; SaFER only says code “will be released,” no official usable repo found. |

## Replication Artifacts: Contents, Last-Updated Signal, Value, Caveats

| Paper/source | Repository/DOI/URL | Contents | Access type | Last-updated signal | Reproducibility value | Caveats |
|---|---|---|---|---|---|---|
| Gilardi et al. 2023 | https://doi.org/10.7910/DVN/PQYF6M | R scripts, Python ChatGPT zero-shot template, human/MTurk/RA annotation data, ChatGPT batch outputs, plots | Open; tweet text constrained | Deposited June 2023; Dataverse V2 visible via API | Very high for LLM-as-annotator prompts, evaluation, cost comparison | Some social-media content only shareable as IDs; older ChatGPT model behavior not exactly reproducible |
| Heseltine & Clemm von Hohenberg 2024 | https://doi.org/10.7910/DVN/V2P6YL | R analysis, Python archive, political tweet CSVs, GPT/human coding results | Open | Dataverse file manifest available; file IDs 8138339–8138346 | Very high for hybrid GPT double-coding + BERTweet downstream comparison | Need inspect `Python.rar` manually; social-media platform/API drift |
| Pangakis et al. 2023/2024 | https://github.com/npangakis/gpt_annotate | `gpt_annotate.py`, sample notebook, README, paper PDF | Open code-only | Repo pushed 2024-09-07; updated 2026-02-27 | High for workflow implementation and consistency-score idea | Validation datasets mostly password-protected/direct-from-authors; not a full replication package |
| Ziems et al. 2024 | https://aclanthology.org/2024.cl-1.8/ | Paper says all datasets, prompts, model outputs released in GitHub data directory | Claimed open, not verified | CL 2024 article | Potentially high for CSS benchmark prompts/model outputs | Exact GitHub URL not found through accessible search; needs manual PDF footnote or author follow-up |
| Ratner/Snorkel | https://github.com/snorkel-team/snorkel | Core weak-supervision library; label model, augmentation, slicing, docs/tests | Open | Pushed 2026-04-10; latest release v0.10.0 Feb 2024 | Very high for implementing label model and probabilistic weak labels | Framework, not paper replication bundle; modern API differs from 2016/2017 papers |
| Snorkel tutorials | https://github.com/snorkel-team/snorkel-tutorials | Spam/spouse/crowdsourcing tutorials, LF examples, label-model examples | Open | Pushed 2026-05-05 | High practical template for mission-text LFs | Tutorials, not research benchmark |
| WRENCH | https://github.com/JieyuZ2/wrench ; https://huggingface.co/datasets/jieyuz2/WRENCH | Benchmark framework, 22 datasets with weak labels, label/end/joint models, dataset format docs | Open | Repo pushed 2024-02-13; HF dataset available | Very high benchmark for weak labels + BERT/end-model evaluation | Some tasks are not mission-like; license/source varies by dataset |
| BoxWRENCH | https://github.com/jeffreywpli/stronger-than-you-think | Realistic weak-supervision benchmark code; datasets via Drive/HF links; LF design pipeline | Open/mixed | GitHub created 2023-12; paper 2024/2025 | High for imbalanced/high-cardinality realistic WS and RoBERTa comparison | Amazon31 not released/retracted; Drive provenance weaker than DOI/HF |
| Zhu et al. BERT-LNL | https://github.com/uds-lsv/BERT-LNL | BERT noisy-label training code, noise functions, trainers, loaders | Open code | Pushed 2022-05-31 | High for evaluating injected vs weak-label noise and co-teaching/label smoothing baselines | Data external; no prompts; modest maintenance |
| Hausa/Yorùbá weak labels | https://github.com/uds-lsv/transfer-distant-transformer-african/tree/master/data | Clean/noisy TSV splits for low-resource topic classification/NER | Open data/code | Pushed 2021-12-16 | High example of feature-dependent weak supervision noise | Multilingual news domain, not nonprofit text; check licenses |
| SaFER | ACL page https://aclanthology.org/2023.acl-industry.38/ | Paper PDF and algorithm; “Code will be released at GitHub” note | No verified official code | ACL 2023 | Medium conceptual value | No official code found; GitHub result `xzy-101/SAFER-code` appears unrelated/ambiguous and should not be treated as official |
| LAFT | https://github.com/SongW-SW/LAFT | Only `README.md` observed | Open but effectively empty | 1 commit, pushed 2023-10-19 | Low as implementation artifact; high conceptual value from paper | Paper promises code, but repo lacks usable implementation/data |
| Lu & Smith 2025 | https://arxiv.org/abs/2504.15432 | Paper only found; datasets include IMDB, Ecommerce, Manifestos, Toxic per text | No package found | arXiv 2025 | High cautionary value | No official code/data found; appears only listed in `LLM4Annotation` survey |
| Ye et al. SiDyP | https://github.com/gtfintechlab/SiDyP | `src`, `scripts`, `datasets/llm/fewshot/mixtral822`, requirements, figure | Open code + partial LLM-labeled data | Created 2025-05; pushed 2025-06-01 | High for LLM-generated noisy-label calibration and iterative refinement | New repo; small stars; TogetherAI model deprecated, but labels supplied |
| Törnberg 2024/2025 | SAGE/author pages; DOI https://doi.org/10.1177/08944393241286471 | Paper and supplemental context found | No package found | Published SSCR 2025 | Medium methodological value | No Dataverse/OSF/GitHub/OpenICPSR/Zenodo package found in accessible searches |

## Usable Datasets and Benchmarks

| Dataset/benchmark | Source repo | Paper/source | Format | Coverage | Access type | File URL |
|---|---|---|---|---|---|---|
| Gilardi annotation/ChatGPT results | Harvard Dataverse | Gilardi et al. 2023 | CSV/TAB/XLSX | Tweets/news articles; relevance, stance, topics, frames, problem/solution | Open with tweet limitations | https://dataverse.harvard.edu/api/access/datafile/7194121 |
| Gilardi training data frames | Harvard Dataverse | Gilardi et al. 2023 | CSV | Frame-classification training examples | Open | https://dataverse.harvard.edu/api/access/datafile/7194122 |
| Heseltine political tweet/coding data | Harvard Dataverse | Heseltine & Clemm von Hohenberg 2024 | CSV/TAB | US/congressional candidate tweets, GPT/human coding results | Open | https://dataverse.harvard.edu/api/access/datafile/8138346 |
| WRENCH | Hugging Face / GitHub | Zhang et al. 2021 | JSON-style benchmark datasets | 22 classification/sequence-tagging tasks with weak labels/LFs | Open | https://huggingface.co/datasets/jieyuz2/WRENCH |
| BoxWRENCH | GitHub + Drive/HF links | Stronger Than You Think | WRENCH-compatible datasets | Banking77, ChemProt, Claude9, MASSIVE; Amazon31 not released | Mixed open | https://github.com/jeffreywpli/stronger-than-you-think |
| Hausa/Yorùbá weak-label news | GitHub | Hedderich et al.; Zhu et al. use | TSV | Clean/noisy labels for low-resource African news classification | Open | https://github.com/uds-lsv/transfer-distant-transformer-african/tree/master/data |
| SiDyP LLM labels | GitHub | Ye et al. 2025 | Dataset directories | Mixtral-labeled zero/few-shot data for LLM-generated noisy-label benchmarking | Open | https://github.com/gtfintechlab/SiDyP/tree/main/datasets/llm/fewshot/mixtral822 |
| Snorkel spam/spouse tutorials | GitHub | Snorkel tutorials | Python loaders/data examples | LF examples for spam and relation extraction | Open | https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam |

## Codebases

| Codebase | Source | Main use | Language | Access type | Key files / dirs | Caveats |
|---|---|---|---|---|---|---|
| `gpt_annotate` | https://github.com/npangakis/gpt_annotate | LLM annotation workflow, repeated sampling/consistency | Python | Open | `gpt_annotate.py`, `sample_annotation_code.ipynb` | Data not bundled |
| Snorkel | https://github.com/snorkel-team/snorkel | Labeling functions, label model, weak-label aggregation | Python | Open | `snorkel/`, docs/tests | Use current API, not legacy paper code |
| Snorkel tutorials | https://github.com/snorkel-team/snorkel-tutorials | Practical LF recipes | Python notebooks/scripts | Open | `spam/`, `spouse/`, `getting_started/` | Tutorial-level |
| WRENCH | https://github.com/JieyuZ2/wrench | Benchmarking label models/end models with weak labels | Python | Open | `wrench/labelmodel`, `wrench/endmodel`, `examples/`, `datasets/` | Some dataset download needed from HF |
| BERT-LNL | https://github.com/uds-lsv/BERT-LNL | BERT under injected/weak label noise | Python | Open | `main.py`, `noise_functions.py`, `trainers/`, `models/` | External datasets |
| African transfer/distant supervision | https://github.com/uds-lsv/transfer-distant-transformer-african | Clean/noisy low-resource data pipeline | Python | Open | `code/`, `data/`, `results/` | Older code |
| SiDyP | https://github.com/gtfintechlab/SiDyP | Calibrating BERT trained on LLM labels | Python/Shell | Open | `src/`, `scripts/llm_inference.sh`, `scripts/train.sh`, `datasets/llm/...` | New; LLM provider/model deprecation caveat |
| LAFT | https://github.com/SongW-SW/LAFT | Claimed LAFT implementation | None usable | Open but empty | `README.md` only | Not currently reusable |

## Prompt and Annotation Artifacts

| Artifact | Source | Contents | Access type | Reuse value | Caveats |
|---|---|---|---|---|---|
| Gilardi ChatGPT zero-shot template | Harvard Dataverse | Python prompt/script for zero-shot annotation tasks | Open | High for prompt logging and task templates | Older ChatGPT behavior; social-media tasks |
| Pangakis `gpt_annotate` workflow | GitHub | Codebook-as-prompt workflow, repeated classifications, consistency score | Open | Very high for validation-first mission-labeling | Need provide own labeled validation data |
| Heseltine prompt/coding instructions | SAGE supplement + Dataverse | GPT-4 double-run/hybrid adjudication coding approach | Open | High for double-run disagreement triage | Need inspect `Python.rar`/supplement for exact prompt text |
| Smith et al. Language Models in the Loop prompts | ACM article appendix | WRENCH prompts as labeling functions; Yes/No prompt + label-map pattern | Article/open appendix, no separate repo found | Very high design pattern for LLM prompts as LFs | Prompt list is in paper appendix, not easily packaged |
| Ziems CSS prompts/model outputs | Paper claims GitHub data directory | Prompts, datasets, model outputs for 24/25 CSS tasks | Claimed open, not verified | High if located | Exact repo URL not found in accessible search |
| SiDyP LLM inference scripts | GitHub | LLM inference scripts for LLM-labeled data | Open | High for reproducing LLM-label generation/calibration | TogetherAI model deprecated; supplied labels mitigate |

## Unavailable or Not-Found Packages

| Paper | Status | Repositories checked | Author/site fallback | Notes |
|---|---|---|---|---|
| Törnberg (2024/2025), *LLMs Outperform Expert Coders...* | No verified package found | Dataverse, OpenICPSR, OSF, Zenodo, GitHub, SAGE/author page | UvA profile checked via web search | Paper found; no replication data/code surfaced. |
| Ziems et al. (2024), *Can LLMs Transform CSS?* | Package claimed but not located | Dataverse, OpenICPSR, OSF, Zenodo, GitHub search, ACL/MIT/SALT pages | SALT/author pages checked | PDF says “Data Directory of our Github Project”; exact URL not exposed in accessible snippets. |
| SaFER (Qi et al. 2023) | No official usable code found | Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACL | ACL page checked | Paper says code will be released. Ambiguous third-party `SAFER-code` does not match enough to trust. |
| LAFT (Wang et al. 2023) | Official repo effectively empty | Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACL/OpenReview | GitHub repo checked | Only README visible; no code/data. |
| Lu & Smith (2025), *Feeding LLM Annotations to BERT...* | No official package found | Dataverse, OpenICPSR, OSF, Zenodo, GitHub, arXiv, survey repo | arXiv and survey repo checked | No code/data link observed; datasets are standard/known but experiment artifacts unavailable. |
| Ratner et al. 2016 Data Programming paper-specific archive | No separate replication package found | Dataverse, OpenICPSR, OSF, Zenodo, GitHub | Snorkel project pages checked | Treat Snorkel library/tutorials as implementation artifacts, not exact replication package. |
| Smith et al. 2024, *Language Models in the Loop* | No standalone code/data repo found | Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACM | ACM full-text checked | Uses WRENCH and appendix prompts; no separate package found. |

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| ChatGPT annotation results | Harvard Dataverse | Gilardi et al. 2023 | TAB/CSV/XLSX | ChatGPT/MTurk/RA annotations across tweet/news tasks | Open, tweet caveat | https://dataverse.harvard.edu/api/access/datafile/7194121 |
| Frame training data | Harvard Dataverse | Gilardi et al. 2023 | CSV | Framing examples for text classification | Open | https://dataverse.harvard.edu/api/access/datafile/7194122 |
| Candidate pre-primary tweets | Harvard Dataverse | Heseltine & Clemm von Hohenberg 2024 | CSV | 2022 candidate tweets for downstream BERTweet study | Open | https://dataverse.harvard.edu/api/access/datafile/8138346 |
| GPT/human coding summary results | Harvard Dataverse | Heseltine & Clemm von Hohenberg 2024 | TAB | F1/percentage coding results | Open | https://dataverse.harvard.edu/api/access/datafile/8138340 |
| WRENCH benchmark | Hugging Face/GitHub | Zhang et al. 2021 | JSON-like benchmark files | 22 weak-supervision datasets with weak labels | Open | https://huggingface.co/datasets/jieyuz2/WRENCH |
| Hausa/Yorùbá weak-label data | GitHub | Zhu et al. 2022 / Hedderich et al. | TSV | Clean/noisy low-resource African news labels | Open | https://github.com/uds-lsv/transfer-distant-transformer-african/tree/master/data |
| SiDyP Mixtral-labeled data | GitHub | Ye et al. 2025 | Directory of LLM labels | LLM-generated noisy labels for benchmarking | Open | https://github.com/gtfintechlab/SiDyP/tree/main/datasets/llm/fewshot/mixtral822 |
| BoxWRENCH datasets | GitHub/Drive/HF | Stronger Than You Think | WRENCH-compatible | Banking77, ChemProt, Claude9, MASSIVE; not Amazon31 | Mixed open | https://github.com/jeffreywpli/stronger-than-you-think |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| Gilardi et al. 2023 | Python | `03-01-chatgpt-Zeroshot-Task-template.py` | LLM zero-shot annotation | https://doi.org/10.7910/DVN/PQYF6M | Open | https://dataverse.harvard.edu/api/access/datafile/7218458 |
| Gilardi et al. 2023 | R | `01-*`, `02-*`, `04-01-figures_annotations.R` | MTurk prep/evaluation/figures | https://doi.org/10.7910/DVN/PQYF6M | Open | https://dataverse.harvard.edu/api/access/datafile/7194106 |
| Heseltine & Clemm von Hohenberg 2024 | R | `PrimaryAnalysis_Final.R`, `CodingAnalysis.R` | GPT/human coding analysis | https://doi.org/10.7910/DVN/V2P6YL | Open | https://dataverse.harvard.edu/api/access/datafile/8138342 |
| Heseltine & Clemm von Hohenberg 2024 | Python | `Python.rar` | Downstream coding/model workflow archive | https://doi.org/10.7910/DVN/V2P6YL | Open | https://dataverse.harvard.edu/api/access/datafile/8138344 |
| Pangakis et al. | Python | `gpt_annotate.py`, `sample_annotation_code.ipynb` | LLM annotation validation workflow | https://github.com/npangakis/gpt_annotate | Open | https://github.com/npangakis/gpt_annotate/blob/main/gpt_annotate.py |
| Snorkel | Python | `snorkel/`, `docs/`, `test/` | Labeling functions + label model | https://github.com/snorkel-team/snorkel | Open | https://github.com/snorkel-team/snorkel/tree/main/snorkel |
| Snorkel tutorials | Python | `spam/`, `spouse/`, `getting_started/` | LF tutorials and examples | https://github.com/snorkel-team/snorkel-tutorials | Open | https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam |
| WRENCH | Python | `wrench/labelmodel`, `wrench/endmodel`, `examples/` | Weak-supervision benchmark, label/end models | https://github.com/JieyuZ2/wrench | Open | https://github.com/JieyuZ2/wrench/tree/main/wrench |
| BERT-LNL | Python | `main.py`, `noise_functions.py`, `trainers/` | BERT fine-tuning under label noise | https://github.com/uds-lsv/BERT-LNL | Open | https://github.com/uds-lsv/BERT-LNL/blob/main/main.py |
| SiDyP | Python/Shell | `src/`, `scripts/llm_inference.sh`, `scripts/train.sh` | Calibration/iterative refinement for LLM noisy labels | https://github.com/gtfintechlab/SiDyP | Open | https://github.com/gtfintechlab/SiDyP/tree/main/scripts |
| BoxWRENCH | Python | `val_size_experiment.py`, `pipelines.py`, `model_search_space/` | Realistic WS benchmark and RoBERTa/CFT pipelines | https://github.com/jeffreywpli/stronger-than-you-think | Open | https://github.com/jeffreywpli/stronger-than-you-think |

## Practical Recommendations for the Mission-Text Classifier

1. **Reuse first:** Snorkel + WRENCH design patterns. Implement mission-text LFs with abstentions, coverage/conflict metrics, and probabilistic labels; use WRENCH only as a benchmark/reference, not domain data.
2. **Prompt workflow:** Start from `gpt_annotate` and Gilardi/Heseltine prompt logging. Run each mission prompt at least twice or across variants; route disagreements to human review.
3. **Validation:** Copy Pangakis’ held-out validation logic: tune prompts on a small subset, then evaluate on untouched human labels. Do not trust aggregate LLM accuracy without task-specific validation.
4. **Noisy-label training baselines:** Use BERT-LNL as the baseline suite: plain BERT/RoBERTa, label smoothing, co-teaching/sample selection, and feature-dependent-noise stress tests.
5. **Calibration/LLM-label denoising:** Inspect SiDyP for LLM-generated labels and iterative calibration. It is closer to the current pipeline than LAFT because code/data are actually available.
6. **Avoid depending on unavailable code:** Treat SaFER and LAFT as conceptual papers unless you reimplement from the algorithm sections; their official reusable artifacts are missing or empty.
7. **Data caution:** Gilardi/Heseltine are excellent LLM annotation examples, but political/news/tweet domains differ from nonprofit mission text. Use their prompts/evaluation structure, not their labels, as transfer material.
8. **Artifact contribution opportunity:** A mission-text benchmark with human audit labels, LFs, LLM prompt outputs, and calibrated BERT predictions would fill a clear gap; publish via Dataverse or Hugging Face with scripts and prompts.

## File Manifest Notes for Verified Open Packages

- **Gilardi Dataverse (`DVN/PQYF6M`)**: 62 files; key types include PDF docs, R scripts, Python prompt template, CSV/TAB/XLSX annotations and batch ChatGPT outputs, plots.
- **Heseltine Dataverse (`DVN/V2P6YL`)**: 8 files: `cands2022trimmed.csv`, `CodingAnalysis.R`, `data_info_GPT_ResPol.txt`, `PrimaryAnalysis_Final.R`, `Python.rar`, `ResPolPrePrimaryTweetsFile.csv`, `Results_F1_Combined.tab`, `Results_Percentages_Combined.tab`.
- **`npangakis/gpt_annotate`**: top level includes `.gitignore`, `README.md`, `gpt_annotate.py`, `llm_annotate_paper.pdf`, `sample_annotation_code.ipynb`.
- **`JieyuZ2/wrench`**: top level includes `datasets/`, `examples/`, `wrench/`, `README.md`, `environment.yml`, `setup.py`, license.
- **`uds-lsv/BERT-LNL`**: top level includes `main.py`, `loading_utils.py`, `noise_functions.py`, `text_dataset.py`, `models/`, `trainers/`, `utils.py`.
- **`gtfintechlab/SiDyP`**: top level includes `datasets/llm/fewshot/mixtral822`, `figure/`, `scripts/`, `src/`, `requirements.txt`, `README.md`.

## Papers With No Package Found

- Törnberg (2024/2025), *Large Language Models Outperform Expert Coders and Supervised Classifiers at Annotating Political Social Media Messages* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, SAGE, author profile. Author website: checked via UvA profile. No package found.
- SaFER (Qi et al. 2023) — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACL. Author website: not individually resolved beyond ACL/web search. No official code found.
- LAFT (Wang et al. 2023) — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACL/OpenReview. Author website: GitHub found but empty. Package effectively unavailable.
- Lu & Smith (2025), *Feeding LLM Annotations to BERT Classifiers at Your Own Risk* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, arXiv, LLM4Annotation survey. Author website: not individually resolved beyond web search. No package found.
- Smith et al. (2024), *Language Models in the Loop* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, ACM. Author website: not individually resolved beyond ACM/web search. No standalone package found; WRENCH + appendix prompts are the usable artifacts.
