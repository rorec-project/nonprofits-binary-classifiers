---
created: 20260605
agent: replication-seeker
scratchpad: .agents/notebooks/20260605-replication-short-text-model-alternatives-scratchpad.md
status: complete
title: Replication - Short Text Model Alternatives
topic: short-text classification; BERT alternatives; fine-tuning recipes; computational social science; mission statements
---

# Replication: Short-Text Model Alternatives and Fine-Tuning Recipes

## Search Strategy and Exact Queries Used

Context used: `.agents/docs/20260605-literature-synthesis-map.md` and `.agents/docs/20260605-literature-short-text-classification.md`. The latter provided the bounded citation list for BERT/RoBERTa/DistilBERT/DeBERTa/ELECTRA/ALBERT/SBERT, Sun, Gururangan, Gweon & Schonlau, Laurer et al., Ma, Wang & Qu, Zhang et al., and SLM industrial classification.

Repository/API queries run:

- Dataverse API: exact title/author fragments including `BERT Pre-training of Deep Bidirectional Transformers`, `RoBERTa Robustly Optimized BERT Pretraining Approach`, `Less Annotating More Classifying BERT-NLI Laurer`, `Selecting Between BERT and GPT for Text Classification Political Science`, `Automated Coding Using Machine Learning Remapping U.S. Nonprofit Sector Ma`, `npo_classifier nonprofit machine learning`.
- Dataverse file APIs: `doi:10.7910/DVN/8ACDTT`, `doi:10.7910/DVN/3TZAEB`.
- Zenodo API: same title/author fragments; inspected records `10839412`, `14475970`, `4707657`.
- OSF API: same title/author fragments using `filter[title]` and `filter[public]=true`.
- OpenICPSR web search: `site:openicpsr.org "BERT" "text classification" replication political science`; `site:openicpsr.org "RoBERTa" OR "DeBERTa" "replication" "text"`; `site:openicpsr.org "nonprofit" "machine learning" "BERT" replication`.
- GitHub/code search: `dont-stop-pretraining`; `less-annotating-with-bert-nli`; `npo_classifier`; `How to Fine-Tune BERT for Text Classification`.
- Web/GitHub fallback queries: `2026 GitHub BERT RoBERTa DeBERTa text classification replication package Dataverse "BERT, RoBERTa, or DeBERTA" Timoneda`; exact titles for Gweon & Schonlau, Wang & Qu, Zhang et al., Small Language Models; Ma/npo_classifier.
- Hugging Face queries: `site:huggingface.co MoritzLaurer DeBERTa-v3-base-mnli-fever-anli`; `site:huggingface.co "pol_DEBATE" "ModernBERT" "DeBERTa"`; foundation model card query for `bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`, `google/electra-base-discriminator`, `albert-base-v2`, `all-MiniLM-L6-v2`.

Caveat: GitHub Contents API was rate-limited (403). GitHub landing pages and repository README pages were used as fallback, as allowed by the workflow.

## Verified Packages

| Paper | Repository | URL | Code Languages | Data Included | Access Type | File Count |
|-------|-----------|-----|----------------|---------------|-------------|------------|
| Devlin et al. (2019), BERT | GitHub + Hugging Face | https://github.com/google-research/bert ; https://huggingface.co/google-bert/bert-base-uncased | Python/TensorFlow | Pretrained checkpoints; no task-specific short-text datasets | Open | GitHub repo; HF model files |
| Liu et al. (2019), RoBERTa | GitHub/fairseq + Hugging Face | https://github.com/facebookresearch/fairseq/tree/main/examples/roberta ; https://huggingface.co/FacebookAI/roberta-base | Python/PyTorch | Pretrained checkpoints; examples for GLUE/custom classification | Open | fairseq example files; HF model files |
| Sanh et al. (2019), DistilBERT | Hugging Face/Transformers | https://huggingface.co/distilbert/distilbert-base-uncased ; https://github.com/huggingface/transformers | Python/PyTorch/TF | Model card/checkpoint; no paper replication bundle found in Dataverse/OSF/OpenICPSR/Zenodo | Open | HF model files |
| He et al. (2021), DeBERTa | GitHub + Hugging Face | https://github.com/microsoft/DeBERTa ; https://huggingface.co/microsoft/deberta-v3-base | Python/PyTorch | Code/checkpoints | Open | GitHub/HF files; API listing unavailable due rate limit |
| Clark et al. (2020), ELECTRA | GitHub + Hugging Face | https://github.com/google-research/electra ; https://huggingface.co/google/electra-base-discriminator | Python/TensorFlow | Pretrained small/base/large checkpoints; OpenWebText training recipe substitute | Open | GitHub/HF files |
| Lan et al. (2020), ALBERT | GitHub + Hugging Face | https://github.com/google-research/albert ; https://huggingface.co/albert/albert-base-v2 | Python/TensorFlow | Pretrained checkpoints | Open | GitHub/HF files |
| Reimers & Gurevych (2019), SBERT | GitHub + Hugging Face | https://github.com/huggingface/sentence-transformers ; https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Python/PyTorch | Embedding models; all-MiniLM maps text to 384-d vectors | Open | GitHub/HF files |
| Sun et al. (2019), How to Fine-Tune BERT | GitHub | https://github.com/xuyige/BERT4doc-Classification | Python; TensorFlow 1.x; PyTorch | AG News/Sogou/IMDB-style scripts and links; checkpoints via Google Drive | Open, partly external-drive | Top-level: `codes`, `data`, README, LICENSE |
| Gururangan et al. (2020), Don't Stop Pretraining | GitHub + Hugging Face + S3 | https://github.com/allenai/dont-stop-pretraining ; https://huggingface.co/allenai | Python, Jsonnet, Shell | S3 task data; DAPT/TAPT HF models | Open | Top-level: 6 dirs + docs; 61 commits |
| Laurer et al. (2024/2023), BERT-NLI | Dataverse + GitHub + Zenodo + Code Ocean + Hugging Face | https://doi.org/10.7910/DVN/8ACDTT ; https://github.com/MoritzLaurer/less-annotating-with-bert-nli ; https://doi.org/10.5281/zenodo.10839412 ; https://doi.org/10.24433/CO.5414009.v2 | Python, notebooks, CSV/XLSX | Cleaned political text datasets, raw data where redistributable, hyperparameter tables, figures, code | Open | Dataverse: 1,623 files; Zenodo: 1 zip (279.6 MB) |
| Timoneda & Vallejo Vera (2025), BERT/RoBERTa/DeBERTa political text | Dataverse; article says GitHub but exact URL was 404 | https://doi.org/10.7910/DVN/3TZAEB ; article DOI https://doi.org/10.1086/730737 | Python; Excel; text logs | Political text datasets: civility, fake news, speeches; model result logs | Open | Dataverse: 230 files |
| Political DEBATE (Burnham et al. 2025/2026) | Dataverse + GitHub + Hugging Face | https://doi.org/10.7910/DVN/SV5VHH ; https://github.com/MLBurnham/pol_DEBATE ; https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0 ; https://huggingface.co/mlburnham/Political_DEBATE_base_v1.0 | Python; HF Transformers | PolNLI data/model assets; boilerplate zero/few-shot code | Open | HF large model includes 1.75 GB model files; Dataverse file count not inspected |
| Ma (2021), nonprofit ML classifier | GitHub + OSF preprint | https://github.com/ma-ji/npo_classifier ; https://doi.org/10.1177/0899764020968153 ; https://osf.io/pt3q9/ | Mostly Jupyter notebooks; Python package under `API` | Universal Classification Files, NTEE classifier outputs, remapped nonprofit sector links | Open | Top-level: `API`, `dataset`, `output`, `reference`, `script`; 338 commits |
| Zhang et al. (2025), BERT-like vs LLMs / TaMAS | ACL + GitHub | https://aclanthology.org/2025.findings-emnlp.1033/ ; https://github.com/jyzhang2002/TaMAS-TextClass | Python | Pre-split ToxiCloakCN datasets; links to LegalText/MaliciousCode/True-False | Open | Top-level: `dataset`, `pipeline`, `visualization`, PDF/PNG, requirements |
| Li et al. (2025), Small Language Models in Real World | ACL + GitHub | https://aclanthology.org/2025.acl-industry.68/ ; https://github.com/DobricLilujun/agentCLS | Jupyter notebooks; Python; shell | Data examples only; proprietary industrial email not fully open | Partly open / proprietary data caveat | Top-level: `.vscode`, `data_examples`, `script`, `utils`; 38 commits |
| ModernBERT and ModernBERT zero-shot classifier assets | Hugging Face + GitHub | https://huggingface.co/answerdotai/ModernBERT-base ; https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | Python/HF | Model cards/checkpoints | Open | HF model files |

## Repository Search Log

| Repository | Query | Results Found | Notes |
|-----------|-------|--------------|-------|
| Dataverse | Exact title/author fragments for priority papers | 2 high-confidence core hits: Laurer `DVN/8ACDTT`; Timoneda/Vallejo Vera `DVN/3TZAEB`; plus political DEBATE `DVN/SV5VHH` from Cambridge page | Many foundation-model title searches returned irrelevant generic BERT/author-name hits; file manifests inspected for `8ACDTT` and `3TZAEB`. |
| OpenICPSR | `site:openicpsr.org "BERT" "text classification" replication political science`; `RoBERTa/DeBERTa`; `nonprofit machine learning BERT` | No direct priority-paper package found | Results were adjacent social-media/economics packages, not model-comparison/fine-tuning artifacts. |
| OSF | API `filter[title]` exact paper/title fragments | No direct OSF node hits | Ma README links OSF preprint `osf.io/pt3q9`, but code/data are on GitHub. |
| Zenodo | API exact title/author fragments | Laurer `10.5281/zenodo.10839412`; single BERT notebook `10.5281/zenodo.14475970`; survey auto-code PDF `10.5281/zenodo.4707657` | Zenodo was mostly noisy for foundation model titles. |
| GitHub | Code search and webfetch for exact repo/paper strings | Verified official/fallback repos for BERT, RoBERTa, ELECTRA, ALBERT, SBERT, Sun, Gururangan, Laurer, Ma, Zhang, Li/SLM | GitHub API rate-limited; landing-page manifests used. |
| Hugging Face | Model-card searches for foundation, Laurer, Political DEBATE, ModernBERT | Verified foundation model cards and reusable NLI/zero-shot classifier assets | Stable for model checkpoints; not a full paper replication archive. |
| ACL/PapersWithCode/author pages | Exact title web searches | ACL pages exposed Zhang/TaMAS and SLM/agentCLS links; arXiv for Wang & Qu says Dataverse pending acceptance | PapersWithCode blocks on arXiv pages did not show extra verified package links in search context. |

## Tables of Code Repositories, Datasets, Hugging Face Assets, and Training Scripts

### Code Repositories

| Artifact | Stable URL | Contents | Access / update signal | Caveats |
|---|---|---|---|---|
| google-research/bert | https://github.com/google-research/bert | TensorFlow BERT architecture; pretrained checkpoint links; fine-tuning for SQuAD/MultiNLI/MRPC | Open; canonical foundation repo | Not a short-text binary-classification comparison package. |
| fairseq RoBERTa | https://github.com/facebookresearch/fairseq/tree/main/examples/roberta | RoBERTa pretraining/fine-tuning examples; model downloads | Open | Uses fairseq; not a paper-specific CSS package. |
| google-research/electra | https://github.com/google-research/electra | Pretraining and fine-tuning code; small/base/large model recipes | Open | Original pretraining data unavailable; OpenWebText recipe provided. |
| google-research/albert | https://github.com/google-research/albert | ALBERT code and TF-Hub/tar checkpoint links | Open | Foundation repo only. |
| sentence-transformers | https://github.com/huggingface/sentence-transformers | Embedding/reranker framework; training/evaluation support | Open | Use for frozen embedding baseline rather than direct classifier. |
| BERT4doc-Classification | https://github.com/xuyige/BERT4doc-Classification | Further pretraining and fine-tuning scripts; layer pooling/truncation recipes | Open; 8 commits; Apache-2.0 | Old dependencies: TF 1.x, torch <=1.2. |
| dont-stop-pretraining | https://github.com/allenai/dont-stop-pretraining | DAPT/TAPT scripts, Jsonnet configs, search space, dataset download scripts | Open; 61 commits | Pinned old AllenNLP stack; use concept/configs, not direct drop-in. |
| less-annotating-with-bert-nli | https://github.com/MoritzLaurer/less-annotating-with-bert-nli | Full replication code/data; Jupyter notebook for training BERT-NLI | Open; also Code Ocean/Dataverse/Zenodo | Some raw datasets externally sourced/licensed. |
| npo_classifier | https://github.com/ma-ji/npo_classifier | Nonprofit NTEE classifier API, datasets, scripts, outputs | Open; 338 commits | Mostly notebooks; multi-class/multi-label NTEE, not binary religion. |
| TaMAS-TextClass | https://github.com/jyzhang2002/TaMAS-TextClass | BERT/ELECTRA/ERNIE fine-tune, LLM internal-state, zero-shot pipelines | Open; 4 commits | Some datasets must be downloaded separately. |
| agentCLS | https://github.com/DobricLilujun/agentCLS | SFT/prompt/SPT/PT scripts and notebooks for SLM classification | Open code; data examples only | Industrial email dataset proprietary/partly unavailable. |

### Datasets and Data Packages

| Dataset / Package | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| Laurer political text benchmark datasets | Dataverse `10.7910/DVN/8ACDTT` | Laurer et al. | CSV/TAB/XLSX/ZIP | Manifesto, sentiment economy, SOTU, Supreme Court, CoronaNet, stance subsets | Open | https://dataverse.harvard.edu/api/access/datafile/6810596 (`5AC.tab`, example) |
| Timoneda/Vallejo Vera political transformer datasets | Dataverse `10.7910/DVN/3TZAEB` | BERT/RoBERTa/DeBERTa political text | XLSX/TXT/Python | Civility, fake-news COVID, speeches | Open | https://dataverse.harvard.edu/api/access/datafile/7678506 (`civility_data.xlsx`) |
| Ma Universal Classification Files / nonprofit datasets | GitHub `ma-ji/npo_classifier` | Ma (2021) | CSV/notebook outputs | NTEE benchmark and remapped nonprofit sector | Open | https://github.com/ma-ji/npo_classifier/tree/master/dataset |
| ToxiCloakCN pre-split data | GitHub `TaMAS-TextClass` | Zhang et al. | Repo dataset files | Chinese implicit hate speech variants | Open | https://github.com/jyzhang2002/TaMAS-TextClass/tree/master/dataset |
| LegalText | Hugging Face | Zhang et al. | HF dataset | Legal text classification | Open | https://huggingface.co/datasets/openSUSE/cavil-legal-text |
| MaliciousCode | Hugging Face | Zhang et al. | HF dataset | Malicious code classification | Open | https://huggingface.co/datasets/Er1111c/Malicious_code_classification |
| True-False / hallucination dataset | Author-hosted zip | Zhang et al. | ZIP | Hallucination / truthfulness classification | Open, author-hosted | http://azariaa.com/Content/Datasets/true-false-dataset.zip |
| Patient Joe open-ended survey | GESIS | Gweon & Schonlau | Survey dataset | 585 short open-ended answers | Open with archive terms | https://doi.org/10.7802/2474 |
| Disclosure data | Schonlau & Couper replication materials | Gweon & Schonlau | Survey replication data | Open-ended participation-risk probe | Archive terms vary | Mentioned in paper; exact package not verified in this run |
| Industrial email dataset | agentCLS paper/repo | Li et al. | Not fully public | Real-world proprietary email classification | Restricted/proprietary | None |

### Hugging Face Assets

| Asset | URL | Contents / use | Caveats |
|---|---|---|---|
| `google-bert/bert-base-uncased` | https://huggingface.co/google-bert/bert-base-uncased | BERT base model card/checkpoint | HF-written card; original Google repo canonical. |
| `FacebookAI/roberta-base` | https://huggingface.co/FacebookAI/roberta-base | RoBERTa base checkpoint | Use for encoder fine-tuning baseline. |
| `distilbert/distilbert-base-uncased` | https://huggingface.co/distilbert/distilbert-base-uncased | DistilBERT checkpoint; 40% fewer params, ~60% faster claim | Distillation code in transformers ecosystem. |
| `microsoft/deberta-v3-base` | https://huggingface.co/microsoft/deberta-v3-base | DeBERTa-v3 base checkpoint | Strong candidate for final model grid. |
| `google/electra-base-discriminator` | https://huggingface.co/google/electra-base-discriminator | ELECTRA discriminator checkpoint | Use discriminator for classification. |
| `albert/albert-base-v2` | https://huggingface.co/albert/albert-base-v2 | ALBERT v2 checkpoint | Parameter-efficient but not always speed-efficient. |
| `sentence-transformers/all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 384-dimensional sentence embeddings | Good frozen embedding + LR baseline. |
| Laurer DeBERTa NLI models | https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli ; https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | Zero-shot/NLI classifiers; 763,913 to 885,242 NLI pairs | Useful for hypothesis-style mission classification. |
| Laurer universal zero-shot classifier | https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 | NLI + 28 classification tasks reformatted as NLI | Training mix includes some social/CSS datasets. |
| Political DEBATE | https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0 ; https://huggingface.co/mlburnham/Political_DEBATE_base_v1.0 | Political-domain DeBERTa NLI classifier; PolNLI | Political text, not nonprofit-specific. |
| ModernBERT zero-shot | https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | Fast/memory-efficient zero-shot encoder classifier | Card says slightly worse than DeBERTa-v3 on tested tasks. |

### Training Scripts / Config Files of Interest

| Paper / Repo | Key files | Reusable recipe values |
|---|---|---|
| Sun et al. / BERT4doc | `run_classifier_single_layer.py`, `run_classifier_discriminative.py`, `create_pretraining_data.py`, `run_pretraining.py` | Further pretraining: max_seq_length 128, max_predictions 20, MLM prob 0.15, seed 12345, batch 32, steps 100k, warmup 10k, LR 5e-5. Fine-tune: max_len 512, batch 24, LR 2e-5, epochs 3/4/6, seed 42; last-two-layer concat; head/tail truncation variants. |
| Gururangan / dont-stop-pretraining | `training_config/classifier.jsonnet`, `search_space/classifier.jsonnet`, `scripts/train.py`, `mlm_study` | Train command uses `ROBERTA_CLASSIFIER_SMALL`, perf `+f1`, hyperparameter search with 100 samples via allentune. DAPT/TAPT model IDs distinguish domain and task data sizes. |
| Laurer / Dataverse | `analysis-transf-hyperparams.py`, `analysis-transf-run.py`, `36-table-hyperparams-deberta-base.csv`, `37-table-hyperparams-deberta-nli.csv`, `38/39-table-hyperparams-*` | Explicit hyperparameter tables for DeBERTa-base, DeBERTa-NLI, SVM-TFIDF, logistic-TFIDF; evaluate F1 macro/micro/balanced accuracy. |
| Timoneda/Vallejo Vera / Dataverse | `BiLSTM_*.py`, many `*_bert_*`, `*_roberta_*`, `*_deberta_*` result logs | Article recommends LR grids: BERT-large {3e-4,1e-4,5e-5,3e-5} with 3e-5 chosen; RoBERTa-large {1e-5,2e-5,3e-5} with 3e-5 chosen; DeBERTa-v3-large {5e-6,8e-6,9e-6,1e-5} with 1e-5 chosen; batch 16 preferred over 8; 32 faster but more GPU RAM. |
| Zhang / TaMAS | `pipeline.bert_finetune`, `pipeline.electra_finetune`, `pipeline.ernie_finetune`, `pipeline.run_all_saplma`, `pipeline.run_all_mm`, `pipeline.llm_ask` | Paper uses 7/1.5/1.5 split, LR 2e-5, 10 epochs, dropout 0.5; data availability ablations at 50%, 10%, 1%. |
| Li / agentCLS | `script/FT_bert`, `script/FT_llama`, `script/SPT`, `utils/prompts.py`, `script/FT_llama/llama3_FT_with_header.py` | `train_seed=3407`, `max_grad_norm=0.3`, `max_length=4096`; methods: SFT, soft prompt tuning, prefix tuning, base/few-shot/CoT/self-consistency/chain-of-draft prompts; classification head hidden_dim 256 with 2–5 layers tested. |
| Ma / npo_classifier | `API`, `script/classification_algorithms`, `dataset/UCF` | NTEE multi-class/multi-label benchmark; useful data acquisition and classifier API patterns, not directly a modern transformer grid. |

## Extracted Reusable Fine-Tuning Grids / Config Values

| Source | Models | LR | Epochs / steps | Batch | Seeds | Loss / metrics / other |
|---|---|---:|---:|---:|---:|---|
| Sun et al. | BERT-base/large | 2e-5 fine-tune; 5e-5 further pretrain | 3, 4, 6 fine-tune; 100k pretrain steps | 24 fine-tune; 32 pretrain | 42; pretrain 12345 | Layer pooling, layer-wise LR decay, truncation head/tail/head+tail, max_len 512. |
| Gururangan et al. | RoBERTa DAPT/TAPT | Search-space driven | Search-space driven | Search-space driven | Not fully extracted | Compare base vs DAPT vs TAPT; report F1; 100 hyperparameter samples. |
| Laurer et al. | DeBERTa-base, DeBERTa-NLI, TF-IDF SVM/LR | In Dataverse hyperparameter CSVs | Per CSV | Per CSV | Multiple data-size runs | F1 macro primary for imbalance; compare 100–2,500 labels. |
| Timoneda & Vallejo Vera | BERT-large, RoBERTa-large, DeBERTa-v3-large, XLM-R, mBERT/mDeBERTa | BERT {3e-4,1e-4,5e-5,3e-5}; RoBERTa {1e-5,2e-5,3e-5}; DeBERTa {5e-6,8e-6,9e-6,1e-5} | Not fully extracted from logs | 16 preferred; 32 faster/more memory | Many logs imply repeated runs | Weighted Adam; compare SVM/BiLSTM/Transformers; further pretraining boosts specialized text. |
| Gweon & Schonlau | BERT vs SVM/RF/XGB | Classical random search values in appendix; BERT code in online appendix | Training sizes 100/200/400 | Not extracted | Same random train/validation/test sets | Fine-tuning essential; BERT advantage grows after 200–400 labels; semi-automatic thresholding recommended. |
| Zhang et al. | BERT/RoBERTa/ERNIE/ELECTRA vs LLM states/zero-shot | 2e-5 | 10 epochs | Not extracted | Not extracted | Dropout 0.5; split 7/1.5/1.5; AUC/Accuracy/F1; data budgets 50%,10%,1%. |
| Li et al. | ModernBERT, Llama/Gemma-like SLMs | Not fully extracted | Not fully extracted | Not extracted | 3407 | max_grad_norm 0.3; max_length 4096; SFT/SPT/PT; prompts; classification head depth test. |

## Papers With No Package Found

- Wang & Qu (2024), *Selecting Between BERT and GPT for Text Classification in Political Science Research* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, arXiv/PapersWithCode, web. Author website not deeply inspected. The arXiv paper states: “All our data and code will be made publicly available and posted on Harvard Dataverse upon the paper’s acceptance.” No package found as of this search.
- Gweon & Schonlau (2024), *Automated Classification for Open-Ended Questions with BERT* — Searched: Dataverse, OpenICPSR, OSF, Zenodo, GitHub, arXiv/JSSAM web. The paper says Python code is in the online appendix and data are external (Patient Joe GESIS; Disclosure replication materials). No standalone GitHub/Dataverse package found.
- Devlin/RoBERTa/DistilBERT/DeBERTa/ELECTRA/ALBERT foundation papers — Searched all target repositories. Found canonical code/model repositories, but no Dataverse/OSF/OpenICPSR/Zenodo replication packages beyond official GitHub/HF assets.
- Fyall et al. (2018) and Haq et al. (2023) were in the literature handoff but not priority targets here; no package search beyond contextual source tracing in this run.

## Handoff: Datasets Found in Packages

| Dataset Name | Source Repo | Paper | Format | Coverage | Access Type | File URL |
|---|---|---|---|---|---|---|
| Political text benchmark bundle | Dataverse `doi:10.7910/DVN/8ACDTT` | Laurer et al. | CSV/TAB/XLSX/ZIP | Manifesto, sentiment-economy, SOTU, Supreme Court, CoronaNet | Open | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8ACDTT |
| Civility data | Dataverse `doi:10.7910/DVN/3TZAEB` | Timoneda & Vallejo Vera | XLSX | Civility text classification | Open | https://dataverse.harvard.edu/api/access/datafile/7678506 |
| Fake news COVID | Dataverse `doi:10.7910/DVN/3TZAEB` | Timoneda & Vallejo Vera | XLSX | Fake-news classification | Open | https://dataverse.harvard.edu/api/access/datafile/7678636 |
| Universal Classification Files | GitHub `ma-ji/npo_classifier` | Ma (2021) | CSV/notebooks | Nonprofit NTEE classification benchmark | Open | https://github.com/ma-ji/npo_classifier/tree/master/dataset/UCF |
| Remapped U.S. nonprofit sector | GitHub/link from Ma README | Ma (2021) | Web/data download | Multi-labeled nonprofits | Open/author-hosted | https://jima.me/?ntee_remap |
| ToxiCloakCN pre-split variants | GitHub `TaMAS-TextClass` | Zhang et al. | Repo files | Chinese implicit hate speech variants | Open | https://github.com/jyzhang2002/TaMAS-TextClass/tree/master/dataset |
| LegalText | Hugging Face | Zhang et al. | HF dataset | Legal text classification | Open | https://huggingface.co/datasets/openSUSE/cavil-legal-text |
| MaliciousCode | Hugging Face | Zhang et al. | HF dataset | Code/text malicious classification | Open | https://huggingface.co/datasets/Er1111c/Malicious_code_classification |
| PolNLI | Dataverse/Hugging Face | Political DEBATE | HF/Dataverse | Political documents across social media/news/bills/courts | Open | https://doi.org/10.7910/DVN/SV5VHH |
| Industrial email data | agentCLS | Li et al. | Not public | Industry email history | Restricted |  |

## Handoff: Code Files Found in Packages

| Paper | Language | Key Files | Method | Repository URL | Access Type | File URL |
|---|---|---|---|---|---|---|
| Sun et al. | Python | `run_classifier_single_layer.py`, `run_classifier_discriminative.py`, `create_pretraining_data.py` | BERT further pretraining/fine-tuning; layer pooling; truncation | https://github.com/xuyige/BERT4doc-Classification | Open | https://github.com/xuyige/BERT4doc-Classification/tree/master/codes |
| Gururangan et al. | Python/Jsonnet | `scripts/train.py`, `training_config/classifier.jsonnet`, `search_space/classifier.jsonnet` | DAPT/TAPT and classifier hyperparameter search | https://github.com/allenai/dont-stop-pretraining | Open | https://github.com/allenai/dont-stop-pretraining/tree/master/training_config |
| Laurer et al. | Python | `analysis-transf-hyperparams.py`, `analysis-transf-run.py`, hyperparameter CSVs | DeBERTa/BERT-NLI vs classical benchmark | https://doi.org/10.7910/DVN/8ACDTT | Open | https://dataverse.harvard.edu/api/access/datafile/6810882 |
| Timoneda & Vallejo Vera | Python | `BiLSTM_civility.py`, result logs, BERT/RoBERTa/DeBERTa scripts/logs | Political text transformer comparison | https://doi.org/10.7910/DVN/3TZAEB | Open | https://dataverse.harvard.edu/api/access/datafile/7678545 |
| Ma (2021) | Jupyter/Python | `API`, `script/classification_algorithms`, `script/data_acquisition` | NTEE classifier and nonprofit text remapping | https://github.com/ma-ji/npo_classifier | Open | https://github.com/ma-ji/npo_classifier/tree/master/script |
| Zhang et al. | Python | `pipeline.bert_finetune`, `pipeline.electra_finetune`, `pipeline.llm_ask` | BERT-like vs LLM states/zero-shot | https://github.com/jyzhang2002/TaMAS-TextClass | Open | https://github.com/jyzhang2002/TaMAS-TextClass/tree/master/pipeline |
| Li et al. | Jupyter/Python | `script/FT_bert`, `script/FT_llama`, `script/SPT`, `utils/prompts.py` | SFT, SPT/PT, prompt engineering for SLM classification | https://github.com/DobricLilujun/agentCLS | Open code / restricted data | https://github.com/DobricLilujun/agentCLS/tree/double_blind_paper_review/script |
| Political DEBATE | Python | boilerplate zero/few-shot code | Political-domain DeBERTa NLI classifiers | https://github.com/MLBurnham/pol_DEBATE | Open | https://github.com/MLBurnham/pol_DEBATE |

## Recommended Reproducible Experiment Scaffold for This Project

1. **Freeze data splits first.** Use project-level train/validation/test CSVs with a manifest recording `DATA_OF_CHOICE`, label source, random seed, and class prevalence.
2. **Baselines:** TF-IDF word+char n-gram logistic regression/SVM with `class_weight=balanced`; all-MiniLM-L6-v2 embeddings + logistic regression.
3. **Encoder grid:** `distilbert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`, `google/electra-base-discriminator`, optionally `albert-base-v2` and `answerdotai/ModernBERT-base`.
4. **Fine-tuning grid grounded in artifacts:** LR `{1e-5, 2e-5, 3e-5, 5e-5}`; DeBERTa extra `{5e-6, 8e-6}`; epochs `{3,5,8,10}` with early stopping; batch `{16,32}`; max length `{128,256}` for mission text; 5 seeds; weighted and unweighted cross-entropy.
5. **DAPT/TAPT secondary arm:** adapt Gururangan/Sun recipes only after baseline fine-tuning is stable; run short MLM/TAPT on unlabeled mission/activity descriptions and compare against no-TAPT.
6. **NLI/zero-shot arm:** test MoritzLaurer DeBERTa NLI and Political DEBATE-style hypothesis prompts for few-label or audit-label regimes; do not replace supervised models unless validated.
7. **Small generative LM arm:** follow agentCLS cautiously: zero/few-shot prompting plus, if compute permits, SFT/LoRA/classification head for one small Llama/Gemma/Phi/Qwen model; record unusable-output rate.
8. **Metrics:** PR-AUC/average precision, ROC-AUC, F1, precision/recall, MCC, Brier score, ECE, calibration curves, threshold tables, and prevalence error. Report mean±sd over seeds.
9. **Artifact layout:** `experiments/configs/*.yaml`, `experiments/runs/<timestamp>/metrics.json`, `predictions.csv`, `model_card.md`, and `splits_manifest.json`. Include exact HF model revisions and package versions.

## Caveats

- GitHub API file inspection was rate-limited; webfetch and search-result manifests were used for GitHub file structures.
- Several foundation-model artifacts are not replication packages in social-science sense; they are canonical code/checkpoint repositories.
- Author-hosted and Google Drive assets (Sun checkpoints; some survey appendices) have weaker archival stability than Dataverse/Zenodo/OSF.
- Proprietary industrial datasets in the SLM paper are not reproducible without access; code still provides useful scaffolding.

## Handoff Quality Gate

- Verified Packages: concrete entries present.
- Repository Search Log: concrete per-repository entries present.
- Papers With No Package Found: explicit missing states present.
- Handoff: Datasets Found in Packages: concrete entries plus restricted marker present.
- Handoff: Code Files Found in Packages: concrete entries present.
