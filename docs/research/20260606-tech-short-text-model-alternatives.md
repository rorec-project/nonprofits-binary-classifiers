---
created: 20260606
agent: tech-seeker
scratchpad: docs/research/notebooks/20260606-tech-short-text-model-alternatives-scratchpad.md
status: complete
title: Tech - Short Text Model Alternatives
topic: short-text; model-alternatives; BERT; LLM
---

# Tech: Short-Text Binary Classification Model Alternatives

## From Replication Packages

Replication-informed mode from `docs/research/20260605-replication-short-text-model-alternatives.md`.

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| Sun et al. | Python | `run_classifier_single_layer.py`, `run_classifier_discriminative.py`, `create_pretraining_data.py` | BERT further pretraining/fine-tuning; layer pooling; truncation | https://github.com/xuyige/BERT4doc-Classification |
| Gururangan et al. | Python/Jsonnet | `scripts/train.py`, `training_config/classifier.jsonnet`, `search_space/classifier.jsonnet` | DAPT/TAPT and classifier hyperparameter search | https://github.com/allenai/dont-stop-pretraining |
| Laurer et al. | Python | `analysis-transf-hyperparams.py`, `analysis-transf-run.py` | DeBERTa/BERT-NLI vs classical benchmark | https://doi.org/10.7910/DVN/8ACDTT |
| Timoneda & Vallejo Vera | Python | BERT/RoBERTa/DeBERTa scripts/logs | Political text transformer comparison | https://doi.org/10.7910/DVN/3TZAEB |
| Ma | Python/Jupyter | `script/classification_algorithms`, `script/data_acquisition` | NTEE/nonprofit classifier | https://github.com/ma-ji/npo_classifier |
| Zhang et al. | Python | `pipeline.bert_finetune`, `pipeline.electra_finetune`, `pipeline.llm_ask` | BERT-like vs LLM states/zero-shot | https://github.com/jyzhang2002/TaMAS-TextClass |
| Li et al. | Python/Jupyter | `script/FT_bert`, `script/FT_llama`, `script/SPT` | SFT/SPT/prompting for small-LM classification | https://github.com/DobricLilujun/agentCLS |

## Package Recommendations

| Language | Package | Version | Key Function | Source |
|----------|---------|---------|-------------|--------|
| Python | `transformers` | Current docs: v4.57.x/v5 docs available | `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `EarlyStoppingCallback` | https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md |
| Python | `sentence-transformers` | Current docs | `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(...)` | https://github.com/huggingface/sentence-transformers |
| Python | `peft` | Current docs | `LoraConfig(task_type=TaskType.SEQ_CLS)`, `get_peft_model` | https://github.com/huggingface/peft |
| Python | `scikit-learn` | Stable docs indexed as 1.8/1.9 | `LogisticRegression(class_weight="balanced")`, `CalibratedClassifierCV`, `calibration_curve` | https://scikit-learn.org/stable/modules/calibration.html |
| Python | `datasets`/`evaluate` | Current Hugging Face ecosystem | Dataset splits, metrics, reproducible preprocessing | https://huggingface.co/docs/datasets |

## Implementation Examples

### Example 1: Encoder fine-tuning grid

**Source**: Hugging Face Transformers sequence-classification docs; Sun et al. (2019); Gururangan et al. (2020); Laurer et al. (2024).  
**Package version**: `transformers` v4.57.x/v5-era syntax.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

model_name = "microsoft/deberta-v3-base"  # also test roberta-base, distilbert-base-uncased, google/electra-base-discriminator
id2label = {0: "NON_RELIGIOUS", 1: "RELIGIOUS"}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

args = TrainingArguments(
    output_dir="experiments/runs/deberta-v3-base",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_pr_auc",
    greater_is_better=True,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
```

**Notes**: Use 5 seeds; compare weighted and unweighted cross-entropy; max length 128/256 for mission statements. Treat TAPT/DAPT as a second-stage arm after stable baselines.

### Example 2: MiniLM/SBERT embeddings baseline

**Source**: Sentence-Transformers docs; Reimers & Gurevych (2019).  
**Package version**: current `sentence-transformers` docs.

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_train = embedder.encode(train_texts, batch_size=128, show_progress_bar=True)
X_valid = embedder.encode(valid_texts, batch_size=128, show_progress_bar=True)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=42),
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_valid)[:, 1]
```

**Notes**: Cheap, reproducible baseline; useful for active learning and audit sampling even when fine-tuned encoders win.

### Example 3: LoRA sequence-classification arm for small open-weight LMs

**Source**: Hugging Face PEFT docs; Li et al. (2025 ACL Industry); Zhang et al. (2025 Findings EMNLP).  
**Package version**: current `peft` docs.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # or Gemma/Phi/Llama if license/compute fit
tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(base, peft_config)
model.print_trainable_parameters()
```

**Notes**: Run only after encoder baselines. Report invalid-output rate for prompting, GPU memory, inference cost, and whether exact model revision/license is archived.

## Version and Compatibility Notes

- `transformers`: current docs use `AutoModelForSequenceClassification` and `Trainer`; use explicit model revisions for reproducibility. Some older examples use `evaluation_strategy`; current examples increasingly use `eval_strategy`.
- `sentence-transformers`: `all-MiniLM-L6-v2` is 384-dimensional, fast, and general-purpose; embedding model rankings change, so record exact model ID/revision.
- `peft`: for classification, set `task_type=TaskType.SEQ_CLS` so classification heads are handled/trainable; decoder models may need padding-token configuration.
- `scikit-learn`: `class_weight="balanced"` scales by inverse class frequency; `CalibratedClassifierCV` supports sigmoid/isotonic and newer docs mention temperature scaling. Avoid isotonic with small calibration sets (`≪1000`) because of overfitting risk.
- Reproducibility: persist splits, seeds, package versions, HF model revision hashes, metrics JSON, predictions, and calibration artifacts.

## Search Strategy and Exact Search Queries Used

Local seed sources: `docs/research/20260605-literature-short-text-classification.md`, `docs/research/20260605-literature-synthesis-map.md`, and `docs/research/20260605-replication-short-text-model-alternatives.md`.

Exact web/doc queries used in this run:

1. `2026 short text classification BERT RoBERTa DeBERTa DistilBERT ELECTRA ALBERT comparison small language models classifiers arXiv ACL`
2. `2026 small open weight language models text classification compared with BERT prompting LoRA sequence classification head`
3. `computational social science short text classification BERT open-ended survey responses political science nonprofit mission statements 2024 2025`
4. `Hugging Face Transformers Trainer text classification fine tuning sequence classification current syntax`
5. `Sentence Transformers encode embeddings classification examples current syntax`
6. `PEFT LoRA sequence classification Hugging Face current syntax`
7. `scikit-learn current documentation CalibratedClassifierCV calibration_curve class_weight balanced logistic regression 2026`

## Key Papers

| # | Paper | Why it matters |
|---|---|---|
| 1 | Devlin et al. (2019), “BERT,” NAACL, https://doi.org/10.18653/v1/N19-1423 | Base encoder architecture and sequence-classification fine-tuning paradigm. |
| 2 | Liu et al. (2019), “RoBERTa,” arXiv, https://arxiv.org/abs/1907.11692 | Stronger BERT pretraining; often superior supervised baseline. |
| 3 | Clark et al. (2020), “ELECTRA,” ICLR/arXiv, https://arxiv.org/abs/2003.10555 | Discriminator pretraining; strong sample/compute efficiency. |
| 4 | Lan et al. (2020), “ALBERT,” ICLR/arXiv, https://arxiv.org/abs/1909.11942 | Parameter-efficient BERT alternative; not always latency-efficient. |
| 5 | Sanh et al. (2019), “DistilBERT,” arXiv, https://arxiv.org/abs/1910.01108 | 40% smaller, faster BERT-like baseline; good production tradeoff. |
| 6 | He et al. (2021), “DeBERTa,” ICLR/arXiv, https://arxiv.org/abs/2006.03654 | Often high-performing encoder; higher memory/latency. |
| 7 | Reimers & Gurevych (2019), “Sentence-BERT,” EMNLP-IJCNLP, https://doi.org/10.18653/v1/D19-1410 | Embeddings + simple classifier baseline; enables cheap few-shot/audit workflows. |
| 8 | Sun et al. (2019), “How to Fine-Tune BERT for Text Classification?”, arXiv, https://arxiv.org/abs/1905.05583 | Fine-tuning sensitivity: LR, epochs, layer pooling, truncation, further pretraining. |
| 9 | Gururangan et al. (2020), “Don’t Stop Pretraining,” ACL, https://doi.org/10.18653/v1/2020.acl-main.740 | DAPT/TAPT evidence for domain/task adaptation. |
| 10 | Gweon & Schonlau (2024), “Automated Classification for Open-Ended Questions with BERT,” JSSAM, https://doi.org/10.1093/jssam/smad015 | Closest survey short-text small-label BERT evidence. |
| 11 | Laurer et al. (2024), “Less Annotating, More Classifying,” Political Analysis, Cambridge link | BERT-NLI improves low-data/imbalanced political text classification. |
| 12 | Ma (2021), “Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector,” NVSQ, https://doi.org/10.1177/0899764020968153 | Closest nonprofit text/NTEE benchmark and reproducible code. |
| 13 | Fyall, Moore & Gugerty (2018), “Beyond NTEE Codes,” NVSQ, https://doi.org/10.1177/0899764018768019 | Mission statement content coding for nonprofit activity measurement. |
| 14 | Haq, Sharma & Bhattacharyya (2023), “Angel,” EMNLP Industry, https://doi.org/10.18653/v1/2023.emnlp-industry.77 | Nonprofit mission/description search and representation use case. |
| 15 | Mellon et al. (2024), “Do AIs know what the most important issue is?”, Research & Politics, https://doi.org/10.1177/20531680241231468 | LLMs vs DistilRoBERTa/BERT embeddings/SVM on sparse open-text survey coding. |
| 16 | Törnberg (2024), “Large Language Models Outperform Expert Coders…,” Social Science Computer Review, https://doi.org/10.1177/08944393241286471 | Shows frontier LLM coding can beat supervised baselines on some social-media tasks. |
| 17 | Wang & Qu (2024), “Selecting Between BERT and GPT…,” arXiv, https://arxiv.org/abs/2411.05050 | Political-science guidance: GPT useful at tiny n/binary tasks; BERT better near 1,000 labels. |
| 18 | Zhang et al. (2025), “Do BERT-Like Bidirectional Models Still Perform Better…,” Findings EMNLP, https://aclanthology.org/2025.findings-emnlp.1033/ | Current BERT-like vs LLM states/zero-shot evidence; task-aware model choice. |
| 19 | Vajjala & Shimangaud (2025), “Small Language Models in the Real World,” ACL Industry, https://aclanthology.org/2025.acl-industry.68/ | Small generative LM prompting/SFT tradeoffs for industrial classification. |
| 20 | Burnham et al. (2025/2026), “Political DEBATE,” Political Analysis/Cambridge | Efficient zero/few-shot NLI classifiers for political text. |

## Synthesis of Model-Choice Evidence

1. **Fine-tuned encoder-only models remain the supervised default.** BERT/RoBERTa/DeBERTa/ELECTRA generally dominate when labels are available and the construct is pattern-driven.
2. **Model ranking is task- and budget-dependent.** DeBERTa often has high accuracy but higher compute; DistilBERT is the best speed/quality baseline; ELECTRA is worth testing for sample efficiency; ALBERT is small but not always faster in wall-clock terms.
3. **Frozen embeddings are not just weak baselines.** MiniLM/SBERT + logistic regression is cheap, stable, auditable, and competitive in low-resource qualitative workflows.
4. **Small-data regimes are unstable.** Below ~100–200 labels, prompting/NLI/embedding approaches can match fine-tuning. Around 500–1,000 clean labels, fine-tuned encoders usually become preferable.
5. **Domain-adaptive pretraining can help but should be secondary.** TAPT/DAPT is justified if unlabeled mission text is large; short/noisy domain corpora can overfit or add noise.
6. **Small open-weight generative LMs are comparison arms, not default production classifiers.** Prompting can be competitive for simple binary tasks or no-label exploration, but supervised encoders remain cheaper and more reproducible at scale. LoRA/SFT can help but increases operational complexity.
7. **Calibration and prevalence are central for applied economics.** Report PR-AUC, MCC, Brier score, ECE/reliability, threshold tables, and prevalence error—not accuracy alone.

## Practical Experiment Grid Recommendations

1. **Freeze data first:** create stratified train/validation/test splits and a manifest with `DATA_OF_CHOICE`, label source, prevalence, seed, and text length distribution.
2. **Baselines:** TF-IDF word+char n-gram logistic regression/SVM; `all-MiniLM-L6-v2` + logistic regression.
3. **Main encoders:** `distilbert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`, `google/electra-base-discriminator`; optional `albert-base-v2`, `answerdotai/ModernBERT-base`.
4. **Fine-tuning grid:** LR `{1e-5, 2e-5, 3e-5, 5e-5}`; DeBERTa include `{5e-6, 8e-6}` if unstable; epochs `{3,5,8,10}` with early stopping; batch `{16,32}`; max length `{128,256}`; 5 seeds.
5. **Loss/imbalance:** unweighted CE and weighted CE first; focal loss only if severe imbalance or poor minority recall persists.
6. **Data-size curve:** train at 50, 100, 200, 500, 1,000, all labels; report mean±sd.
7. **Small-LM arm:** one local open-weight model via zero/few-shot prompting; if compute permits, LoRA sequence classification. Record invalid outputs, latency, GPU memory, and model revision.
8. **Calibration:** calibrate finalists on clean validation labels; compare Platt/sigmoid, temperature scaling, and isotonic only if calibration set is large enough.

## Gaps, Caveats, and Papers to Read First

- Direct evidence on binary classification of **organization mission statements** is thin; borrow from nonprofit NTEE coding, open-ended surveys, tweets, abstracts, and political text.
- LLM prompting studies are version-sensitive and can be hard to reproduce; archive prompts, raw outputs, decoding settings, model IDs, and dates.
- Label noise and construct ambiguity may dominate architecture differences; invest in human audit/adjudication.
- Papers to read first: Gweon & Schonlau (2024), Ma (2021), Laurer et al. (2024), Gururangan et al. (2020), Zhang et al. (2025), Wang & Qu (2024), Mellon et al. (2024).

## Handoff Quality Gate

- From Replication Packages: concrete entries present.
- Package Recommendations: concrete entries present.
- Implementation Examples: concrete entries present.
- Version and Compatibility Notes: concrete entries present.
