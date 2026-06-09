# Repository Audit — Religious vs. Non-Religious Nonprofit Mission Classifier

**Audited:** 2026-06-02
**Repo:** `BINARY-CLASSIFIER-MISSIONS` (branch `master`)
**Scope:** End-to-end ML pipeline that (1) labels US nonprofit mission/activity texts as
religious (1) / non-religious (0) with GPT-4o-mini, then (2) fine-tunes `bert-base-uncased`
on those labels and classifies the full IRS Form 990 corpus.
**Type:** Read-only audit. The only artifact produced is this document; no repository
source or data files were modified.

---

## 1. Summary & Scope

The "real" pipeline is **2 Python scripts + 5 Jupyter notebooks**:

| File | Role |
|------|------|
| `generate_training_data.py` | GPT-4o-mini labelling (Stage 1) |
| `split_data.py` | stratified split + oversampling (Stage 2) |
| `final_finetuning.ipynb` | BERT fine-tune on **missions** (Stage 3) |
| `final_finetuning_activities.ipynb` | BERT fine-tune on **activities** (Stage 3) |
| `inference.ipynb` | batch prediction over full corpus (Stage 4) |
| `inspect_results.ipynb` | results inspection (Stage 5) |
| `test.ipynb` | scratch / exploratory notebook (not part of the pipeline) |

Everything else in the tree is agent/skill scaffolding (`.agents/`, `.claude/`,
`.opencode/`, `opencode.json`, `skills-lock.json`) and the documentation site of the
**upstream** project is consulted separately.

**Artifacts present locally:** the labelled CSVs and all train/test splits are present.
**Artifacts absent locally (gitignored):** all `*.parquet` inputs/outputs, the fine-tuned
model directories (`my_model`, `my_model_prompt2`, `model_on_missions`,
`model_on_activities`), and the `results/` training checkpoints. The audit therefore
verifies the parquet stages **by code inspection** of the read/write calls, and verifies
the CSV stages **directly** (headers + row counts).

**What was cross-checked:** every filename, column name, and row count cited below was
verified against `generate_training_data.py`, `split_data.py`, the code cells of the 5
notebooks, the CSV headers (`head -1`) and line counts (`wc -l`), and the upstream
`output-contracts.md` / `we-want-to-do-wobbly-treehouse.md`.

---

## 2. Pipeline Map (script → input file(s)/columns → output file(s)/columns)

`DATA_OF_CHOICE` (currently `'activities'` in both scripts) switches the missions vs.
activities branch.

| Stage | Driver | Input file(s) | Input column(s) | Output file(s) | Output column(s) |
|-------|--------|---------------|-----------------|----------------|------------------|
| **0. Upstream** | (NonProfitData project) | — | — | `data/501c3_charity_geocoded_missions_clean.parquet`, `data/501c3_charity_geocoded_activities_clean.parquet` *(gitignored, not present)* | missions: `CANONICAL_MISSION`; activities: `CONCATENATED_ACTIVITY` |
| **1. Label** | `generate_training_data.py` | `data/501c3_charity_geocoded_{DATA}_clean.parquet` | `CONCATENATED_ACTIVITY` (hardcoded — see §4) | `data/classified_{DATA}_gpt4omini_PROMPT2.csv` | `mission, label, reason` |
| **2. Split** | `split_data.py` | `data/classified_{DATA}_gpt4omini_PROMPT2.csv` | `label` (stratify) | `train_test_datasets/train_{DATA}_PROMPT2.csv`, `test_{DATA}_PROMPT2.csv`, `train_balanced_{DATA}_PROMPT2.csv` | `mission, label, reason` |
| **3. Fine-tune (missions)** | `final_finetuning.ipynb` | `train_balanced_PROMPT2.csv`, `test_PROMPT2.csv` | `mission` (tokenize), `label` | model dir `./my_model_prompt2` | — |
| **3. Fine-tune (activities)** | `final_finetuning_activities.ipynb` | `train_balanced_activities_PROMPT2.csv`, `test*` | `mission`, `label` | **none** (`save_model` commented out) | — |
| **4. Inference** | `inference.ipynb` | `./my_model_prompt2`, `data/501c3_charity_geocoded_missions_clean.parquet` | `CANONICAL_MISSION` | `data/501c3_charity_geocoded_missions_clean_classified_prompt2.parquet` | adds `RELIGIOUS` (0/1) |
| **5. Inspect** | `inspect_results.ipynb` | `data/501c3_charity_geocoded_activities_clean_classified.parquet` ⚠️ **no producer in repo** | `RELIGIOUS_ACTIVITY` ⚠️, `CONCATENATED_ACTIVITY` | — (display only) | — |

### Verified row counts (CSV data rows, header excluded)

| File | Rows | Expected |
|------|------|----------|
| `data/classified_missions_gpt4omini_PROMPT2.csv` | 3,000 | first 3,000 non-null |
| `data/classified_activities_gpt4omini_PROMPT2.csv` | 3,000 | first 3,000 non-null |
| `train_PROMPT2.csv` / `test_PROMPT2.csv` | 2,100 / 900 | 70/30 of 3,000 ✓ |
| `train_activities_PROMPT2.csv` / `test_activities_PROMPT2.csv` | 2,100 / 900 | 70/30 of 3,000 ✓ |
| `train_balanced_PROMPT2.csv` (missions) | 3,792 | 2 × majority |
| `train_balanced_activities_PROMPT2.csv` | 3,804 | 2 × majority |
| `classified_missions_gpt4omini_PROMPT1.csv` (legacy) | 3,000 | — |

All labelled/split CSVs carry the literal header **`mission,label,reason`** — including the
activities files, where the text is an *activity summary*, not a mission (see §4).
Legacy non-suffixed `train.csv` / `test.csv` / `train_balanced.csv` are also present.

---

## 3. Upstream Interface & Compatibility

The upstream producer (`~/Documents/Projects/NonProfitData`) was refactored. Two upstream
documents disagree about the contract this repo depends on:

- **`docs/.../output-contracts.md`** (current published contract) still lists
  `data/processed/501c3_charity_geocoded_missions_clean.parquet` and
  `…_activities_clean.parquet`, with keys `EIN2` + `TAX_YEAR`. This matches the names this
  repo hardcodes.
- **`.agents/plans/we-want-to-do-wobbly-treehouse.md`** (refactor plan) **renames** the
  cross-sections to **`missions_cross_section.parquet`** / **`activities_cross_section.parquet`**
  (one row per `EIN2`, `COMMON_LEVEL1 == "501C3 CHARITY"`), and explicitly states *"no
  `501c3_*` side-outputs"*. The old `501c3_*_clean.parquet` producers are moved to
  `src/panel/legacy/`.

**Implications:**

- **Filename drift (Med):** if the refactor lands, the `501c3_charity_geocoded_*_clean.parquet`
  names this repo reads at Stage 1 and Stage 4 will no longer be produced. The classifier
  would read a stale legacy file or fail outright.
- **Columns appear stable:** the plan's cross-sections still expose `CANONICAL_MISSION`
  (missions) and `CONCATENATED_ACTIVITY` (activities), the exact columns this repo consumes.
  Only the *file names* change, not the column contract.
- **No join key carried through (High):** the classified output keeps only the original
  text columns + the new `RELIGIOUS` label. The upstream key **`EIN2`** is **never selected
  into the classified parquet**, so predictions cannot be joined back to the panel by key —
  only by row order or by exact text match (fragile if upstream dedup/cleaning changes row
  order or text).

---

## 4. LLM Labelling Design & Idempotency

**Design (`generate_training_data.py`):**

- Two prompt templates ("mission" and "activities") that encode *economics of religion*
  rules + 7 few-shot examples; both ask for strict JSON.
- `model="gpt-4o-mini"`, `temperature=0`, `response_format={"type": "json_object"}`.
- Output validated by Pydantic `MissionLabel(label: int | None, reason: str)`.
- `time.sleep(0.2)` between calls; `RateLimitError` → exponential backoff (`2**attempt`),
  up to `max_retries=5`.
- Checkpoint to CSV every 100 rows and at the end.
- Resume: if the output CSV exists, `start_index = len(existing_df)`.

**Idempotency / correctness gaps:**

1. **Resume keyed on row count, not content (Med):** `start_index = len(existing_df)`. If
   the input subset, ordering, or `DATA_OF_CHOICE` changes between runs, the resume offset
   points at the wrong row — silently mislabelling/misaligning the continuation. There is a
   `# TODO: Review logic of this code` already in place acknowledging this.
2. **Failed rows poison the resume offset (Med):** a non-rate-limit exception appends a
   `{"label": None, "reason": <error>}` row and `break`s (no retry). That row still counts
   toward `len(existing_df)`, so a re-run **skips** the failed item instead of retrying it.
3. **`label` can be `None` (Med):** both on hard errors and on malformed/unvalidated JSON
   (`MissionLabel(label=None, …)`). Nothing downstream handles `None` — `split_data.py`
   stratifies on `label` and the notebooks cast to int; a `None`/NaN label would either
   error or be silently coerced.
4. **Order-dependent sample (High-ish / flagged High in register):** the supervised set is
   `missions[0:3_000]` after `.dropna()` — the first 3,000 rows in upstream file order, with
   **no shuffle and no seed**. The labelled sample is entirely determined by upstream row
   ordering; any upstream re-sort changes the training set.

---

## 5. BERT Fine-Tuning Configuration

Verified in `final_finetuning.ipynb` (missions); `final_finetuning_activities.ipynb` is
structurally identical except its `save_model` is commented out.

**Model & layers:**
- `bert-base-uncased`, `num_labels=2`.
- Freeze the whole encoder, then unfreeze `layer.9 / 10 / 11` + keep the classifier head
  trainable.

**Tokenization (mixed static/dynamic padding — Low):**
- `tokenize_function` uses `padding='max_length', truncation=True, max_length=128` →
  **static padding to 128**.
- `DataCollatorWithPadding(tokenizer)` is *also* passed to the trainer for **dynamic
  padding**. Because the data is already padded to 128, the collator is a no-op here;
  harmless but contradictory — pick one strategy.

**Training arguments:**
- `learning_rate=5e-5`, `per_device_train/eval_batch_size=16`, `num_train_epochs=10`,
  `weight_decay=0.01`, `fp16=True`, `eval_strategy="epoch"`, `save_strategy="epoch"`,
  `save_total_limit=2`, `load_best_model_at_end=True`, `EarlyStoppingCallback(patience=4)`.
- Manual `optim.AdamW(model.parameters())` (default LR, **not** `5e-5` — the
  `TrainingArguments.learning_rate` is overridden by the externally-supplied optimizer) +
  linear `get_scheduler` passed via `optimizers=(optimizer, scheduler)`.
- `num_training_steps = len(train) // batch_size * epochs` — integer floor (`//`)
  slightly under-counts steps, so the LR scheduler reaches ~0 a touch early (minor).

**Class weights — doc/code mismatch + dead computation (Med):**
- Cell 2 computes `class_weights = compute_class_weight('balanced', …)` and prints it…
- …then Cell 5 **overwrites** it with a hardcoded `class_weights = np.array([1.0, 1.1])`,
  used by the custom weighted `CrossEntropyLoss`. The balanced computation is **dead code**.
- The **README documents `[1.0, 1.3]`** (and narrates "1.3x penalty"), but the **code uses
  `1.1`**. Doc and code disagree on the single most-tuned hyperparameter.

**Best-model selection — silently not F1 (Med):**
- `load_best_model_at_end=True` but **`metric_for_best_model` is not set**, so it defaults
  to eval **loss**. The README and inline comments claim the best model / early stopping is
  chosen by **F1**. As written, both "best model" and early-stopping patience track loss,
  not F1.

---

## 6. Evaluation Metrics — Present vs. Exported

**Present (computed in-notebook):**
- `compute_metrics` returns **accuracy, f1 (binary), precision, recall** per epoch via
  `sklearn.metrics` — these are logged each eval epoch and in the final `trainer.evaluate()`.
- A `classification_report` and a `confusion_matrix` (+ `ConfusionMatrixDisplay.plot()`) are
  printed/plotted at the end.

**Not exported (High):**
- `evaluate.load('f1')` is assigned to `metric` but **never used** (metrics come from
  sklearn instead) — dead load.
- **No metrics file** (no JSON/CSV of accuracy/F1/precision/recall) is written to disk.
- **No confusion-matrix image** is saved — only `disp.plot()` renders inline.
- The inference class-balance counts (e.g. first model `0:471429 / 1:96158`; second model
  `0:488977 / 1:78610`) live as a **code comment** in `inference.ipynb`, not an exported
  artifact.
- **Throughput** (rows/sec at inference) — research-critical for scaling to the full corpus
  — is **not measured or logged** anywhere.

Net: every reported number in the README's "Results" section (F1 ~0.85–0.90, etc.) is
unverifiable from repository artifacts because nothing is persisted.

---

## 7. Reproducibility Hygiene

**Seeds:**
- `random_state=42` is set in `split_data.py` (split, oversample, shuffle) — good.
- **No `transformers.set_seed`, no `torch.manual_seed`, no `np.random.seed`** anywhere in
  either fine-tuning notebook → **training is non-reproducible** (weight init of the
  classifier head, data shuffling, fp16 nondeterminism). Confirmed: `grep` for
  `set_seed|manual_seed|np.random.seed` across all notebooks returns nothing.
- The labelling subset selection (`[0:3_000]`) is unseeded and order-dependent (§4).

**Dependencies (`requirements.txt`):**
- **Fully unpinned** except `accelerate>=0.26.0` and `openai>=0.27.0`. `pandas`,
  `transformers`, `torch`, `datasets`, `evaluate`, `scikit-learn`, `numpy`, `matplotlib`
  have **no version bounds** → silent breakage on a future resolve.
- `dotenv` is listed, but the importable PyPI package is **`python-dotenv`** (`import
  dotenv` works only because `python-dotenv` provides the `dotenv` module — `pip install
  dotenv` installs a different, deprecated stub). Should read `python-dotenv`.
- **No Python version pin** in the repo (env is 3.12; README says "Python 3.8+", but the
  code uses `int | None` PEP-604 syntax which requires **3.10+**).

**Hardcoded / foreign paths:**
- Relative data paths throughout (fine, but couples runs to CWD).
- `test.ipynb` contains an absolute **Windows** path
  `C:\Users\rqg886\Desktop\BINARY-CLASSIFIER-MISSIONS\data\classified_activities_gpt4omini_PROMPT2.csv`.
- Other notebooks' saved output cells carry another author's absolute macOS paths
  (`/Users/caro/Desktop/...`). Harmless, but leaks cross-machine provenance — clear on
  commit.

**Caching / `.gitignore`:**
- `*.parquet`, `results`, `my_model`, `model_on_missions`, `model_on_activities` are
  gitignored — appropriate.
- **Malformed line (Low):** `.gitignore` contains
  `my_model_prompt2my_model_prompt2/model.safetensors` — two paths concatenated with no
  separator. As written it ignores a nonexistent path; the intended `my_model_prompt2/`
  directory is **not** actually ignored by this line (it is not matched). Should be a clean
  `my_model_prompt2/`.

**Config duplication:**
- `DATA_OF_CHOICE` is a literal in **both** `generate_training_data.py` and `split_data.py`
  and must be edited in lockstep; there is no single shared config. The PROMPT1 → PROMPT2
  history is implied only by filenames, not recorded.

---

## 8. Inconsistency Register

| ID | Location | Issue | Severity | Recommended fix |
|----|----------|-------|----------|-----------------|
| **R-01** | `inference.ipynb` → `inspect_results.ipynb` | Inference writes **missions** output `…_missions_clean_classified_prompt2.parquet` with column **`RELIGIOUS`**; inspect reads **`…_activities_clean_classified.parquet`** (activities, *no* `_prompt2`) and filters **`RELIGIOUS_ACTIVITY`** — a file no notebook produces and a third column name written nowhere. | **High** | Make the producer/consumer agree: have inference also emit the activities output (run the activities branch and persist its model first), standardize one column name (`RELIGIOUS`), and point `inspect_results.ipynb` at the file that is actually written. |
| **R-02** | both fine-tuning notebooks | No `set_seed`/`torch`/`numpy` seeding → non-reproducible training. | **High** | Add `transformers.set_seed(42)` (or `set_seed` + `torch.use_deterministic_algorithms`) at the top of both notebooks. |
| **R-03** | `final_finetuning*.ipynb` | No metrics, no confusion-matrix image, no throughput exported to disk; README numbers unverifiable. | **High** | After `trainer.evaluate()`, dump metrics to `results/metrics_{DATA}_PROMPT2.json`, `plt.savefig` the confusion matrix, and log rows/sec in inference. |
| **R-04** | inference output schema | `EIN2` join key never carried into the classified parquet → cannot key-join predictions back to the panel. | **High** | Select `EIN2` (+ `TAX_YEAR` if needed) into `data` before `to_parquet`, so the label table is joinable by key, not row order. |
| **R-05** | `final_finetuning*.ipynb` cell 5 vs README | Hardcoded `class_weights=[1.0, 1.1]` overrides the computed `compute_class_weight` value (dead code); README documents `[1.0, 1.3]`. | **Med** | Decide on one source of truth: either use the computed balanced weights or a documented constant, and update README to match the code's actual value. |
| **R-06** | `final_finetuning*.ipynb` TrainingArguments | `load_best_model_at_end=True` without `metric_for_best_model` → selects on eval **loss**, not F1, contradicting README. | **Med** | Set `metric_for_best_model="f1"`, `greater_is_better=True`. |
| **R-07** | `requirements.txt` | Fully unpinned (except 2 pkgs); no Python pin; pins for a torch/transformers stack are essential for reproducibility. | **Med** | Pin exact versions (`pip freeze` of the working env), add `python_requires`/`.python-version` (3.10+). |
| **R-08** | `generate_training_data.py` | Idempotency gaps: count-based resume, failed rows consume the offset (never retried), `label=None` unhandled downstream. | **Med** | Resume by content/id (hash of text), persist a status column, and re-attempt `None`/error rows on rerun; validate no `None` labels before Stage 2. |
| **R-09** | this repo vs upstream refactor | Hardcoded `501c3_*_clean.parquet` names will be renamed to `*_cross_section.parquet` (no more `501c3_*` side-outputs) by the upstream refactor. | **Med** | Parameterize input paths via config; track the upstream contract; switch to `missions_cross_section.parquet` / `activities_cross_section.parquet` when the refactor lands (columns are unchanged). |
| **R-10** | `generate_training_data.py` | Training sample is `missions[0:3_000]` — first rows in upstream order, unseeded, unshuffled. | **Med** | Shuffle with a fixed seed before slicing, or sample with `df.sample(n=3000, random_state=…)`, and record the selection. |
| **R-11** | `test.ipynb`; saved output cells | Absolute Windows path `C:\Users\rqg886\…`; foreign macOS paths `/Users/caro/Desktop/…` in outputs. | **Low** | Replace with relative paths; clear notebook outputs before commit (`nbstripout`). |
| **R-12** | `.gitignore` | Malformed line `my_model_prompt2my_model_prompt2/model.safetensors` — two paths concatenated; does not actually ignore `my_model_prompt2/`. | **Low** | Replace with `my_model_prompt2/`. |
| **R-13** | `requirements.txt` | `dotenv` should be `python-dotenv`. | **Low** | Rename to `python-dotenv`. |
| **R-14** | `generate_training_data.py` + `split_data.py` | `DATA_OF_CHOICE` duplicated; must be edited in lockstep; no shared config. | **Low** | Extract to a single `config.py`/env var imported by both scripts. |
| **R-15** | `README.md` "Project Structure" / "Results" | References files/dirs not present as described (`my_model/`, `results/checkpoint-*`, PROMPT1-only layout) and weight value `[1.0, 1.3]` ≠ code. | **Low** | Reconcile README with the actual tree, PROMPT2 filenames, and real hyperparameters. |
| **R-16** | `final_finetuning*.ipynb` | Mixed padding: static `max_length=128` in tokenizer **and** `DataCollatorWithPadding` (dynamic) — collator is a no-op. | **Low** | Use one: dynamic padding (drop `padding='max_length'`, keep collator) is the efficient choice. |
| **R-17** | `final_finetuning*.ipynb` | `evaluate.load('f1')` assigned to `metric` but never used. | **Low** | Remove the dead load (metrics come from sklearn). |

---

## 9. "What I Found vs. Did Not Find" Inventory

| Item | Present? | Notes |
|------|:--------:|-------|
| Labelling script (`generate_training_data.py`) | ✅ | verified |
| Split script (`split_data.py`) | ✅ | verified |
| Missions fine-tune notebook | ✅ | saves `./my_model_prompt2` |
| Activities fine-tune notebook | ✅ | `save_model` **commented out** — no persisted activities model |
| Inference notebook | ✅ | writes `…_missions_clean_classified_prompt2.parquet`, col `RELIGIOUS` |
| Inspect notebook | ✅ | but reads a parquet **no repo notebook produces** (R-01) |
| Labelled CSVs (3,000 each) | ✅ | header `mission,label,reason`; counts verified |
| Train/test/balanced CSVs (PROMPT2) | ✅ | 2,100 / 900 / 3,792 (miss), 3,804 (act) verified |
| Legacy PROMPT1 + non-suffixed CSVs | ✅ | `classified_missions_gpt4omini_PROMPT1.csv`, `train.csv`, `test.csv`, `train_balanced.csv` |
| Stage-0/4 input parquet (`501c3_*_clean.parquet`) | ❌ | gitignored, **not present locally** — verified by code only |
| Inference output parquet | ❌ | gitignored, not present locally |
| Inspect's input (`…_activities_clean_classified.parquet`) | ❌ | **no producer in repo** + gitignored |
| Fine-tuned model dirs (`my_model_prompt2`, etc.) | ❌ | gitignored, not present locally |
| `results/` checkpoints | ❌ | gitignored, not present |
| Persisted metrics file | ❌ | none written (R-03) |
| Confusion-matrix image file | ❌ | plotted inline only (R-03) |
| Throughput log | ❌ | not measured (R-03) |
| `EIN2` in classified output | ❌ | dropped — no key join-back (R-04) |
| Training seed | ❌ | none in either notebook (R-02) |
| Split seed | ✅ | `random_state=42` in `split_data.py` |
| Pinned dependencies | ❌ | mostly unpinned (R-07) |
| Python version pin | ❌ | none in repo (code needs 3.10+) |
| `requirements.txt` | ✅ | but `dotenv` mis-named (R-13) |
| README | ✅ | several mismatches vs code (R-05, R-06, R-15) |
| Upstream contract docs | ✅ | `output-contracts.md` (legacy names) + refactor plan (new names) consulted (R-09) |
| Foreign/Windows absolute paths | ⚠️ present | `test.ipynb` Windows path; `/Users/caro/…` in outputs (R-11) |
| Malformed `.gitignore` line | ⚠️ present | concatenated paths (R-12) |

---

### Severity tally

- **High (4):** R-01 (naming break, no producer) · R-02 (no training seed) · R-03 (no
  exported metrics) · R-04 (no `EIN2` join key).
- **Med (6):** R-05 (class-weight doc/code + dead compute) · R-06 (`metric_for_best_model`
  unset) · R-07 (unpinned deps) · R-08 (idempotent-resume gaps) · R-09 (upstream filename
  drift) · R-10 (unseeded sample selection).
- **Low (7):** R-11 (foreign paths) · R-12 (`.gitignore` line) · R-13 (`dotenv`) · R-14
  (duplicated `DATA_OF_CHOICE`) · R-15 (README drift) · R-16 (mixed padding) · R-17 (dead
  `evaluate.load`).

*End of audit.*
