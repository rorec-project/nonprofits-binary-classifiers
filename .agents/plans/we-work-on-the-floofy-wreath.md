# Plan — Re-engineer the Nonprofit Mission Classifier (Points 1–2)

## Context

The current pipeline is a pair of flat scripts (`generate_training_data.py`, `split_data.py`) plus notebooks, with documented defects in `docs/audits/repo_auditing.md`: no training seed (non-reproducible), class-weight doc/code mismatch, `EIN2` join key dropped from inference output, broken producer/consumer between inference and inspection, `DATA_OF_CHOICE` duplicated across two files, and `bert-base-uncased` (a dominated model) hard-wired. We are rebuilding it from scratch into a reproducible, config-driven `src/` package that generalizes beyond religious classification (pregnancy centers, education, international, …) and runs on UCloud B200 GPUs.

This plan details **point 1 (upstream cleanup + decisions)** and **point 2 (LLM annotation)** of `.agents/stubs/pipeline-roadmap.md`, and stubs points 3–6 so they slot in. Every decision is grounded in the research handoffs in `.agents/docs/` and in a direct profile of the real input corpus (`missions_cross_section.parquet`, 560,351 rows).

The deliverable of points 1–2 is a **versioned, frozen, LLM-labelled train/validation/test dataset of high-quality missions** (keyed by `EIN2`), ready for the fine-tuning phase.

---

## Locked decisions

| #                       | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Rationale (source)                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Entity                  | First target = **missions**, field = **`LONGEST_MISSION`**; entity is a config parameter (generalizes the old `DATA_OF_CHOICE`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `LONGEST_MISSION` is strictly longer than `CANONICAL_MISSION` in 80% of rows; already lowercased/ASCII/non-empty.                                                                                                                                                                                                                                                                           |
| Label architecture      | **LLM-as-primary**, stored in a **long/tidy label table** that is weak-supervision-ready. **No Snorkel.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Snorkel's last code commit was ~2.5 yrs ago, team pivoted to a commercial product, only nominally Py3.13-compatible. The long/tidy store pivots cleanly into the maintained alternatives later.                                                                                                                                                                                             |
| Aggregation / denoising | Majority vote now; **crowd-kit** (Dawid-Skene) and **cleanlab 2.9** (CROWDLAB/noise detection) as drop-in comparison arms later. Both Py3.13-ready.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `20260606-tech-llm-weak-supervision-noisy-labels.md` + live package scan.                                                                                                                                                                                                                                                                                                                   |
| Annotator               | Provider-agnostic adapter. **Bake-off slate** (pin exact IDs at bake-off time): one current frontier **OpenAI** snapshot (closed reference) + **Qwen3-235B-A22B-Instruct (Apache-2.0)** (open production candidate) + one lightweight Apache-2.0 control (Qwen3-32B / Gemma-4). **Production = multi-model × multi-prompt ensemble**: the open-weight Qwen3 on the B200s (free/reproducible backbone) **+** an OpenAI snapshot **+** a 2nd open model, each × 2–3 prompts → cross-model **and** cross-prompt labels per mission; temp 0, guided-JSON. **No Anthropic** (cost); the OpenAI source is the one paid component (~20k × prompts — bounded, budget it). | GPT-4-class is the validated CSS prior (Carlson & Burbano 2026; GivingTuesday); open-weight is frontier-class on an easy binary _and_ permanently pinnable/archivable (reproducibility). Open-weight self-hosted labeling is ≈free; the OpenAI source adds cross-model disagreement signal at a bounded API cost. Multiple noisy sources are exactly what crowd-kit/cleanlab consume later. |
| NTEE role               | NTEE major group (first letter of `NTEE_IRS`) is a **balanced-coverage stratifier only**, never the label basis. Religious signal is sought/enriched **within every macro-group**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 57% of religious signal sits outside group X; the classifier must see it across the full spectrum.                                                                                                                                                                                                                                                                                          |
| Sample sizing           | Over-provision the **silver pool to ~20k** high-quality missions; the _optimal training N is found empirically_ via the point-3 learning curve (extensible to ~40k if not plateaued). Human **gold ~400**, kept small — the ~200-row frozen test is reported with **bootstrap CIs** (small-N).                                                                                                                                                                                                                                                                                                                                                                    | Self-hosted labeling is ≈free; the plateau, not a guessed N, defines "optimal."                                                                                                                                                                                                                                                                                                             |
| Scope of the classifier | Validated scope = **HIGH + MEDIUM** quality tiers — silver pool, validation, and test are all sampled from both tiers (train/serve distribution match). **LOW / bare-label missions are handled by the rule layer at inference, not dropped** (they are enriched in true positives).                                                                                                                                                                                                                                                                                                                                                                              | Advisor + corpus profile: ~6.8k bare-label churches (~9% of positives) fall in LOW; dropping them biases prevalence (point 6).                                                                                                                                                                                                                                                              |
| Reproducibility         | One global `SEED` in config; every stochastic step (sampling, splitting, training, annotation temperature/seed) is seeded. `EIN2` carried through **every** artifact.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Roadmap point 1; audit R-02/R-04/R-10.                                                                                                                                                                                                                                                                                                                                                      |

---

## Corpus facts that ground the design (from the data profile)

> The full quantitative + qualitative findings behind this section — corpus profile, NTEE join, the quality rubric `Q`, per-stratum HIGH pools, the verbatim example bank, religious-signal-by-sector contrasts, and boundary-case prevalences — live in the companion annex **`.agents/plans/we-work-on-the-floofy-wreath-annex.md`**. This section is the executive summary.

- 560,351 missions, one per `EIN2` (unique) — the `COMMON_LEVEL1 == "501C3 CHARITY"` subset of the corpus codebook's **586,718**-EIN2 mission set (the ~4.5% gap is the charity filter; confirm/reconcile at implementation). `LONGEST_MISSION`: 0 empty, 99.2% English, median 22 words.
- Dominant quality problem is **brevity** (9.3% <5 words, 25% <10 words), not junk (~2–3% boilerplate). 5.8% exact duplicates (dominated by single-word missions). ~0.75% soft-truncated at ~1000 chars.
- NTEE join via BMF `NTEE_IRS[0]` (A–Z major group): **98.8% match** (strict A–Z over all 560,351; the annex's 99.4% puts the non-alpha `?` bucket in the denominator — same join, looser count). Three distinct shares, do not conflate: corpus religious **base rate ≈ 13%** (the positive class), NTEE **group X (Religion) = 7.27%** (one sector), religious-**lexicon hit rate ≈ 11.2%** (keyword prior).
- **Quality tiers** (computable score `Q`, max 6: length-capped + purpose verb + named beneficiary + named activity + specificity + multi-clause, minus boilerplate/vague/too-short penalties): **HIGH `Q≥5.0` = 31.9% (178,863)**, MEDIUM 38.6%, LOW 29.5%.
- **Content-quality filtering does not strip positives**: at `Q≥5.0`, religious 30.0% HIGH vs secular 31.9% — the 1.9pp gap is entirely bare-label churches (correctly excluded). Descriptive positives are slightly _longer_ than average.
- Per-stratum HIGH pools: all 26 groups ≥200 HIGH; **thin strata = V (257), Y (200), U (581)**; X still yields 6,679 HIGH positives.

---

## Architecture (point 1)

Move all legacy code to `src/legacy/` untouched (scripts + notebooks) — **quarantined, never executed**, so audit R-11's foreign absolute paths (Windows/macOS) are inert and deliberately left as-is. New code is a config-driven package with thin CLI entrypoints; each stage runs standalone given its input exists, and an orchestrator chains them.

```
src/binary_classifier/
  config.py            # typed config (pydantic-settings/dataclass) loaded from config/*.yaml; holds SEED, paths, entity, field, thresholds, model slate
  paths.py             # pathlib path registry (no string concatenation)
  data/
    load.py            # read cross-section parquet; EIN2 join to bmf_unified_processed.parquet -> NTEE major group
    quality.py         # the Q score + HIGH/MED/LOW tiering; religious-lexicon prior
    sample.py          # stratified + positive-enriched sampling; seeds; EIN2 + stratum + inclusion-probability manifest
  annotate/
    schema.py          # long/tidy label schema + pydantic parse of LLM JSON
    prompts/           # versioned codebook prompt templates (v1, v2, v3)
    annotators/
      base.py          # provider-agnostic Annotator interface (annotate(text)->LabelRecord)
      openai_annotator.py
      vllm_annotator.py    # open-weight via local OpenAI-compatible (vLLM) endpoint
    run_annotation.py  # batch labeling; resume by EIN2; full raw-response + metadata logging
    aggregate.py       # majority vote now; crowd-kit / cleanlab hooks
  qc/
    agreement.py       # LLM-vs-human agreement, Krippendorff alpha
  legacy/              # old scripts + notebooks, verbatim
scripts/               # thin standalone entrypoints
  01_build_sample.py  02_bakeoff_prompts.py  03_annotate.py  04_quality_check.py  run_pipeline.py
config/
  religious_missions.yaml   # the first task config (entity=missions, field=LONGEST_MISSION, label=religious, SEED, model slate, thresholds)
```

**Conventions (mandatory):** `uv run python …`; Ruff (line length 88); `pathlib.Path` everywhere; imports stdlib→third-party→local; Google/NumPy docstrings explaining business/economic intent; section-header comments above blocks, never inline (`.agents/architecture/conventions/`). Drop `bert-base-uncased`. Fix `pyproject.toml` (`python-dotenv` not `dotenv`), add `pyarrow`, `crowd-kit`, `cleanlab`, and the full **`vllm`** (B200-pinned, for self-hosted serving — not just a client, so `uv.lock` reproduces it); keep Python `>=3.13`. `requirements.txt` stays legacy (ignored).

---

## Point 2 — LLM annotation

### Stage 2.1 — Sample construction (`scripts/01_build_sample.py`)

1. Load `missions_cross_section.parquet`; join `EIN2 → bmf_unified_processed.parquet` for NTEE major group (`NTEE_IRS[0]`).
2. Compute `Q` and tier (`data/quality.py`); drop exact duplicates (keep one); flag truncated rows. _(The `Q` feature weights, tier thresholds, per-NTEE HIGH-pool sizes, and the positive-protective rescue rule are specified in the annex `.agents/plans/we-work-on-the-floofy-wreath-annex.md`.)_
3. **Silver pool (~20k):** sample from **HIGH + MEDIUM** (`Q≥3.0`; pool ≈395k — LOW excluded, it goes to the rule layer), **stratified across all 26 NTEE macro-groups** — proportional with a **floor** for thin strata (V/Y/U) and a **cap** on fat strata (B/P). **Positive-enrich per stratum** to ~30–40% via the religious-lexicon prior. (Including MEDIUM subsumes the earlier HIGH-only "positive-protective rescue" and matches the classifier's HIGH+MEDIUM validated scope.) Over-provisioned so the point-3 learning curve can find the plateau.
4. **Gold set (~400, ~15/group):** drawn from **HIGH + MEDIUM** (matching the silver scope) but **deliberately retain boundary cases** (saint-named-secular, spiritual-not-religious, generic ministry/mission, faith-heritage). **Persist each gold record's stratum + inclusion probability** for later reweighting to deployment prevalence (point 6).
5. **Human splits:** carve gold into **prompt-dev (~50)** / **validation** / **frozen test**. Persist an `EIN2` manifest for every split; all sampling seeded.

### Stage 2.1b — Human gold coding (the only human-in-the-loop step)

- The ~400 gold records are **hand-coded by the research engineer**, independent of any LLM output, using the codebook from `20260606-tech-religious-vs-nonreligious-mission-prompts.md`. A **subset is double-coded** by a second coder and disagreements adjudicated → **Krippendorff α** on that subset.
- Human labels enter the long/tidy store as `source_type='human'` (binary 0/1 + an `ambiguous`/`insufficient` flag for boundary cases). They are the references for: prompt selection on **prompt-dev**, the **≥85% agreement** gate on **validation**, and final metrics on the **frozen test**. Distinct from Verification step 6 (which only spot-reviews LLM output, not independent coding).

### Stage 2.2 — Prompt / codebook + bake-off (`scripts/02_bakeoff_prompts.py`)

- Codebook-style prompt (Halterman & Keith format; `20260606-tech-religious-vs-nonreligious-mission-prompts.md`): expert role; construct = **observable** religious/spiritual mission/expression (not latent religiosity); conservative default; domain codes; required **evidence spans**; **JSON output** `{binary_label ∈ religious|nonreligious|ambiguous_review|insufficient_information, confidence, domains_present, evidence_spans, boundary_notes}`; derived binary rule from domains. **Source of truth = `binary_label`**: `religious`→1, `nonreligious`→0; `ambiguous_review` and `insufficient_information` → **abstain (NaN)** in the tidy store (excluded from the silver training label, routed to human review). The domain-derived rule is a **consistency cross-check** on `binary_label`, not an independent label.
- Author **2–3 prompt variants**; run the **4-model bake-off** on **prompt-dev (~50)** only, scored against the **human prompt-dev labels** (Stage 2.1b). Select a **production model SET** (≥2 models — the open-weight backbone + an OpenAI snapshot, optionally a 2nd open model) and **retain 2–3 prompt variants**. Production runs the **full model × prompt matrix** → several labels per `EIN2` (cross-model **and** cross-prompt) for majority-vote + a disagreement/uncertainty signal. **Confirm the model-set × prompt-set on validation** before the full run (don't joint-select on 50 examples).

### Stage 2.3 — Labeling run (`scripts/03_annotate.py`)

- Run the **full model × prompt matrix** (each `(model, prompt)` = one `source_id`, several per `EIN2`) — OpenAI via API, open-weight via the in-job vLLM endpoint; **resume by `(EIN2, source_id)`** (not row count — fixes audit R-08); temp 0, fixed seed, JSON-schema-forced output.
- Write the **long/tidy label store**: one row per `(EIN2, source_id)` with `source_type ∈ {llm_prompt, rule, human}`, `label` (NaN = abstain, never a class), `confidence`, `model_id`, `prompt_id`, `temperature`, `seed`, `run_timestamp`, `raw_response`. This is the WS-ready hook (pivots into crowd-kit/cleanlab).
- Maintain a **canary set** re-run when model IDs change (drift detection for any closed model).

### Stage 2.4 — Aggregation + QC (`scripts/04_quality_check.py`)

- Aggregate the per-`EIN2` **model × prompt** labels by **majority vote** → silver training label (+ agreement as confidence); ties or all-abstain → route to human review, not a forced label. The model×prompt matrix is the natural input for **crowd-kit Dawid-Skene** (per-source reliability) and **cleanlab** — kept as drop-in comparison arms.
- **QC gate:** LLM-vs-human agreement on **validation ≥85%**; below → revise prompt and re-label. **Krippendorff α** on the double-coded gold subset. Once accepted, **version and freeze** the labelled file; do not re-label.

---

## Stubs (points 3–6) — architecture targets only

- **3. Training:** HF `Trainer`, **PR-AUC primary** (`metric_for_best_model`), early stopping, **seeds**, weighted+unweighted CE compared, **dynamic padding via `DataCollatorWithPadding` only** (fixes R-16). Grid (this **is** roadmap point-1's fine-tune-model decision — the _set_; the single production model is chosen empirically from the learning curve + eval): TF-IDF+LR and **MiniLM+LR** baselines; encoders **DeBERTa-v3-base / RoBERTa-base / DistilBERT**; small open-weight LLM (LoRA) as comparison arm. **Learning-curve sweep** {0.5k,1k,2k,4k,8k,16k} on the silver pool → plateau = optimal N. Early-stop/select on **human validation**, report on **frozen human test**. **Declared deviation:** roadmap asks best-val-**F1**; we make **PR-AUC** the selection metric for imbalance (F1 still computed) — supersedes R-06. **Log per-epoch train loss, val loss, val F1, and PR-AUC to a file**, not just the terminal (fixes R-03).
- **4. Evaluation:** PR-AUC + precision/recall/F1/MCC/balanced-acc + calibration (Brier/ECE/reliability) + subgroup error by NTEE group/length, all with **bootstrap CIs** (the ~200-row frozen test is small-N); acceptance criteria set before unlocking test.
- **5. Inference at scale:** full corpus, GPU-batched; **HIGH/MEDIUM → classifier; LOW/bare-label → rule layer (not dropped)**. Store predicted label **and** positive-class probability; keep `EIN2`; record model version + checkpoint hash + inference date as metadata columns.
- **6. Visualization / prevalence:** n-gram word clouds by predicted class; metric plots; **reweight gold sampling design → population prevalence with uncertainty.**

---

## Critical files

- New package: `src/binary_classifier/**`, `scripts/**`, `config/religious_missions.yaml`.
- Move to legacy: `generate_training_data.py`, `split_data.py`, `*.ipynb` → `src/legacy/`.
- Edit: `pyproject.toml` (deps + dotenv fix), `.gitignore` (fix malformed `my_model_prompt2` line), `README.md` (de-stale).
- Inputs (read-only, upstream): `…/NonProfitData/data/processed/corpus/missions/missions_cross_section.parquet`, `…/NonProfitData/data/processed/panel/core/bmf_unified_processed.parquet`.

---

## Implementation sprint (orchestrated)

The build is executed by an **orchestrator** that spawns **specialized sub-agents**, one per work package. The orchestrator does not write code itself; it dispatches, collects each sub-agent's report, integrates, runs the local verification gate, and resolves cross-agent contracts. Work is decoupled by directory so agents do not edit each other's files.

### Standard briefing (every sub-agent receives this in fresh context)

1. **Read the full plan** `.agents/plans/we-work-on-the-floofy-wreath.md` and the research annex `.agents/plans/we-work-on-the-floofy-wreath-annex.md` before doing anything.
2. **Follow the conventions** in `.agents/architecture/conventions/` (uv only; Ruff line-length 88; `pathlib.Path`; imports stdlib→third-party→local; Google/NumPy docstrings explaining intent; section-header comments above blocks, never inline).
3. **Scope discipline:** create/edit only the files in your package; treat every other path as read-only. Do not guess on ambiguous design — surface it to the orchestrator as a blocker rather than inventing.
4. **Reproducibility:** read `SEED`, paths, and the entity/field from `config/religious_missions.yaml`; carry `EIN2` through every artifact.
5. **Return the reporting contract** (below) as your final message.

### Reporting contract (every sub-agent returns this)

- **deliverables** — files created/modified (absolute paths).
- **interfaces** — the exact public functions/classes/columns/config keys downstream agents depend on (e.g. the long/tidy column list, the `Q`-tier function signature).
- **commands_run** — each command + pass/fail + key output (lint, config load, dry-runs).
- **assertions** — which plan-specified checks passed (e.g. `EIN2` unique per split; all 26 NTEE groups present).
- **deviations** — any departure from the plan/annex, with the reason.
- **blockers** — unresolved questions needing the orchestrator or user.
- **handoffs** — what the next agent in the dependency chain must know.

### Work packages (sub-agents) and dependency waves

| Agent                    | Package / scope                                                                                                                                                                             | Key deliverables                                                                                                                                                                                                                                                                                                                                                                                                                                    | Report-back contract highlights                                                                                                                        | Depends on             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| **S1 — scaffold**        | `src/binary_classifier/{config,paths}.py`, `config/religious_missions.yaml`, `pyproject.toml`, Ruff config, `.gitignore`, move legacy → `src/legacy/`                                       | Package skeleton; typed config (`SEED`, paths, entity=missions, field=`LONGEST_MISSION`, model slate, `Q` thresholds); `uv add` pyarrow/crowd-kit/cleanlab/openai/**vllm (B200-pinned)**/python-dotenv; drop `bert-base-uncased`; keep Py≥3.13; `uv.lock` refreshed                                                                                                                                                                                 | `uv run ruff check` clean; config loads; final `uv.lock`; any dependency-resolution conflicts under Py3.13                                             | — (Wave 0, blocks all) |
| **D1 — data + sampling** | `src/binary_classifier/data/{load,quality,sample}.py`, `scripts/01_build_sample.py`                                                                                                         | Cross-section load + `EIN2`→BMF NTEE-major-group join; **`Q` rubric exactly as in the annex** (feature weights, tier thresholds); silver from **HIGH+MEDIUM** (`Q≥3.0`), stratified+enriched (~20k) + gold (~400) + prompt-dev/val/test splits; seeded `EIN2`+stratum+inclusion-probability manifests; + **high-precision rule labels** in `quality.py` (`source_type='rule'`, doubling as the LOW/bare-label inference rule layer for point 5)     | Per-stratum allocation table; assertions (`EIN2` unique, all 26 groups, ~30–40% positive, byte-identical re-run); any deviation from annex pool counts | S1 (Wave 1)            |
| **A1 — annotation core** | `src/binary_classifier/annotate/{schema,prompts/,annotators/,run_annotation,aggregate}.py`, `scripts/03_annotate.py`, **`scripts/02_bakeoff_prompts.py`** (real bake-off harness)           | Long/tidy label schema + pydantic JSON parse; codebook prompt v1 + 2 variants (from `…-mission-prompts.md`); provider-agnostic `base`+`openai`+`vllm` annotators; batch run over the **model × prompt matrix** with **resume-by-`(EIN2, source_id)`** + full metadata logging; majority-vote aggregate + crowd-kit/cleanlab hooks; the **bake-off harness** (model-set × prompt scoring vs human prompt-dev labels) + the **canary-set** definition | Exact long/tidy column list; prompt file inventory; dry-run on ~5 missions (schema-valid JSON + resume works)                                          | S1 (Wave 1)            |
| **U1 — UCloud runtime**  | `init.sh`, **thin CLI entrypoint wrappers (argparse shells) for `scripts/01–04`** (stage logic owned by D1/A1/Q1; `run_pipeline.py` owned by the Orchestrator), `docs/RUNNING_ON_UCLOUD.md` | `/work`-aware `init.sh` (uv sync, env, optional vLLM); UCloud runbook (job submission, `gpu-nvidia-b200` SKU, SSH, `.env` secret, vLLM serve, Blackwell/CUDA check); the CLI wrappers (no stage logic)                                                                                                                                                                                                                                              | `init.sh` + runbook; each `scripts/0N --help` works; explicit list of UCloud caveats to confirm in-job                                                 | S1 (Wave 1)            |
| **Q1 — quality check**   | `src/binary_classifier/qc/agreement.py`, wire `scripts/04_quality_check.py`                                                                                                                 | LLM-vs-human agreement (%) + Krippendorff α; ≥85% gate; version+freeze of the labelled file                                                                                                                                                                                                                                                                                                                                                         | Functions + synthetic test; the freeze artifact format                                                                                                 | A1 (Wave 2)            |
| **Orchestrator**         | integration only                                                                                                                                                                            | Assemble `run_pipeline.py` 2.1→2.4; run the **local verification gate**; sprint report                                                                                                                                                                                                                                                                                                                                                              | Consolidated status; GPU stages deferred to manual UCloud run                                                                                          | all (Wave 3)           |

**Waves:** Wave 0 = S1. Wave 1 (parallel) = D1, A1, U1. Wave 2 = Q1. Wave 3 = orchestrator integration + local verification gate. GPU-dependent stages (prompt bake-off, full annotation) are **not** run by the agents — they are deferred to the manual UCloud session per the Verification section.

---

## UCloud runtime (init.sh & job session)

How the pipeline actually runs on the B200 node (from the UCloud docs — see [References](#references) for URLs). U1 implements `init.sh` + the runbook from this.

- **Job:** the **Terminal** app on machine type **`gpu-nvidia-b200`** (8× B200 192 GB, full GPUs — **not** the `-mig` fractional variant; needs a project grant allocation). Enable the **SSH server** at submission; connect via `ssh ucloud@ssh.cloud.sdu.dk -p <PORT>` (upload your public key once in Resources → SSH keys).
- **Persistence (critical):** only `/work` survives job termination. The **repo, input parquet data, outputs, the `.venv`, and uv caches** all live under a Drive mounted at `/work`. Redirect `UV_CACHE_DIR=/work/.uv-cache` and `UV_PYTHON_INSTALL_DIR=/work/.uv-python`.
- **Environment:** `uv` is preinstalled (0.11.3); system Python is 3.12 but **`uv` self-manages 3.13** (`uv python install 3.13` / `uv sync` against our pin) — no system Python change needed.
- **`init.sh`:** supplied as the optional **Initialization** parameter at job submission (a `.sh` file). It runs **at job start**, blocks until done, has **network + sudo**. Caveat: `export`ed vars may **not** propagate to the interactive shell — load secrets in the shell instead.
- **Secrets:** no native injection; keep a `.env` (with `OPENAI_API_KEY`) in a **private `/work` Drive**, `set -a; source .env; set +a` (repo already uses `python-dotenv`). Never commit `.env`.
- **vLLM annotator:** serve on **localhost inside the job** (`uv run vllm serve <model> --tensor-parallel-size 8 --port 8000`); the annotator calls `http://127.0.0.1:8000/v1` — no public link/port-forward needed.
- **Confirm in-job (docs don't state):** the **Blackwell B200 CUDA/driver version** vs. the vLLM/PyTorch build (genuine compatibility risk — run `nvidia-smi`, pin a B200-compatible vLLM); init.sh failure/idempotency semantics; interactive-job wall-time limit.

`init.sh` skeleton (U1 finalizes):

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="/work/BINARY-CLASSIFIER-MISSIONS"          # mounted Drive (or git clone here)
export UV_CACHE_DIR="/work/.uv-cache"
export UV_PYTHON_INSTALL_DIR="/work/.uv-python"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
cd "$REPO_DIR"
uv python install 3.13 || true
uv sync
# optional: pre-pull the open-weight annotator weights into /work/models for fast start
```

---

## Verification (points 1–2)

The GPU stages (annotation across the model slate, and later training) run on a **UCloud B200 terminal job** configured by an SSH startup script — they are **verified manually on UCloud**, not in local CI. Split accordingly:

**Local (CPU, dev machine) — automatable:**

1. `uv run ruff check src scripts` passes; `uv run python -c "from binary_classifier.config import load_config; load_config('config/religious_missions.yaml')"` loads.
2. `uv run python scripts/01_build_sample.py` on a fixed `SEED` (pandas/pyarrow joins + sampling, CPU-only) → silver (~20k) + gold (~400) manifests; assert: `EIN2` unique per split, every NTEE group represented, positive share ~30–40% in silver, gold carries stratum + inclusion-probability columns, re-running reproduces identical `EIN2` sets.

**UCloud B200 (GPU) — manual checks in the terminal job:** 3. Provision the job (repo + `uv sync` + model weights/keys via SSH startup script); smoke-test that the vLLM open-weight endpoint serves and returns **schema-valid JSON** for a handful of missions. 4. `scripts/02_bakeoff_prompts.py` on prompt-dev (~50) → manually inspect the per-model×prompt agreement table; **confirm the finalist on validation** before the full run. 5. `scripts/03_annotate.py --limit 200` → manually verify the long/tidy store carries every metadata column + `raw_response`; kill + rerun to confirm resume-by-`EIN2` (no duplicate labels). 6. `scripts/04_quality_check.py` → manually read validation agreement (**≥85% gate**) + Krippendorff α, and **eyeball a sample of labels + reasoning** before freezing. This manual review **is** the roadmap's single human-in-the-loop checkpoint; only on pass is the versioned, frozen labelled file written. 7. `scripts/run_pipeline.py` chains 2.1→2.4 on the configured entity (run end-to-end on UCloud once the stages pass individually).

---

## Sampling & quality research

The empirical basis for the sampling design — the **full mission-quality characterization** from sub-agent `aa42836beb4803b01` — is kept in the companion annex **`.agents/plans/we-work-on-the-floofy-wreath-annex.md`**: the quality rubric `Q` and feature weights, tier counts, the 26-row per-NTEE HIGH-pool table, the verbatim example bank (HIGH/MED/LOW), religious-vs-secular HIGH examples per sector, the five boundary-case patterns with prevalences, and the sampling synthesis. **§2.1 (sample construction) is implemented directly from that annex** — the `Q` rubric, per-stratum HIGH pools, the positive-protective rescue rule, and the concrete high-quality mission examples to sample. The corpus-shape/text-quality distributions and NTEE-join numbers in the [Corpus facts](#corpus-facts-that-ground-the-design-from-the-data-profile) section above come from the companion corpus-profile pass.

## References

### Internal research handoffs

- **Sampling & quality research (this project's own data):** `.agents/plans/we-work-on-the-floofy-wreath-annex.md` — the full mission-quality characterization (rubric `Q`, per-NTEE HIGH pools, example bank, boundary cases) from sub-agent `aa42836beb4803b01`; the source for the high-quality mission examples to sample in §2.1.
- `20260605-literature-synthesis-map.md` — overall empirical design (weak supervision as strongest design; annotation pipeline; model grid; evaluation bundle).
- `20260606-tech-synthesis-map.md` — technical takeaways, package shortlist, experiment scaffold, prompt/codebook domains.
- `20260606-tech-llm-weak-supervision-noisy-labels.md` — LLM-as-LF, label sources to compare, "add noisy-label methods only if they beat LLM-only on human-held-out", three human splits, drift/canary.
- `20260606-tech-religious-vs-nonreligious-mission-prompts.md` — codebook prompt skeleton, domains, derived binary rules, boundary cases, GivingTuesday precedent.
- `20260606-tech-short-text-model-alternatives.md` — encoder grid, HF Trainer recipe, hyperparameter sweep, PR-AUC selection.
- `20260606-tech-imbalanced-text-evaluation.md` / `20260606-tech-calibration-quantification-prevalence.md` — metric bundle + calibration/prevalence (points 4/6).
- Upstream data codebooks (in `NonProfitData`): `docs/codebooks/corpus-missions.md`, `docs/codebooks/corpus-activities.md`; BMF tidy `src/panel/01_tidy_unified_bmf.R`; `src/config.R` (`PATH_BMF_UNIFIED_PROCESSED`).

### External resources

**Annotation, weak supervision, noisy labels**

- Carlson & Burbano (2026), _SMJ_ — LLM annotation guidelines (low-temp, validation): https://doi.org/10.1002/smj.70023
- Baumann et al. (2025), "LLM Hacking" — https://arxiv.org/abs/2509.08825
- Pangakis & Wolken (2024), Knowledge Distillation in Automated Annotation — https://doi.org/10.18653/v1/2024.nlpcss-1.9
- Pangakis et al. (2023), Automated Annotation Requires Validation — https://doi.org/10.48550/arXiv.2306.00176
- Gilardi et al. (2023), ChatGPT Outperforms Crowd Workers — https://doi.org/10.1073/pnas.2305016120
- Smith et al. (2024), Language Models in the Loop — https://doi.org/10.1145/3617130
- Ratner et al. (2017), Snorkel (reference design only) — https://doi.org/10.14778/3157794.3157797
- Zhu et al. (2022), Is BERT Robust to Label Noise? — https://doi.org/10.18653/v1/2022.insights-1.8

**Prompt / codebook / domain construct**

- Halterman & Keith (2025), Codebook LLMs — _Political Analysis_
- Atreja et al. (2025), What's in a Prompt? — https://doi.org/10.1609/icwsm.v19i1.35807
- Organizational Religious Expression (ORE/IU) codebook — https://scholarworks.indianapolis.iu.edu/bitstreams/90959d19-3e24-457a-b205-ff6cf708e4c2/download
- Ressler/Fulton/Paxton religious dictionary (900 words) — https://www.pamelapaxton.com/religious-dictionary-holding
- GivingTuesday `religious_org_v1` (closest precedent) — https://huggingface.co/GivingTuesday/religious_org_v1 ; dataset: https://huggingface.co/datasets/GivingTuesday/religious_orgs_training

**Annotator models & serving**

- Qwen3-235B-A22B-Instruct — https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 ; Qwen3-32B — https://huggingface.co/Qwen/Qwen3-32B
- Gemma model cards — https://huggingface.co/google
- OpenAI Structured Outputs — https://platform.openai.com/docs/guides/structured-outputs
- vLLM (Blackwell/CUDA, guided JSON) — https://docs.vllm.ai/ ; SGLang — https://github.com/sgl-project/sglang ; xgrammar — https://github.com/mlc-ai/xgrammar

**Label-aggregation / noise tooling (Py3.13-ready)**

- cleanlab (v2.9, Jan 2026) — https://github.com/cleanlab/cleanlab
- crowd-kit (v1.4.2; Dawid-Skene/GLAD) — https://github.com/Toloka/crowd-kit
- Snorkel (dormant; reference only) — https://github.com/snorkel-team/snorkel

**Nonprofit-text precedent**

- Ma (2021), NVSQ — https://doi.org/10.1177/0899764020968153 ; code: https://github.com/ma-ji/npo_classifier
- Santamarina et al. (2023) — https://github.com/fjsantam/bespoke-npo-taxonomies
- Nonprofit Open Data Collective mission classifiers — https://nonprofit-open-data-collective.github.io/machine_learning_mission_codes/

**Data source**

- NCCS Unified BMF — https://nccs.urban.org/nccs/datasets/bmf/
- Upstream repo: `~/Documents/Projects/NonProfitData` (`data/processed/corpus/missions/missions_cross_section.parquet`, `data/processed/panel/core/bmf_unified_processed.parquet`)

**Runtime / UCloud (SDU / DeiC Interactive HPC)**

- `init.sh` mechanism — https://docs.cloud.sdu.dk/hands-on/init-sh.html ; conda/pip variants: https://docs.cloud.sdu.dk/hands-on/init-conda.html , https://docs.cloud.sdu.dk/hands-on/init-pip.html
- Submitting jobs (Initialization parameter, machine types, public links) — https://docs.cloud.sdu.dk/guide/submitting.html
- Terminal app (uv 0.11.3 / Python 3.12 / Lmod preinstalled) — https://docs.cloud.sdu.dk/Apps/terminal.html
- Machine SKUs (`gpu-nvidia-b200`) — https://docs.cloud.sdu.dk/guide/resources-products.html
- File management & `/work` persistence rule — https://docs.cloud.sdu.dk/guide/file-management.html
- SSH login — https://docs.cloud.sdu.dk/hands-on/ssh-login.html ; resource grants — https://docs.cloud.sdu.dk/guide/resources-intro.html ; monitoring — https://docs.cloud.sdu.dk/guide/monitoring.html

## Housekeeping notes for the user

- Optional upstream ask: lift the ~1000-char `LONGEST_MISSION` truncation in `NonProfitData` (affects ~0.75% of rows).
