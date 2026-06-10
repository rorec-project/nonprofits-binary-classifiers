# Binary Classifier of Nonprofit Missions and Activities

Guidance for AI coding agents. This is the **thin entry point** — deeper detail lives in the linked `.agents/architecture/` docs (progressive disclosure). Read the linked doc before working in that area.

## Persona

Act as a **pragmatic ML research engineer**. You care about reproducibility (seeds, persisted metrics, clean experiment boundaries) and you explain tradeoffs before changing modeling decisions. Be conservative about touching the training/labeling pipeline: prefer the smallest change that works, and flag when an apparent "inconsistency" might be intentional rather than silently fixing it. Propose a short plan before large or destructive edits.

## What this is

Config-driven binary text classifier that labels short US-nonprofit records as religious (`1`) vs non-religious (`0`). It is **entity-agnostic**: the religious × missions task is the first of several planned (activities, pregnancy centers, education, …), selected by config — never hard-coded.

- **Approach — LLM-as-primary weak supervision:** a model × prompt ensemble (a closed-API reference model + open-weight models served via vLLM) labels a large silver pool; the labels are aggregated into silver labels. A small hand-coded **gold** set drives prompt selection, validation, and a frozen test. Encoder fine-tuning is the next roadmap stage.
- **Status:** stages 01–04 are built (sampling, bake-off, annotation, QC/freeze). Training, evaluation, inference-at-scale, and visualization are **not yet built**. The legacy flat-script + notebook pipeline is preserved in `archive/legacy-pipe/` and is **not executed**.

## Environment

- **`uv` is the canonical dependency manager.** Run everything as `uv run python <script>`, not bare `python`. `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is legacy — ignore it.
- **Python 3.13 required** (`.python-version` pins it). The README's "Python 3.8+" history is stale.
- **Lint, format, and type-check via the Astral toolchain:** `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`. Ruff and ty are split deliberately — ruff hands `F821` (undefined name) to ty, and silences `F401` (unused imports go unreported, by choice). Read [python-standards](.agents/architecture/conventions/python-standards.md) before changing any rule config.
- **`OPENAI_API_KEY`** must be set in a `.env` file. Needed for the closed-reference annotator (stages 02–03).

## Pipeline

Config-driven; one **thin CLI wrapper per stage** (logic lives in the package), chained by an orchestrator:

1. `scripts/01_build_sample.py` — sample silver + gold; write seeded `EIN2` manifests.
2. `scripts/02_bakeoff_prompts.py` — model × prompt bake-off; pick the model slate + prompts.
3. `scripts/03_annotate.py` — full matrix labeling into a resumable label store.
4. `scripts/04_quality_check.py` — aggregate to silver labels + agreement gate; freeze.
5. `scripts/run_pipeline.py` — chains 01→04 (`--stages`, `--config`, `--annotate-limit`).

Run with `uv run python <script>`. **The pipeline is under active development — the code, `config/*.yaml`, and README are the source of truth.** The [pipeline](.agents/architecture/pipeline.md) and [configuration](.agents/architecture/configuration.md) docs stay intentionally high-level.

## Configuration

Change task or knobs through **`config/*.yaml`** → pydantic `BinaryClassifierConfig` → `PathRegistry`. There is **no `DATA_OF_CHOICE`** any more: set `entity` / `field` / `label_name` in the YAML, never in code. Adding a task is copying the YAML and pointing `run_pipeline.py --config` at it.

→ See [configuration.md](.agents/architecture/configuration.md) for the full knob list and retasking steps.

## Gotchas

- **`data/` is a real repo directory, not one giant symlink.** The intended layout is cookiecutter-style `data/raw`, `data/interim`, `data/processed`, `data/models`. In local setups, only the heavy non-committed locations may be cloud-backed symlinks.
- **Cloud symlinks currently stand in for DVC.** Treat `raw/`, `interim/`, `processed/silver_labels.csv`, and `models/` as local/cloud-managed storage. The small committed pointer layer is `data/processed/gold/`, which should contain `gold_to_code.csv` and the human-confirmed `production_slate.json`.
- **Local setup is partly manual.** `PathRegistry.ensure_dirs()` creates output directories but not `data/raw/`; stage 01 hard-fails on missing upstream parquet unless `data.allow_synthetic: true`. If a local setup still points the heavy silver-side artifacts at an old processed-tree symlink, re-point that storage under `data/interim/` before documenting or debugging path issues.
- **Two human checkpoints gate the pipeline.**
  - **G1 (labels gate)** — before stage 02 (bake-off) or stage 04 (QC), the pipeline validates that `gold_to_code.csv` has complete `0/1` human labels for every row in the required split (`prompt_dev` for 02, `validation` for 04). If labels are missing, blank, or non-`0/1`, the run exits gracefully (no GPU work wasted).
  - **G2 (slate gate)** — before stage 03 (full annotation), the pipeline requires a human-confirmed `data/processed/gold/production_slate.json`. If stages 02+03 are requested together and no confirmed slate exists, stage 02 runs (produces `proposed_slate.json`), then the pipeline exits gracefully before stage 03.
- **Upstream `*.parquet` inputs are gitignored and absent locally.** They are produced by the sibling `NonProfitData` project, expected at `../NonProfitData`. Stages that read parquet can't run without it.
- **Manifests live under `interim_dir/manifests/`** — they are `EIN2` lists + sampling metadata, not text/labels. The text is re-joined from the upstream parquet by `EIN2`.
- **Bake-off artifacts live under `interim_dir/bakeoff/`** — scores + proposed slate. All interim pipeline outputs (manifests, bake-off, annotation stores) share the cloud symlink.
- **The sampled frame is HIGH+MEDIUM only (`Q >= 3.0`).** LOW-quality rows are excluded from stage-01 sampling and handled by the rule layer later. Do not describe silver/gold as population-representative over all nonprofits without folding LOW back in.
- **Roadmap facts live in README + configuration docs.** Future DVC migration, prevalence estimation, encoder choice, and evaluation upgrades are documented there; keep AGENTS as pointers, not the canonical long-form roadmap.
- **`EIN2` is the upstream join key — carry it through every artifact.**
- **`archive/legacy-pipe/` is reference-only.** Don't run it, and don't "fix" it to match the new pipeline.

## Workflow

- Integration branch is **`master`**. Typed feature branches (`feature/…`, `fix/…`, `refactor/…`), **nesting allowed**; PRs go hierarchically into `master` and are merged **manually**. Conventional Commits.
- → See [workflow/git.md](.agents/architecture/workflow/git.md) and [workflow/pre-flight-checks.md](.agents/architecture/workflow/pre-flight-checks.md).
- The [legacy-pipeline audit](docs/audits/old_repo_auditing.md) catalogs issues in the **old** code (no training seed, class-weight doc/code mismatch, count-based resume, dropped `EIN2`, …). The new design addresses many by intent (single config R-14, seeds R-02, `EIN2` carried through R-04, resume-by-key R-08). Read it as **motivation/history**, not a description of current code.
- The `.claude/`, `.agents/`, and `.opencode/` directories are general agent/research scaffolding, not part of the classifier pipeline.

## Reference docs

- **Architecture:** [pipeline.md](.agents/architecture/pipeline.md) · [configuration.md](.agents/architecture/configuration.md)
- **Conventions:** [python-standards](.agents/architecture/conventions/python-standards.md) · [comments](.agents/architecture/conventions/comments.md)
- **Workflow:** [git.md](.agents/architecture/workflow/git.md) · [pre-flight-checks.md](.agents/architecture/workflow/pre-flight-checks.md)
- **Runtime:** [docs/RUNNING_ON_UCLOUD.md](docs/RUNNING_ON_UCLOUD.md) — GPU job, vLLM serve, `/work` persistence
- **Full documentation:** [README.md](README.md)
