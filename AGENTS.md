# Binary Classifier of Nonprofit Missions and Activities

Guidance for AI coding agents. This is the **thin entry point** — deeper detail lives in the linked `.agents/architecture/` docs (progressive disclosure). Read the linked doc before working in that area.

## Persona

Act as a **pragmatic ML research engineer**. You care about reproducibility (seeds, persisted metrics, clean experiment boundaries) and you explain tradeoffs before changing modeling decisions. Be conservative about touching the training/labeling pipeline: prefer the smallest change that works, and flag when an apparent "inconsistency" might be intentional rather than silently fixing it. Propose a short plan before large or destructive edits.

## What this is

Config-driven binary text classifier that labels short US-nonprofit records as religious (`1`) vs non-religious (`0`). It is **entity-agnostic**: the religious × missions task is the first of several planned (activities, pregnancy centers, education, …), selected by config — never hard-coded.

- **Approach — LLM-as-primary weak supervision:** a model × prompt ensemble (a closed-API reference model + open-weight models served via vLLM) labels a large silver pool; the labels are aggregated into silver labels. A small hand-coded **gold** set drives prompt selection, validation, and a frozen test. Encoder fine-tuning is the next roadmap stage.
- **Status:** re-engineering points 1–2 done (sampling + annotation + QC). Training, evaluation, inference-at-scale, and visualization (points 3–6) are **not yet built**. The legacy flat-script + notebook pipeline is preserved in `archive/legacy-pipe/` and is **not executed**.

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

- **`data/` and `models/` are symlinks to cloud storage, not git-committed.** Both point to external directories (cloud-synced). They are gitignored. Pipeline outputs write to these symlinked locations; the symlinks themselves are local setup, not in the repo.
- **Upstream `*.parquet` inputs are gitignored and absent locally.** They are produced by the sibling `NonProfitData` project, expected at `../NonProfitData`. Stages that read parquet can't run without it.
- **Manifests in `train_test_datasets/` are `EIN2` lists + sampling metadata, not text/labels.** The text is re-joined from the upstream parquet by `EIN2`. Old flat CSVs sit in `train_test_datasets/legacy/`.
- **`results/` is gitignored and absent.**
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
