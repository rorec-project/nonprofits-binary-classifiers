# Pipeline Overview

High-level map only. The pipeline is under **active development**, so this doc stays intentionally shallow — the **code, `config/*.yaml`, and the [README](../../README.md) are the source of truth** for specifics (exact I/O, columns, thresholds). Don't grow a deep copy here.

## Shape

Reusable logic lives in the `src/binary_classifier/` package (`data/`, `annotate/`, `qc/`). `scripts/01…04` are **thin CLI wrappers** that load the config + a `PathRegistry` and call a package function; `run_pipeline.py` chains them. Put logic in the package, not in `scripts/`.

## Stages

1. **01 build_sample** — sample a silver pool + a small gold set; write seeded `EIN2` manifests.
2. **02 bakeoff_prompts** — model × prompt bake-off on prompt-dev; pick the model slate + prompts.
3. **03 annotate** — full matrix labeling into a resumable long/tidy store.
4. **04 quality_check** — aggregate to silver labels + LLM-vs-human agreement gate; freeze.

`uv run python scripts/run_pipeline.py` runs 01→04 (`--stages`, `--config`, `--annotate-limit`); each stage also runs standalone via its own script. Stage 10 visualization is intentionally script-only (`uv run python scripts/10_visualize.py --config ...`) and is not wired into the orchestrator.

## Status

Stages 01–04 exist. Training, evaluation, and inference-at-scale are **roadmap** areas in this high-level note; stage 10 visualization exists as an optional renderer over artifacts from those stages and skips missing inputs. The legacy flat-script + notebook pipeline is parked in `archive/legacy-pipe/` and is **not executed**.

## Inputs

Cross-section parquets come from the sibling **`NonProfitData`** repo (expected at `../NonProfitData`); they are gitignored/absent locally, as are `data/` and `models/`. `EIN2` is the join key — keep it on every artifact.

## Related

- [configuration.md](configuration.md) — config-driven design + retasking
- [README](../../README.md) — full narrative
- [docs/RUNNING_ON_UCLOUD.md](../../docs/RUNNING_ON_UCLOUD.md) — GPU runtime
- [docs/audits/old_repo_auditing.md](../../docs/audits/old_repo_auditing.md) — legacy-pipeline audit (history)
