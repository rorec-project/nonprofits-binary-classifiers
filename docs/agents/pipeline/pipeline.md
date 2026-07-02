# Pipeline Overview

High-level map only. The pipeline is under **active development**, so this doc stays intentionally shallow — the **code, `config/*.yaml`, and the [README](../../../README.md) are the source of truth** for specifics (exact I/O, columns, thresholds). Don't grow a deep copy here.

## Shape

Reusable logic lives in the `src/binary_classifier/` package (`data/`, `annotate/`, `qc/`, `train/`, `evaluation/`, `inference/`, `prevalence/`, `viz/`). `scripts/01…10` are **thin CLI wrappers** that load the config + a `PathRegistry` and call a package function; `run_pipeline.py` chains them. Put logic in the package, not in `scripts/`.

## Stages

1. **01 build_sample** — sample a silver pool + a small gold set; write seeded `EIN2` manifests.
2. **02 bakeoff_prompts** — model × prompt bake-off on prompt-dev; pick the model slate + prompts.
3. **03 annotate** — full matrix labeling into a resumable long/tidy store.
4. **04 quality_check** — freeze majority-vote silver labels after the LLM-vs-human agreement gate.
5. **05 build_anchor** — full-frame anchor sample (incl. LOW) with design weights for prevalence.
6. **06 train** — baselines + DeBERTa/ModernBERT sweep, model selection, final-seed refit.
7. **07 evaluate** — cross-fit calibration, base-rate precision diagnostics, rule validation, and the one-shot frozen-test acceptance gate.
8. **08 infer** — full-corpus sharded inference with LOW-tier rule routing, triple labels, and expand-back from deduplicated text rows to raw-`EIN2` `predictions_full.parquet`.
9. **09 prevalence** — per-organization prevalence using HIGH/MEDIUM PPI, LOW classifier-routed PPI, LOW rule-only Rogan–Gladen, and raw-`EIN2` tier-share recombination.
10. **10 visualize** — figure rendering over evaluation, inference, and prevalence artifacts, including the new score-distribution, prevalence-decomposition, rule-validation, and subgroup plots.
11. **11 aggregation comparison** — script-only sensitivity diagnostics for alternative silver-label aggregation methods.

`uv run python scripts/run_pipeline.py` can orchestrate 01→10 (`--stages`, `--config`, `--annotate-limit`, `--infer-limit`, `--force`); each stage also runs standalone via its own script. Orchestrated runs now also emit `data/processed/run_manifest.json`. Stage 11 remains script-only.

## Status

Stages 01–10 are built and wired into the orchestrator behind four human gates ([G1–G4](human-gates.md)); stage 11 is a script-only helper over the artifacts the main pipeline produces. The legacy flat-script + notebook pipeline is parked in `archive/legacy-pipe/` and is **not executed**.

The real frozen-test artifact is still governed by one-shot semantics: smoke runs can exercise the enriched stage-07 schema locally, but the shared production `test_evaluation.json` is only finalized after the controlled post-sprint UCloud re-evaluation described in [human-gates.md](human-gates.md).

## Inputs

Cross-section parquets come from the sibling **`NonProfitData`** repo (expected at `../NonProfitData`); they are gitignored/absent locally, as are `data/` and `models/`. `EIN2` is the join key — keep it on every artifact.

## Related

- [configuration.md](configuration.md) — config-driven design + retasking
- [human-gates.md](human-gates.md) — G1–G4 checkpoint detail
- [../operations/gotchas.md](../operations/gotchas.md) — data layout, local setup, roadmaps
- [README](../../../README.md) — full narrative
- [docs/RUNNING_ON_UCLOUD.md](../../../docs/RUNNING_ON_UCLOUD.md) — GPU runtime
- [docs/audits/old_repo_auditing.md](../../../docs/audits/old_repo_auditing.md) — legacy-pipeline audit (history)
