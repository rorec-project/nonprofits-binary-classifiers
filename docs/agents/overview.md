# Project Overview

Config-driven binary text classifier that labels short US-nonprofit records as religious (`1`) vs non-religious (`0`). It is **entity-agnostic**: the religious × missions task is the first of several planned (activities, pregnancy centers, education, …), selected by config — never hard-coded.

## Approach — LLM-as-primary weak supervision

A model × prompt ensemble (a closed-API reference model + open-weight models served via vLLM) labels a large silver pool; the labels are aggregated into silver labels. A small hand-coded **gold** set drives prompt selection, validation, and a frozen test.

## Status

Stages 01–10 are built and wired into the orchestrator behind four human gates. Stage 11 (aggregation comparison) remains a standalone script. The released inference surface now distinguishes between the deduplicated scoring artifact (`predictions.parquet`) and the per-organization expand-back artifact (`predictions_full.parquet`). The legacy flat-script + notebook pipeline is preserved in `archive/legacy-pipe/` and is **not executed**.

## Related

- [pipeline/pipeline.md](pipeline/pipeline.md) — detailed stage map
- [pipeline/configuration.md](pipeline/configuration.md) — config-driven design
- [pipeline/human-gates.md](pipeline/human-gates.md) — G1–G4 checkpoints
- [operations/gotchas.md](operations/gotchas.md) — situational notes
- [../nontechnical-overview.md](../nontechnical-overview.md) — plain-language summary for non-technical readers
- [README](../../README.md) — full narrative
