# Plan: Close the validation loop + harden the pipeline (sprint, orchestrator + subagents)

## Context

The 2026-06-09 review (`docs/audits/20260609-pipeline-review-best-practices.md`) found the
built pipeline (stages 01–04) runs end-to-end but its three QC mechanisms are silently
inert: bake-off scoring, prompt provenance, and the agreement gate — so it freezes
**unvalidated** silver labels. Root cause: an open human-in-the-loop validation loop (no
human-coding artifact; nothing refuses to proceed without one).

This plan closes that loop and fixes the high/medium defects with surgical changes, executed
as a **sprint by an orchestrating agent that dispatches subagents**.

**Target workflow — two human checkpoints, GPU work only on UCloud:**

```
LOCAL (CPU):  01 build_sample → gold_to_code.csv
              └─[HUMAN 1: code human_label 0/1]─┘
UCLOUD (GPU): 02 bake-off (OpenAI tiers + Gemma arm) → bakeoff_results.json + proposed_slate.json
              └─[HUMAN 2: review scores → confirm production_slate.json]─┘
UCLOUD (GPU): 03 annotate (confirmed slate) → 04 QC gate (validation labels) → freeze
```

## Locked decisions

1. **`human_label` is strict `{0,1}`** (no human abstain). Preflight rejects blanks/other.
2. **Two graceful human gates.** `run_pipeline.py` accepts all stages and inserts two
   checkpoints that fail gracefully (clear message, non-zero exit, no wasted GPU):
   **(G1) labels gate** — before stage 02, validate human `0/1` labels for every requested
   label-dependent stage (`prompt_dev`→02, `validation`→04); **(G2) slate gate** — before
   stage 03, require a human-**confirmed** `production_slate.json`. If 02+03 run in one
   invocation with no confirmed slate, the run executes 02 then stops gracefully before 03.
   Stages 02/03/04 also raise internally as a backstop.
3. **Single in-place template** `gold_to_code.csv` (`EIN2, split, text, human_label`-blank).
4. **Clobber protection / idempotency:** stage 01 **skips regeneration if the template
   exists** (logs "N coded, skipping"); `--force` is the only path that overwrites.
5. **gold committed / silver in cloud**, via config-driven `gold_dir` / `silver_dir`.
6. **Closed-only start, ensemble opt-in later.** The slate is a **config-driven list of
   candidates** (each tagged `provider`). Bake-off candidates: `gpt-4o-mini`, `gpt-5-mini`,
   `gpt-5-nano` (pinned snapshots; `reasoning_effort: minimal` for GPT-5) **plus one
   open-weight comparison arm** (best available Gemma, `provider: vllm`). Production default
   `gpt-5-mini`, but **stage 03 only runs what the human confirms** (G2). The open-weight
   arm needs the vLLM server up for stage 02; comment it out for a pure-OpenAI bake-off.
   Adopting the open model for production is then a review choice, not a code change.
7. **Two phases → two nested branches** off `refactor/harmonize-pipeline`:
   `fix/close-validation-loop` (PR1), then `fix/pipeline-hardening` (PR2, branched from PR1).
8. **README user guide** (both checkpoints + the ensemble flip) is the final sprint task.

### Defaulted (flagged, no objection raised)
`pytest` added to dev deps; bootstrap CI = 1000 resamples / 95%; the ≥85% gate is measured on
agreement (Phase 2 *reports* the fuller bundle alongside); G1 requires only
`prompt_dev`+`validation` coding for 02/04; `gold_dir`/`silver_dir` default to
`data/processed/train_test_datasets/{gold,silver}`.

## Out of scope (user handles manually)
Physical folder restructuring (`data/processed/train_test_datasets/{gold,silver}` + the
`silver`→cloud symlink). The plan only adds config keys + registry properties; the user sets
YAML values to match. **Exact pinned snapshot IDs** for the three OpenAI models **and the
exact Gemma model ID** (must fit the B200 node + `/work` storage) are confirmed by the user
during T1.0 (placeholders until then).

---

## Orchestration model

**Subagent contract (every dispatch).** Each subagent receives, in fresh context: (a) this
full plan file, (b) its single task block, (c) this contract. It must:
- Edit **only the files in its `Owns` list**. If it needs a change elsewhere, stop and
  report a blocker — do not edit outside the set.
- Implement the `Operations`; satisfy every `Acceptance` item.
- Write/extend its own `tests/test_*.py`; run `uv run pytest <its test file>`,
  `uv run ruff check <owned files>`, `uv run ty check`.
- **Report back** (the orchestrator's contract): `{task_id, files_changed[],
  summary, acceptance: [{criterion, pass|fail, evidence}], pytest: <output>,
  ruff_ty: ok|errors, deviations[], blockers[]}`.

**Protocol.** The orchestrator runs tasks in **waves** by dependency. Within a wave, tasks
have **disjoint `Owns` sets** → safe to run in parallel. Between waves the orchestrator runs
the **full** `ruff` + `ty` + `pytest` suite, confirms each subagent's acceptance report, and
only then launches the next wave. A failed acceptance item blocks dependents. PR1 must be
green and merged before PR2's branch is cut.

---

## Phase 1 — `fix/close-validation-loop` (PR1)

| ID | Title | Owns | Deps |
|----|-------|------|------|
| T1.0 | Config/registry/factory foundation | `config.py`, `config/religious_missions.yaml`, `paths.py`, `annotate/annotators/factory.py` (new), `pyproject.toml`, `tests/conftest.py` | — |
| T1.1 | Stage 01 template emit + clobber | `data/sample.py`, `scripts/01_build_sample.py`, `tests/test_sample.py` | T1.0 |
| T1.2 | Synthetic opt-in | `data/load.py`, `tests/test_load.py` | T1.0 |
| T1.3 | Prompt_id provenance + confirmed-slate consume | `annotate/run_annotation.py`, `tests/test_run_annotation.py` | T1.0 |
| T1.4 | Preflight: G1 labels + G2 slate | `qc/preflight.py` (new), `tests/test_preflight.py` | T1.0 |
| T1.6 | Bake-off scoring + proposed_slate (incl. Gemma arm) | `annotate/bakeoff_prompts.py`, `tests/test_bakeoff.py` | T1.0 |
| T1.7 | QC gate blocks + reads validation labels | `qc/agreement.py`, `scripts/04_quality_check.py`, `tests/test_agreement.py` | T1.0 |
| T1.5 | Orchestrator wiring (both gates) | `scripts/run_pipeline.py`, `tests/test_run_pipeline.py` | T1.3, T1.4, T1.6 |

Waves: **W0** = T1.0 → **W1** = {T1.1, T1.2, T1.3, T1.4, T1.6, T1.7} (parallel) → **W2** = T1.5.

### T1.0 — Foundation
- **Operations:** (a) `config.py`+YAML: replace `ModelSlateConfig` with a config-driven list:
  `model_slate.bakeoff_candidates: [{id, provider, reasoning_effort?}]`,
  `model_slate.production: <id>`. Seed candidates with the three OpenAI models (user confirms
  snapshots) **and one `provider: vllm` Gemma arm** (user confirms ID). Add `paths.gold_dir`,
  `paths.silver_dir`; `qc.agreement_threshold: 0.85`; `data.allow_synthetic: false`.
  (b) `paths.py`: add properties `gold_dir`, `silver_dir`, `gold_coding_template`
  (`gold_dir/"gold_to_code.csv"`), `proposed_slate` (`results_dir/"proposed_slate.json"`),
  `production_slate` (`gold_dir/"production_slate.json"` — committed, human-confirmed),
  `bakeoff_results` (`results_dir/"bakeoff_results.json"`), `annotation_store`
  (`silver_dir/"annotation_store.csv"`), `bakeoff_store` (`silver_dir/"bakeoff_labels.csv"`);
  extend `ensure_dirs`. (c) New `annotators/factory.py::make_annotator(cfg, spec, prompt_id,
  prompt_text)` routing by `spec.provider` (`openai`→`OpenAIAnnotator`, `vllm`→`VLLMAnnotator`),
  passing `reasoning_effort` only when present. (d) `pyproject.toml`: add `pytest` to dev group;
  minimal `tests/conftest.py` with a tiny synthetic-config fixture.
- **Acceptance:** config loads + validates with the new slate-list schema; `PathRegistry`
  exposes every new property as a `Path`; `make_annotator` returns the correct class per
  provider and only forwards `reasoning_effort` when set; import smoke passes; ruff/ty clean.

### T1.1 — Stage 01 template + clobber
- **Operations:** in `build_sample`, after `split_human_sets`, write
  `registry.gold_coding_template` from `gold_all` with columns `EIN2, split, text,
  human_label`(empty). **Skip if the file exists** unless `force=True` (log "gold present, N
  coded — skipping"). `scripts/01`: add `--force`; pass to `build_sample`. Manifests unchanged.
- **Acceptance:** fresh run writes the 4-column template with blank `human_label`; re-run
  without `--force` does not overwrite (asserted via a sentinel coded value); `--force`
  regenerates; existing stage-01 assertions still pass.

### T1.2 — Synthetic opt-in
- **Operations:** `load_missions` reads `cfg.data.allow_synthetic`. Missing parquet + not
  allowed → `raise FileNotFoundError(expected_path + remediation)`. Allowed → `logger.warning`,
  add `data_source="synthetic"`, register `atexit` temp-dir cleanup.
- **Acceptance:** missing parquet + `false` raises with a clear path; `true` warns, returns
  data stamped `synthetic`, cleans the temp dir at exit.

### T1.3 — Prompt_id provenance + confirmed-slate consume
- **Operations:** rebuild `run_annotation`'s factory to use `factory.make_annotator(cfg, spec,
  prompt_id, prompt_text)` so `prompt_id` is the real stem → `source_id =
  f"{model_id}__{prompt_id}"` and the resume key matches. **Production model set =
  `registry.production_slate`** (the human-confirmed file); store path =
  `registry.annotation_store`.
- **Acceptance:** annotating one row under v1/v2/v3 yields 3 distinct `source_id`s; a resume
  re-run skips done `(EIN2, source_id)` pairs (no duplicates); the production set is exactly
  what `production_slate.json` lists.

### T1.4 — Preflight: G1 labels + G2 slate
- **Operations:** new `qc/preflight.py::validate_gates(cfg, registry, stages) -> list[str]`.
  **G1 (labels):** for each split a requested stage needs (`prompt_dev` if `"02"`,
  `validation` if `"04"`), read `gold_coding_template` and flag missing file/columns or any
  blank/`NaN`/non-`{0,1}` `human_label`, or `EIN2` mismatch with the gold manifest.
  **G2 (slate):** if `"03"` is requested, require `production_slate.json` to exist and be
  valid (≥1 entry, ids resolvable to configured candidates). Returns all problems.
- **Acceptance:** complete 0/1 coding → no G1 problems; one blank `validation` row (with `"04"`)
  → a counted problem; a `2`/`abstain` → a problem; `"03"` requested without
  `production_slate.json` → a G2 problem; `["01"]` only → `[]`.

### T1.6 — Bake-off scoring + proposed_slate (incl. Gemma arm)
- **Operations:** `run_bakeoff` defaults human labels to `gold_coding_template`
  (`split==prompt_dev`); **raise** if absent. Score each `bakeoff_candidates × prompt`
  (accuracy/precision/recall/F1 + abstain rate — full bundle is Phase 2) against human 0/1;
  route via `factory.make_annotator` (so the `vllm` Gemma arm scores when its server is up; a
  vLLM connection error is reported per-candidate, not fatal). Write `registry.bakeoff_results`
  (all scores) and `registry.proposed_slate` (auto-pick of candidates×prompts clearing
  `agreement_threshold`, else the single best) — **with `"confirmed": false`**. Do **not**
  write `production_slate.json` (that is the human's G2 step). Use registry store/results paths.
- **Acceptance:** with a coded prompt_dev fixture, emits non-empty scores + a `proposed_slate`
  marked unconfirmed; missing labels → raises (not a silent `note`); a stubbed vLLM failure
  degrades that one arm without aborting the OpenAI arms.

### T1.7 — QC gate blocks + reads validation labels
- **Operations:** `run_quality_check` defaults validation labels to `gold_coding_template`
  (`split==validation`); compute agreement vs aggregated silver; **freeze only if
  `agreement >= cfg.qc.agreement_threshold`**, else raise/exit non-zero with the "revise
  prompts and re-label" guidance and **write nothing**. Fix the docstring's false
  "defaults to the validation manifest" claim. Use `registry.annotation_store`. (Metric bundle
  + removing "Krippendorff alpha" wording is T2.1.)
- **Acceptance:** agreement ≥ threshold → frozen file written; below → raises, no file;
  absent labels → raises.

### T1.5 — Orchestrator wiring (both gates)
- **Operations:** `run_pipeline` runs requested stages in order. **Before 02:** call
  `validate_gates(..., requested)` for G1; on problems print + `sys.exit(2)` (graceful,
  no model touched). **Before 03:** re-check G2; if no confirmed `production_slate.json`
  (e.g. 02 just produced only a *proposed* one), print "review results/bakeoff_results.json,
  confirm production_slate.json, then run --stages 03,04" and `sys.exit(2)` — **after** 02 has
  run, so no annotation work is wasted. Pass `--force` to stage 01. Keep `--annotate-limit`.
- **Acceptance:** `--stages 01,02,03,04` with no coded labels → runs 01, prints G1 problems,
  exits 2, touches no GPU/model; with labels coded but no confirmed slate → runs 01+02, exits
  2 at G2 before 03; with labels coded + confirmed slate → full run proceeds.

---

## Phase 2 — `fix/pipeline-hardening` (PR2, off PR1)

| ID | Title | Owns | Deps |
|----|-------|------|------|
| T2.A | Annotation hardening (schema+annotators) | `annotate/schema.py`, `annotate/annotators/openai_annotator.py`, `annotate/annotators/vllm_annotator.py`, `annotate/run_annotation.py`, `tests/test_schema.py` | PR1 |
| T2.1 | QC metric bundle + CIs | `qc/agreement.py`, `scripts/04_quality_check.py`, `tests/test_metrics.py` | PR1 |
| T2.5 | Registry path migration (remainder) | `annotate/bakeoff_prompts.py`, `paths.py` | T2.A |
| T2.6 | Evidence-span guard | `qc/evidence.py` (new), `scripts/04_quality_check.py` | T2.1 |
| T2.7 | Docs (architecture) | `AGENTS.md`, `docs/agents/configuration.md` | T2.A,T2.1 |
| T2.8 | README user guide | `README.md` | all |

Waves: **WA** = {T2.A, T2.1} (disjoint) → **WB** = {T2.5, T2.6} → **WC** = {T2.7} → **WD** = {T2.8}.

- **T2.A (D5,D6,D8):** move `build_json_schema()` into `schema.py`, strict-compatible (all
  fields in `required`, nullable types, `additionalProperties:false`); OpenAI annotator uses
  `response_format={"type":"json_schema","json_schema":{...,"strict":true}}` (forward
  `reasoning_effort` per spec); vLLM imports the same schema for `guided_json`; remove the dead
  duplicate. **Append-only `AnnotationStore`** + vectorised resume set (`set(zip(...))`). Add
  optional `system_fingerprint` field; capture from the OpenAI response. Soften "deterministic"
  docstrings to "best-effort". *Acceptance:* OpenAI path emits schema-valid JSON; store append
  round-trips without full rewrite; fingerprint persisted; resume O(n).
- **T2.1 (D4):** replace raw % with sklearn bundle (confusion matrix, minority P/R/F1, MCC,
  balanced accuracy, PR-AUC when scores exist) + bootstrap CI (1000/95%); add Cohen's κ *with
  the prevalence caveat*; **remove all "Krippendorff alpha" wording** in the module docstring +
  stage-04 help. Gate still triggers on agreement vs threshold.
- **T2.5 (D7):** migrate remaining CWD-relative paths in `bakeoff_prompts.py` onto the registry.
  *Acceptance:* `grep -n 'Path("data\|Path("results' src/binary_classifier/annotate
  src/binary_classifier/qc` returns nothing.
- **T2.6 (D11):** new `qc/evidence.py` verifying each stored `evidence_span` is a verbatim
  substring of the source text (joined via `registry.missions_parquet`); flag fabricated-span
  records; config flag to abstain failed positives pre-aggregation; report the fabrication rate
  in the QC output; wire a call into stage 04. *Acceptance:* fabricated span flagged; real span
  passes; rate in the QC dict.
- **T2.7:** update `AGENTS.md` gotchas (gold-committed/silver-symlinked layout; two
  checkpoints) and `configuration.md` (new path/threshold/slate keys).
- **T2.8 — README guide:** end-to-end operator guide: (1) local `01` + coding (`gold_to_code.csv`,
  fill `human_label` 0/1, commit); (2) sync/pull to UCloud + serve the Gemma arm for the
  bake-off; (3) `--stages 02`, read `bakeoff_results.json`, **confirm `production_slate.json`**
  (HUMAN 2); (4) `--stages 03,04`; (5) enabling the open-weight model for *production* (confirm
  it in the slate) and the general ensemble flip; (6) `--force` + synthetic flags. *Acceptance:*
  a new operator can run the whole two-checkpoint loop from the README without reading source.

---

## Verification

- **Static (every wave):** `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`.
- **Unit (every wave):** `uv run pytest` — covers both gates + hardening without GPU/API keys.
- **Local smoke (no GPU), PR1:** set `allow_synthetic: true`; `uv run python
  scripts/run_pipeline.py --stages 01,02,03,04` → **G1 fails gracefully** (uncoded). Hand-fill
  `human_label` 0/1 → re-run → 01 skips regeneration, 02 runs (OpenAI arms; Gemma arm skipped
  if no server), then **G2 fails gracefully** (no confirmed slate). Copy/confirm
  `production_slate.json` → re-run `--stages 03,04` → proceeds.
- **End-to-end (UCloud), post-PR2:** with OpenAI models + Gemma served + coded gold, run
  `--stages 02`; confirm `bakeoff_results.json` scores all four candidates and writes a
  *proposed* slate; confirm `production_slate.json`; run `--stages 03,04`; verify distinct
  `source_id` per prompt, resume skips done work, stage 04 **blocks** on sub-threshold
  agreement and writes the metric bundle. Then confirm an open-weight model in the slate and
  re-run to prove the ensemble path needs no code change.

## Needs user input during implementation
- Exact pinned snapshot IDs for `gpt-4o-mini`, `gpt-5-mini`, `gpt-5-nano`, and the Gemma model
  ID (must fit the B200 node + `/work` storage) — T1.0.
- Confirm `gold_dir`/`silver_dir` YAML values match the manual folder layout.
