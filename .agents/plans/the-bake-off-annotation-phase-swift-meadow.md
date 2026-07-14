---
created: 2026-06-22
---

# Bake-off annotation phase — remediation sprint

> **Audience:** an orchestrator agent with **fresh context** that will dispatch subagents.
> This document is self-contained: §1 is shared context every subagent must receive; §2 is
> the task graph (what can run in parallel vs. sequentially); §3 is the task cards (one per
> dispatchable unit, each with a report-back contract); §4 is global integration verification.
> Nothing here assumes prior knowledge of the codebase beyond what an agent can grep/read.

---

## 0. How to use this document (orchestrator instructions)

1. **Read all of §1 (shared context).** Forward §1 verbatim to every subagent — it contains
   the diagnosis, repo map, run commands, and cross-cutting invariants each agent needs.
2. **Use §2 to schedule.** Tasks are labelled with `Depends on`, `Parallel-safe with`, and
   `Shared-file hotspots`. Launch independent tasks in parallel; sequence the ones that share
   a file or a contract.
3. **Dispatch each subagent with:** (a) all of §1, (b) the single task card from §3 it owns,
   (c) the **Report-back contract** (below). A subagent should NOT edit files outside its
   card's "Files & locations" without flagging it back.
4. **Standard report-back contract (every subagent returns this to the orchestrator):**
   - `task_id` and final `status`: `done` | `blocked` | `partial`.
   - `files_changed`: paths + one-line purpose each.
   - `acceptance`: each acceptance-criterion from the card marked pass/fail with evidence
     (test names + outcomes, command output snippets).
   - `deviations`: anything implemented differently from the card, and why.
   - `discoveries`: facts found that the plan didn't anticipate (new fields, broken tests,
     hidden callers) — especially anything affecting other tasks.
   - `handoff`: concrete inputs the next/dependent task needs (e.g. exact new field name,
     new function signature, palette constant names).
5. **No-information-loss rule:** the diagnosis (§1.4) is the *why*; do not let a subagent
   "simplify" away a documented behaviour (e.g. abstain semantics, the Slate consumer
   contract) without surfacing it.

---

## 1. Shared context (forward to every subagent)

### 1.1 Project orientation (one-screen primer)

This repo is a **config-driven binary text classifier** for US nonprofit missions
(religious `1` vs non-religious `0`). Pipeline stages: 01 sample/gold → **02 bake-off**
(score model×prompt arms on a human-coded prompt-dev set) → **03 annotate** (full
model×prompt matrix over the silver pool) → 04 QC/freeze silver → … → 10 visualize. An
ensemble of LLMs (OpenAI + an open-weight Gemma served via vLLM) produces weak-supervision
labels; a small human gold set validates them.

- Python project managed by **uv**. Package root: `src/binary_classifier/`.
- Config: `config/religious_missions.yaml` (the task config). A `PathRegistry`
  (`src/binary_classifier/paths.py`) resolves every artifact path from that config.
- Run a stage script: `uv run python scripts/NN_*.py --config config/religious_missions.yaml`.
- Run tests: `uv run pytest tests/ -k "<expr>"`.

### 1.2 Repo map (where things live)

| Area | Path | What it is |
|---|---|---|
| Label schema + store | `src/binary_classifier/annotate/schema.py` | `LabelRecord` pydantic model, `build_json_schema()`, `AnnotationStore` (CSV/Parquet read/write) |
| Annotators | `src/binary_classifier/annotate/annotators/{base,openai_annotator,vllm_annotator,factory}.py` | provider clients returning `LabelRecord` |
| Prompts | `src/binary_classifier/annotate/prompts/{v1,v2,v3}.txt` | system prompts; each ends with an "Output format" JSON block |
| Bake-off harness | `src/binary_classifier/annotate/bakeoff_prompts.py` | `run_bakeoff`, `_build_proposed_slate`, `_score_vs_human` |
| Stage-03 runner | `src/binary_classifier/annotate/run_annotation.py` | full-matrix runner, `_selected_prompt_pairs`, canary audit |
| Aggregation | `src/binary_classifier/annotate/aggregate.py` | majority vote / Dawid–Skene / CROWDLAB |
| Config models | `src/binary_classifier/config.py` | `BinaryClassifierConfig`, `BakeoffCandidate`, `Slate`, `load_slate` |
| Paths | `src/binary_classifier/paths.py` | `PathRegistry` properties (see §1.3) |
| Plotting helpers | `src/binary_classifier/viz/{curves,prevalence_plots,ngrams}.py` + `__init__.py` | pure `(data, ax) -> None` drawing functions |
| Figure orchestrator | `scripts/10_visualize.py` | `run_visualization`, skip-tolerant `_maybe_render_*`, `_save_plot` |
| Stage CLIs | `scripts/02_bakeoff_prompts.py`, `scripts/03_annotate.py`, `scripts/10_visualize.py` | thin wrappers |

**Where the bake-off OUTPUT data lives (already produced, local):** `data/interim` is a
**symlink** to a Cloud-Sync drive. The stage-02 outputs are:
- `data/interim/bakeoff/bakeoff_labels.csv` — long/tidy label store (`registry.bakeoff_store`)
- `data/interim/bakeoff/bakeoff_results.json` — per-arm score bundle (`registry.bakeoff_results`)
- `data/interim/bakeoff/proposed_slate.json` — unconfirmed slate (`registry.proposed_slate`)

There is **no stage-03 output** present locally (no `annotation_store.csv`, `silver_labels.csv`,
confirmed `production_slate.json`, or `canary_drift_audit.jsonl`).

### 1.3 Key `PathRegistry` properties (in `paths.py`)

`figures_dir` = `processed_dir/"figures"` (`data/processed/figures`) · `bakeoff_store` =
`bakeoff_dir/"bakeoff_labels.csv"` · `bakeoff_results` = `bakeoff_dir/"bakeoff_results.json"`
· `proposed_slate` = `bakeoff_dir/"proposed_slate.json"` · `production_slate` =
`gold_dir/"production_slate.json"` · `annotation_store` = `interim_dir/"annotation_store.csv"`
· `silver_labels` = `processed_dir/"silver_labels.csv"` · `prompts_dir` =
`src/binary_classifier/annotate/prompts` · canary audit = `interim_dir/"canary_drift_audit.jsonl"`.

### 1.4 The diagnosis (why this sprint exists — evidence-backed, do not discard)

The stage-02 bake-off ran **once** on **2026-06-21, 10:48–11:30 UTC** (timestamps in the
store), on current `master`-era code. The matrix is **complete and consistent**: 4 models ×
3 prompts × 50 prompt-dev rows = **600 rows**, 50 distinct EIN2, no gaps. Models:
`gpt-4o-mini-2024-07-18`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07` (provider `openai`),
`google/gemma-3-27b-it` (provider `vllm`). Prompts: `v1`, `v2`, `v3`. **Scoring and slate
selection succeeded** — they are not broken; they consume only the numeric `label`.

**Issue 1 — `reason` is an orphan field (schema ⇄ prompt divergence). CORE.**
- `build_json_schema()` + `LabelRecord` (`schema.py`) declare **6** required output fields:
  `binary_label, confidence, domains_present, evidence_spans, boundary_notes, reason`.
- All three prompts declare only **5** (no `reason`). `reason` has **never** appeared in any
  prompt (verified `git log -S`; it entered `schema.py` in old commits `0cc7ab0`/`008d35f`).
- Result — `reason` is null in **238/600** rows, provider-dependent: Gemma 150/150 null (key
  absent in raw JSON); gpt-4o-mini ~73/150 null (strict schema forces the *key* but the model
  emits `"reason": null` because the prompt never asks for content); gpt-5 mostly populated.
  `boundary_notes` is populated in **all 600 rows**.
- Severity: **not** a label-correctness bug. Verified `aggregate.py` consumes only
  `EIN2, source_id, label, confidence`; `reason`/`boundary_notes`/`domains_present`/
  `evidence_spans` are diagnostic only. It is a data-cleanliness / lost-diagnostic-value
  problem — but the user wants `reason` populated for human reviewers.

**Issue 2 — provider-inconsistent structured-output enforcement.**
- OpenAI arm uses strict `response_format={"type":"json_schema",…,"strict":true}` →
  all 6 keys always present. vLLM/Gemma stored output has only the 5 prompt fields and **no
  `reason` key**, despite `guided_json: true` in config and the factory forwarding it
  (`vllm_annotator.py` sends `extra_body={"guided_json": build_json_schema(), …}`). Because the
  schema makes `reason` required, a truly schema-constrained decode could not omit it →
  guided decoding was **not effectively enforced** for the vLLM arm in this run. (The markdown
  ```json fence on Gemma output is **not** proof — Gemma-3 fences even under guided decoding,
  handled by `_strip_code_fence`; the **missing required key** is the proof.)
- Likely root cause (needs **live** verification): the B200 vLLM server silently didn't honor
  xgrammar `guided_json` for Gemma-3, or a client/server wiring gap. Not resolvable from data.
- Severity: at stage-03 scale (~20k rows) an unconstrained Gemma arm will intermittently drift
  (prose, malformed JSON, bad enum) and silently abstain/error.

**Issue 3 — CSV double-JSON-encoding makes the store unreadable.**
- `LabelRecord.to_flat_dict()` runs free-text fields through `_csv_safe_text()` (=`json.dumps`)
  and `raw_response` is itself a JSON string; then pandas `to_csv` quotes again → cells like
  `""```json\n{\"binary_label\": \"nonreligious\", ...` (verified, user-confirmed unacceptable).
  The `json.dumps` layer is **redundant**: pandas already RFC-4180-quotes fields with commas/
  quotes/newlines and reads them back. Round-trips today only via `_restore_csv_safe_text`.

**Secondary flag (not in scope to *fix*, but surface in slate/figure):** high abstain rates
inflate apparent accuracy — `gpt-5-nano__v2` shows acc 1.0 at **44% abstain on n=28**.

### 1.5 SOTA figure standard (decided bar: venue-agnostic, applied to ALL pipeline figures)

Current figures are **plain matplotlib defaults**: DejaVu Sans, ad-hoc colors (`tab:blue`,
`0.65`), `grid(alpha=0.25)`, saved PNG@`dpi=200` + SVG. No `.mplstyle`, no `rcParams`, no
styling dependency (matplotlib only) anywhere. They are clean drafts, **not** publication-ready.

Target standard (from current best practice): design at final size (**column ≈ 3.5 in**,
**page ≈ 7.0 in**) with document-size text; **sans-serif** (Nature: Helvetica/Arial);
vector **PDF**/EPS for line art + raster **≥300 DPI**; deliberate **colorblind-safe** palette
(**Okabe–Ito**), simulator-testable; enforce via **one shared style sheet** (custom
`rcParams`/`.mplstyle`; `SciencePlots` is the alternative but needs LaTeX, so avoid).
Sources: [F. Schuch](https://www.fschuch.com/en/blog/2025/07/05/publication-quality-plots-in-python-with-matplotlib/),
[Nature/Science/Cell guidelines](https://conceptviz.app/blog/how-to-make-figures-for-nature-science-journals),
[SciencePlots](https://github.com/garrettj403/SciencePlots/wiki/Gallery),
[matplotlib_for_papers](https://github.com/jbmouret/matplotlib_for_papers).

### 1.6 Cross-cutting invariants & gotchas (all agents respect these)

- **Slate consumer contract:** `proposed_slate.json` is copied/edited to
  `production_slate.json` (human gate G2) and parsed by `_selected_prompt_pairs`
  (`run_annotation.py`) + validated by `load_slate`/`Slate` (`config.py`,
  `selected: list[dict[str, Any]]`). **Every `selected[]` row must keep `model_id` (or `id`)
  + `prompt_id`**, with `model_id` ∈ `models`, or stage 03 raises `ValueError`.
- **Lever split for `reason`:** `reason` is `string|null` in a `required` schema, so guided
  decoding/strict mode only guarantees the **key exists**, not non-null **content**. Only the
  **prompt** drives content. The acceptance gate for "reason works" is **non-null content**,
  not key-presence.
- **κ has no CI in the data:** `_score_vs_human` emits `bootstrap_ci` for `accuracy` and
  `minority_f1` only — **not** `cohens_kappa`. Plot κ as a point, never an interval.
- **Abstain semantics:** `binary_label ∈ {ambiguous_review, insufficient_information}` →
  numeric `label = NaN` (abstain). Don't "fix" abstains into 0/1.
- **No UCloud unless required:** WS that read existing artifacts run locally at zero cost. The
  only UCloud step is the bake-off re-run that needs `OPENAI_API_KEY` + a live vLLM server.
- **`data/interim` is a symlink** to a Cloud-Sync drive; treat files there as real artifacts.

### 1.7 Model-set & prompt-edit semantics (resume behavior — important)

How `run_bakeoff` (`bakeoff_prompts.py`) treats changes to the candidate set or prompts:

- **`bakeoff_labels.csv` (raw long store) is append-only and resume-safe**, keyed by
  `(EIN2, source_id)` where `source_id = model_id + "__" + prompt_id`. Re-running skips any
  pair already present (`store.already_done(...)`); only genuinely new pairs are annotated.
- **`bakeoff_results.json` and `proposed_slate.json` are fully overwritten each run, scoped to
  the CURRENT candidate list** in `config/religious_missions.yaml` (`results` is built only from
  the current `candidates × prompts`). Old models' scores are recomputed from their *cached*
  store rows (free); they are not re-annotated.
- **Adding a model** (e.g. `deepseek-ai/DeepSeek-V4-Flash`, `provider: vllm`): additive and
  cheap. Existing arms are reused from the store (no API/GPU cost); only the new model's arms are
  annotated and appended. `results.json`/slate/figure are rebuilt to show the new model
  **together with** the already-labelled ones — old labels are **not** replaced.
- **Removing a model:** it disappears from `results.json`/slate/figure (rebuilt over current
  candidates), but its rows **remain** in `bakeoff_labels.csv` (the store is never pruned). No
  built-in command removes a model's rows from the store.
- **THE GOTCHA — prompt edits are invisible to resume:** because `source_id` uses `prompt_id`
  (the file stem `v1`/`v2`/`v3`), **editing a prompt's text without changing its id is skipped on
  re-run** (the pair is "already done") → stale labels. This directly affects T1 (adding `reason`
  to v1/v2/v3 keeps the same ids). **Chosen handling (operator-manual, no code change): delete or
  rename `data/interim/bakeoff/bakeoff_labels.csv` before the T9 re-run** so all arms re-annotate
  fresh under the new prompts. (Bonus: a from-scratch run also writes the store directly in the
  WS2 clean-CSV format.)
- **Post-fix enhancement (T10) changes the overwrite behavior:** the *current* overwrite/scoped
  semantics above are the **pre-enhancement** state. T10 (Wave E) makes `results.json`/slate/
  figure an **additive full-store view** (scored over every `source_id` present in
  `bakeoff_labels.csv`), so models labelled across separate runs all appear together. After T10,
  removing a model from the YAML no longer drops it from the artifacts unless its store rows are
  pruned. Subagents touching `_build_proposed_slate`/scoring should read T10 before assuming the
  scoped behavior.

---

## 2. Task graph & parallelization

Eleven tasks. **T1–T9 are the bug-fix sprint**; **T10–T11 are post-fix enhancements** (Wave E,
optional, build on the fixes — the user asked for them "after we have fixed the bugs"). Group by
wave; within a wave, tasks touch disjoint files and may run in parallel.

```
Wave A (parallel, no deps):     T1 (prompts)   T2 (schema)   T5 (slate)   T6 (figure style)
Wave B (after T2):              T3 (annotators)   T4 (csv re-encoder)
Wave B' (after T6):             T7 (figure drawing fns)
Wave C (after T6 + T7):         T8 (figure integration in 10_visualize.py + __init__)
Wave D (after T1,T2,T3; UCloud):T9 (live vLLM check + stage-02 re-run + reason gate)
Wave E (post-fix enhancements): T10 (additive artifacts; after T5+T7) → T11 (sequential serving;
                                after T10; UCloud)
```

**Shared-file hotspots (sequence, don't parallelize these against each other):**
- `schema.py` → **T2 only** (T3/T4 read its result; do not also edit it).
- `scripts/10_visualize.py` → edited by **T6** (`_save_plot`) and **T8** (`run_visualization`
  + new render steps). Disjoint functions, but run **T6 before T8**.
- `viz/__init__.py` → **T8 only** (adds exports for T7's functions).
- `viz/bakeoff.py` (new) → authored by **T7**; T8 imports from it.

**Tests:** each task owns its own test updates/additions (in its acceptance criteria). §4 is
the final cross-task integration pass.

---

## 3. Task cards

### T1 — Prompts: add `reason` to v1/v2/v3
- **Depends on:** none. **Parallel-safe with:** all. **Shared-file hotspots:** none.
- **Objective:** make every model emit a non-null `reason`, and disambiguate it from
  `boundary_notes`, by fixing the prompt contract (the only lever for *content* — see §1.6).
- **Files & locations:** `src/binary_classifier/annotate/prompts/v1.txt`, `v2.txt`, `v3.txt`.
  Each ends with an "## Output format" section containing a fenced JSON object listing 5 keys
  (`binary_label, confidence, domains_present, evidence_spans, boundary_notes`) and a bullet
  list describing each.
- **Steps:**
  1. Add `"reason": "..."` to the JSON schema block in all three prompts.
  2. Add a bullet defining the two free-text fields with **distinct** roles:
     - `reason` — concise overall justification for the chosen `binary_label`; **always**
       populated.
     - `boundary_notes` — note flagging *edge-case difficulty* (saint name, spiritual-without-
       tradition, 501(c)(3) boilerplate, faith-heritage); brief/empty when the case is clear.
  3. Keep wording consistent with each prompt's existing voice (v1 verbose, v2/v3 terser).
- **Constraints:** do NOT remove or rename existing keys; order keys to match
  `build_json_schema()` for readability (not required by parsers).
- **Acceptance:** all three prompts list 6 keys incl. `reason`; the role split is explicit; a
  human reading the prompt understands `reason` ≠ `boundary_notes`.
- **Report-back:** the exact `reason` instruction text added (so T9 can sanity-check outputs).

### T2 — Schema: clean CSV serialization + dedicated `error` field
- **Depends on:** none. **Parallel-safe with:** T1, T5, T6. **Shared-file hotspots:**
  `schema.py` (this task is its sole editor).
- **Objective:** (a) eliminate the unreadable double-JSON-encoding (Issue 3); (b) stop error
  strings from masquerading as model reasoning by moving them out of `reason`.
- **Files & locations:** `src/binary_classifier/annotate/schema.py` only. Relevant members:
  `_csv_safe_text` (~L102), `_restore_csv_safe_text` (~L109), `LabelRecord` fields (~L122–196),
  `to_flat_dict` (~L223–250), `from_flat_dict` (~L252–296), `AnnotationStore.COLUMNS`
  (~L311–329), `_load`/`_save`/`append`/`append_many`.
- **Steps:**
  1. **Clean CSV:** in `to_flat_dict`, stop wrapping free-text fields (`reason`,
     `boundary_notes`, `raw_response`) in `_csv_safe_text`/`json.dumps`; store plain strings and
     let pandas `to_csv` (default RFC-4180 quoting) handle commas/quotes/newlines. In
     `from_flat_dict`, drop the matching `_restore_csv_safe_text` decode for those fields. Keep
     list fields (`domains_present`, `evidence_spans`) as compact JSON arrays (readable, e.g.
     `["faith_tradition"]`). Remove `_csv_safe_text`/`_restore_csv_safe_text` if no longer used.
  2. **`error` field:** add `error: str | None = None` to `LabelRecord` and to
     `AnnotationStore.COLUMNS` (append at end for backward-compatible reindexing); include it in
     `to_flat_dict`/`from_flat_dict`. This is the new home for failure messages (T3 fills it).
  3. Verify `_load`'s column-reindex still works for older CSVs missing the new column (it
     adds missing columns as `None`).
  4. Add a short docstring note documenting the now-plain CSV contract.
- **Constraints:** do NOT change the schema field set used by `build_json_schema()` (still the
  6 LLM-output keys; `error` is a store/record field, not an LLM-output field). Preserve
  abstain semantics and `compute_numeric_label`. `append`/`append_many` use `mode="a"` — ensure
  multi-line quoted cells still round-trip.
- **Acceptance (tests in `tests/`):**
  - Round-trip test: a `LabelRecord` whose `reason`/`boundary_notes`/`raw_response` contain
    quotes, commas, **and** newlines survives `AnnotationStore.append` → reload `to_frame`/
    `from_flat_dict` with equality; AND the raw CSV bytes contain **no** `\"`-style nested
    escaping (assert the cell is human-readable).
  - `error` field present in `COLUMNS`, round-trips, defaults to `None`.
  - Existing schema tests still pass (`uv run pytest tests/ -k "schema or store"`).
- **Report-back (handoff):** the exact new field name (`error`), final `COLUMNS` order, and
  confirmation of which fields are now plain-text vs JSON-array (T3, T4 depend on this).

### T3 — Annotators: conformance check + relocate error string
- **Depends on:** **T2** (needs the new `error` field). **Parallel-safe with:** T4 (disjoint
  files). **Shared-file hotspots:** none beyond the two annotator files.
- **Objective:** surface non-conformant LLM responses (e.g. Gemma silently dropping a required
  key) instead of nulling silently; route failure messages to `error`, not `reason`.
- **Files & locations:**
  `src/binary_classifier/annotate/annotators/openai_annotator.py` (`_parse_raw` ~L176–209,
  `_error_record` ~L211–225), and `…/vllm_annotator.py` (`_parse_raw` ~L153–186, `_error_record`
  ~L188–202, `_strip_code_fence` ~L33–40). Required-key set comes from `build_json_schema()`
  (`schema.py`).
- **Steps:**
  1. In both `_parse_raw`, after `json.loads`, **log a warning** (and set a flag/record marker)
     when the parsed object is missing any required schema key or has an out-of-enum
     `binary_label`. Keep current graceful behaviour (missing/abstain still produces a record),
     but make non-conformance visible. Do not raise.
  2. In both `_error_record`, write the failure message to the new **`error`** field (T2), not
     `reason`. Leave `reason=None` on error records.
  3. Keep `raw_response` preserved exactly as returned (don't pre-encode it — T2 makes the CSV
     layer handle it).
- **Constraints:** don't change request bodies, retry logic, or the guided_json wiring here
  (that's verified live in T9). Both files share the same change pattern — keep them symmetric.
- **Acceptance (tests):** a raw response missing `reason` (or with a bad enum) produces a
  record **and** emits the conformance warning/flag; an error path writes to `error` and leaves
  `reason=None`; `uv run pytest tests/ -k "annotat"` passes.
- **Report-back:** the conformance-flag mechanism (log only? record field?) so T9 knows what to
  look for in the re-run logs.

### T4 — One-off re-encoder for legacy CSV stores
- **Depends on:** **T2** (must produce the new clean format and know the old decoder).
  **Parallel-safe with:** T3. **Shared-file hotspots:** none (new file).
- **Objective:** convert any already-collected store written in the old double-encoded format
  to the new clean format, without re-running annotation.
- **Note on overlap with T9 (§1.7):** the UCloud re-run will **delete and regenerate** the
  stage-02 `bakeoff_labels.csv` fresh (already clean), so this utility's lasting value is for
  (a) getting a readable view of the *current* data **before** any re-run, and (b) any other/
  legacy or stage-03 store that exists in the old format. Keep it general, not stage-02-specific.
- **Files & locations:** new small utility, e.g. `scripts/reencode_label_store.py` (or a
  function in `schema.py`/a `utils/` module — pick the lightest, document choice). Operates on
  `registry.bakeoff_store` / `registry.annotation_store` or an explicit `--path`.
- **Steps:** load the CSV via the **old** decoding semantics (`_restore_csv_safe_text` logic for
  the legacy layer), then re-write via the new clean serializer (T2). Idempotent: re-running on
  an already-clean file is a no-op (detect/skip).
- **Constraints:** make a backup or write to a temp then atomically replace; never corrupt the
  Cloud-Sync symlinked file on partial failure.
- **Acceptance:** running it on the current `data/interim/bakeoff/bakeoff_labels.csv` yields a
  human-readable CSV that still round-trips through `AnnotationStore` with identical values;
  a second run is a no-op.
- **Report-back:** confirmation the existing 600-row store re-encodes cleanly + row/value parity.

### T5 — Restructure `proposed_slate.json` + local regen entrypoint
- **Depends on:** none (uses the existing `bakeoff_results.json`). **Parallel-safe with:** T1,
  T2, T6. **Shared-file hotspots:** `bakeoff_prompts.py` (sole editor).
- **Objective:** make the slate human-consultable for gate G2 (ranked summary + recommendation)
  AND regenerable locally from current results without re-annotating.
- **Files & locations:** `src/binary_classifier/annotate/bakeoff_prompts.py` —
  `_build_proposed_slate` (~L269–350; currently returns `{confirmed, models, selected,
  kappa_threshold, f1_ci_floor, agreement_threshold}` where `selected` holds full nested score
  dumps). `_score_vs_human` (~L371–459) documents available metrics: flat `accuracy/precision/
  recall/f1/abstain_rate/n_valid/n_total` + a `metrics` bundle with `cohens_kappa`,
  `bootstrap_ci.minority_f1.lower`, etc. `Slate` model in `config.py` (~L109).
- **Steps:**
  1. Replace the verbose per-arm dumps in `selected` with a **ranked summary table** — one row
     per (model, prompt) with: `model_id`, `prompt_id`, `accuracy`, `f1` (minority),
     `cohens_kappa`, `f1_ci_lower`, `abstain_rate`, `n_valid`, `clears` (bool). Sort by minority
     F1 (or κ). Add a top-level `recommended` block + a one-line rationale string.
  2. **Preserve the consumer contract (§1.6):** each `selected[]` row MUST retain `model_id`
     (or `id`) + `prompt_id`; `model_id` values must appear in `models`. Keep
     `confirmed`/`models`/thresholds keys.
  3. Add a thin **local entrypoint** (function + small CLI, e.g.
     `scripts/rebuild_slate.py` or a `--from-results` flag on `02_bakeoff_prompts.py`) that
     reads an existing `registry.bakeoff_results` JSON and rewrites `registry.proposed_slate` in
     the new format — **no annotation calls**. `_build_proposed_slate` already takes the
     `results` list, so this is a thin reader/writer.
  4. Keep the full detail in `bakeoff_results.json` untouched.
- **Constraints:** do not change selection logic semantics (κ≥threshold AND F1-CI≥floor, with
  single-best fallback); only restructure the output object. Surface `abstain_rate`/`n_valid`
  prominently (Secondary flag, §1.4).
- **Acceptance (tests):** `_build_proposed_slate` emits the new structure; a slate built from
  the current `bakeoff_results.json` still validates via `load_slate`/`Slate` and yields valid
  `(model_id, prompt_id)` pairs through `_selected_prompt_pairs`; local regen produces
  `proposed_slate.json` without network/UCloud. `uv run pytest tests/ -k "bakeoff or slate"`.
- **Report-back:** the new `selected[]` row schema (key names) — T8's figure may reuse it.

### T6 — Paper-ready figure style + `_save_plot` upgrade + retrofit
- **Depends on:** none. **Parallel-safe with:** T1, T2, T5. **Shared-file hotspots:**
  `scripts/10_visualize.py` (`_save_plot` only — must precede T8); `viz/curves.py`,
  `viz/prevalence_plots.py` (color retrofit).
- **Objective:** establish the §1.5 SOTA standard once, enforced for **all** figures via a
  shared style layer and the single `_save_plot` choke point.
- **Files & locations:**
  - new `src/binary_classifier/viz/style.py` + `src/binary_classifier/viz/paper.mplstyle`.
  - `scripts/10_visualize.py` `_save_plot` (~L295–324): currently `fig, ax =
    plt.subplots(figsize=...)`; `draw(ax)`; `tight_layout`; `savefig` PNG `dpi=200` + SVG,
    `bbox_inches="tight"`; `plt.close`. `matplotlib.use("Agg")` already set (top of file).
  - color retrofit in `viz/curves.py` (`reliability_diagram` uses `color="tab:blue"`,
    `"black"`) and `viz/prevalence_plots.py` (`"0.65"`, `"tab:blue"`).
- **Steps:**
  1. **Style module:** define the Okabe–Ito palette constants and a `paper.mplstyle` (or an
     `rcParams` dict + a `style_context()` helper) setting: sans-serif family with a Linux-safe
     fallback chain (Arial/Helvetica → Liberation Sans/Arimo → DejaVu Sans) to avoid
     font-substitution warnings; document-size fonts (~8–9 pt base, consistent label/tick/legend
     sizes); `axes.prop_cycle` = Okabe–Ito; despined top/right; light grid (`alpha≈0.25`);
     vector-friendly `pdf.fonttype=42`, `svg.fonttype="none"`. Expose `COLUMN_WIDTH=3.5`,
     `PAGE_WIDTH=7.0` (inches) helpers.
  2. **`_save_plot` upgrade:** wrap rendering in the style (`plt.style.context(...)` or rcParams)
     so it's uniform and side-effect-free; emit **PDF + SVG + PNG@300** (bump dpi 200→300),
     keep `bbox_inches="tight"`. Accept/standardize figsize via the width helpers (keep adaptive
     height for row-heavy figures).
  3. **Retrofit** the hardcoded colors in `curves.py`/`prevalence_plots.py` to palette
     references (preserve semantic intent — "suppressed" stays a muted grey from the palette).
- **Constraints:** keep `matplotlib.use("Agg")`. Don't alter the `(data, ax)` signatures of
  existing viz functions. Don't change `run_visualization` here (that's T8).
- **Acceptance (tests + manual):** `uv run python scripts/10_visualize.py` re-renders existing
  figures to **`.pdf` + `.svg` + `.png`@300** with no font-substitution warnings in the log;
  palette/despine/typography visibly consistent; a viz unit test asserts the style applies
  (e.g. prop_cycle colors, savefig produces 3 files). `uv run pytest tests/ -k "viz"`.
- **Report-back (handoff):** palette constant names + import path, the width-helper API, and
  the final `_save_plot` signature/output-file naming (T7, T8 depend on these).

### T7 — Figure drawing functions (bake-off + stage-03), pure `(data, ax)`
- **Depends on:** **T6** (palette + conventions; soft — can author in parallel and import once
  T6 lands). **Parallel-safe with:** T3, T4 (disjoint files). **Shared-file hotspots:** none
  (new file `viz/bakeoff.py`); does NOT edit `10_visualize.py` or `__init__.py` (that's T8).
- **Objective:** author the new pure drawing helpers; no integration wiring.
- **Files & locations:** new `src/binary_classifier/viz/bakeoff.py`, following the
  `(data, ax) -> None` convention in `viz/curves.py`. Inputs are the already-local artifacts
  (their schemas are in §1.4 / `_score_vs_human` / `aggregate.py` / `run_annotation.py`).
- **Steps — author these functions:**
  1. `bakeoff_summary(results, ax)` — from the `bakeoff_results.json` bundle (list of
     `{model_id, prompt_id, source_id, scores:{accuracy,f1,abstain_rate,n_valid,metrics:{...}}}`).
     Per-arm (12 rows) plot of **minority-F1 with bootstrap-CI error bars** (`metrics.bootstrap_ci
     .minority_f1.{lower,upper}`) + **Cohen's κ as a point** (`metrics.cohens_kappa`; **no CI** —
     §1.6), threshold reference lines at κ≥0.7 / F1-CI≥0.7, annotate `abstain_rate`/`n_valid`,
     sort by minority-F1, clear labels/legend. Make the high-abstain trap visible.
  2. `production_annotation_summary(df, ax)` — from `annotation_store.csv`/`silver_labels.csv`
     long/tidy frame (cols incl. `EIN2, source_id, label, …`): per-model abstain rate,
     inter-model agreement, vote-count/tie distribution.
  3. `canary_drift(rows, ax)` — from `canary_drift_audit.jsonl` records (κ/α change vs baseline
     over monitor runs; see `run_annotation._record_canary_audit`).
  4. Raise a clear `ValueError` on empty/missing-column input (matches existing viz helpers, so
     T8's skip-tolerant wrapper degrades cleanly).
- **Constraints:** pure functions only — no file I/O, no savefig, no rcParams mutation (styling
  comes from T6 via `_save_plot`). Use the T6 palette constants for any explicit colors.
- **Acceptance (tests):** each function renders into a provided `Axes` from a small synthetic
  fixture without error and raises on empty input; `bakeoff_summary` runs on the **real**
  `data/interim/bakeoff/bakeoff_results.json`. `uv run pytest tests/ -k "viz or bakeoff"`.
- **Report-back (handoff):** exact function names + signatures + expected input schema for each
  (T8 wires these and must match).

### T8 — Figure integration into `10_visualize.py` + `viz/__init__.py`
- **Depends on:** **T6** (`_save_plot`) **and T7** (drawing functions). **Parallel-safe with:**
  none in its wave (sole editor of the wiring). **Shared-file hotspots:** `scripts/10_visualize.py`
  (`run_visualization` + new `_maybe_render_*`), `viz/__init__.py` (exports).
- **Objective:** wire the new figures into the skip-tolerant orchestrator so they render from
  whatever artifacts exist, locally, no UCloud.
- **Files & locations:** `viz/__init__.py` (add exports for T7's functions); `scripts/10_visualize.py`
  — `run_visualization` (~L55–84 iterates a tuple of render steps) and the `_maybe_render_*`
  pattern (each checks `path.exists()`, logs a skip if missing, else calls `_save_plot`).
- **Steps:**
  1. Export `bakeoff_summary`, `production_annotation_summary`, `canary_drift` from
     `viz/__init__.py`.
  2. Add `_maybe_render_bakeoff_summary` (input `registry.bakeoff_results`),
     `_maybe_render_production_summary` (input `registry.annotation_store`/`silver_labels`),
     `_maybe_render_canary_drift` (input `interim_dir/"canary_drift_audit.jsonl"`). Each mirrors
     existing `_maybe_render_*`: exists-check → skip-with-log or `_save_plot(registry, name,
     lambda ax: fn(data, ax), figsize=<column/page width from T6>)`.
  3. Register the new steps in `run_visualization`'s render-step tuple.
- **Constraints:** preserve skip-tolerance (missing input = clean log skip, not error). Use T6
  width helpers for figsize. Don't modify `_save_plot` (T6 owns it).
- **Acceptance:** `uv run python scripts/10_visualize.py` produces
  `bakeoff_summary.{pdf,svg,png}` from the **existing** `bakeoff_results.json` and logs clean
  skips for the absent stage-03 inputs; all other figures still render. `uv run pytest tests/
  -k "visualize or viz"`.
- **Report-back:** list of figure base-names emitted + which inputs were present vs skipped.

### T9 — UCloud: live vLLM guided-decode check + stage-02 re-run + `reason` gate
- **Depends on:** **T1, T2, T3** landed (prompts populate `reason`; schema/annotator changes in
  effect). **Parallel-safe with:** none (operational, on UCloud). **Shared-file hotspots:** none.
- **Objective:** settle Issue 2's root cause on a live server and regenerate clean stage-02
  data with `reason` populated across all arms.
- **Environment:** UCloud job with `OPENAI_API_KEY` set and a vLLM server up
  (`utils/serve-llm.sh annotate`). The earlier *local* re-run failed precisely for lack of these
  (no OpenAI creds, no vLLM server).
- **Steps:**
  1. **Live vLLM check:** send one Gemma request via `VLLMAnnotator(..., guided_json=True)` and
     assert the response JSON is schema-exact (all 6 keys incl. `reason`, valid enum). If the key
     is still dropped, guided decoding is not enforced server-side → record findings (xgrammar
     backend availability, vLLM version, `extra_body` handling) for a follow-up.
  2. **Clear the stale store first (REQUIRED — see §1.7):** the prompt edits in T1 keep the same
     `prompt_id`s, so resume would skip every existing arm and leave `reason` null. Delete or
     rename `data/interim/bakeoff/bakeoff_labels.csv` before re-running so all 12 arms
     re-annotate fresh under the new prompts. (A from-scratch run also writes the store directly
     in the WS2 clean-CSV format.)
  3. **Re-run stage 02:** `uv run python scripts/02_bakeoff_prompts.py --config
     config/religious_missions.yaml`. Regenerates `bakeoff_labels.csv` (clean per T2),
     `bakeoff_results.json`, `proposed_slate.json` (new format per T5).
  4. **Gate (the real acceptance):** confirm `reason` is **non-null content** for Gemma rows and
     broadly across arms (not merely key-present); confirm the CSV is human-readable; confirm the
     conformance flag (T3) shows no missing-key warnings for the vLLM arm.
- **Constraints:** UCloud-only; do not run as part of the local CI path. Don't touch the frozen
  test set. Pin OpenAI snapshot ids as configured. Do **not** rely on resume after a prompt edit
  (§1.7) — clearing the store in step 2 is mandatory, not optional.
- **Acceptance:** live Gemma response is schema-exact; post-run `reason` non-null for all 12
  arms; clean CSV; updated slate + (re-rendered) figure reflect new data.
- **Report-back:** per-arm `reason` non-null rate before/after; whether vLLM guided decoding was
  confirmed enforced; any server-side findings.

### T10 — Additive bake-off artifacts (score from the full store) — POST-FIX ENHANCEMENT
- **Depends on:** **T5** (slate restructure) and **T7** (figure drawing fn), and conceptually the
  fixes (T1–T3) so the accumulated store is clean. **Parallel-safe with:** T11 is its only
  dependent. **Shared-file hotspots:** `bakeoff_prompts.py` (co-owned with T5 — sequence T5→T10).
- **Objective:** make `bakeoff_results.json` / `proposed_slate.json` / the figure an **additive
  full-store view** so models benchmarked across separate runs (incl. sequential local serving,
  T11) all appear together — the store (`bakeoff_labels.csv`) becomes the single source of truth.
- **Why:** today `results` is built only from the *current* YAML `candidates × prompts` and the
  file is overwritten (see §1.7). The raw store is already additive; only the derived artifacts
  are scoped/overwritten.
- **Files & locations:** `src/binary_classifier/annotate/bakeoff_prompts.py` — `run_bakeoff`
  (the loop that builds `results` from `groups`, ~L165–224) and `_score_vs_human`/
  `_build_proposed_slate`. Reuse the T5 `--from-results`/local-regen entrypoint, extended to read
  the **store** (richer/more current than `results.json`).
- **Steps:**
  1. Add a **score-from-store** path: build `results` over the **union** of the current `groups`
     and **all distinct `source_id`s present in the store** (parse `model_id`/`prompt_id` from
     each `source_id` = `model_id + "__" + prompt_id`). Score each from stored labels via the
     existing `_score_vs_human`. Make this the default, or a `--full-store` flag (document choice).
  2. **Provider resolution:** for arms whose `model_id` is in the current YAML candidates, use the
     YAML provider; for store-only arms (old/other experiments not in the YAML), set
     `provider: null` / `"unknown"` and surface them as "not currently configured" so the human
     resolves at gate G2. Do not invent a provider.
  3. **Preserve the Slate consumer contract (§1.6):** every `selected[]`/`models[]` entry keeps
     `model_id` + `prompt_id`; only models with a known provider are eligible for the
     `recommended` block (stage 03 needs the provider to serve them).
  4. The figure (T7 `bakeoff_summary`) then naturally renders all arms in the store.
- **Constraints:** keep selection logic semantics unchanged (κ/F1-CI thresholds + fallback).
  Don't prune the store. Be explicit that, post-T10, removing a model from the YAML no longer
  removes it from the artifacts (it must be pruned from the store) — document this in §1.7 terms.
- **Acceptance (tests, local/zero-cost):** with a store containing arms beyond the current YAML
  candidate set, `results.json`/slate/figure include **all** store arms; arms not in the YAML
  appear with `provider: null` and are excluded from `recommended`; slate still validates via
  `load_slate`/`Slate`. `uv run pytest tests/ -k "bakeoff or slate"`.
- **Report-back:** the additive trigger (default vs flag), and how store-only/no-provider arms are
  represented (T11 relies on this to assemble the combined view).

### T11 — Automatic sequential local-LLM serving driver — POST-FIX ENHANCEMENT (UCloud)
- **Depends on:** **T10** (so the per-model runs accumulate into one combined artifact view) and
  the fixes (T1–T3). **Parallel-safe with:** none. **Shared-file hotspots:** none (new driver +
  a small CLI filter).
- **Objective:** run a multi-model bake-off on **one GPU serving one vLLM model at a time**,
  automatically — serve model A → annotate A's arms → stop → serve B → annotate B's arms → … —
  while OpenAI arms run once. Leans on the append-only store (§1.7) + additive artifacts (T10).
- **Why it's needed:** `run_bakeoff` fires all arms concurrently and every vLLM arm hits the same
  `VLLM_BASE_URL` (`factory.py`: `base_url=os.environ.get("VLLM_BASE_URL", ":8000")`,
  `model_id=spec.id`); with one model loaded, a second vLLM candidate 404s on model mismatch. So
  multiple open-weight arms must be served sequentially.
- **Design (Option A — thin driver; chosen over in-process lifecycle):**
  1. Add a small filter to the stage-02 CLI so a run can be restricted to one model, e.g.
     `scripts/02_bakeoff_prompts.py --only-model <id>` (the underlying `run_bakeoff` already
     accepts a `candidates=` override — wire the CLI to it).
  2. Add a driver (e.g. `scripts/bakeoff_sequential.py` or extend `utils/serve-llm.sh` with a
     loop) that, for each `provider: vllm` candidate in the YAML: start the vLLM server for that
     model id (the serve script reads the model id from the YAML candidate — single source of
     truth), wait for readiness (health poll on `:8000`), run `02 --only-model <id>`, then stop
     the server. Run OpenAI candidates once (no server needed).
  3. After the loop, run the T10 additive rebuild so `results.json`/slate/figure cover every
     model in one combined view.
- **Constraints:** UCloud/GPU only; not part of local CI. Reuse `serve-llm.sh` for serving and
  the existing resume (append) so re-runs are idempotent. Handle server-start failure (don't
  leave a zombie server; don't silently skip a model). Respect `LLM_API_KEY`/`VLLM_BASE_URL`
  env contract (`factory.py` notes a key mismatch 401s every request into error records).
- **Acceptance:** on UCloud, a single driver invocation benchmarks ≥2 vLLM models + the OpenAI
  arms with only one model resident at a time; the final `bakeoff_results.json`/figure show all
  of them together (via T10); the store has each model's rows appended once.
- **Report-back:** the driver entrypoint + CLI filter flag, per-model serve/run timings, and any
  serving caveats (GPU memory, readiness, multiple vLLM ports).

---

## 4. Global integration verification (orchestrator runs after the relevant tasks land)

**Local (zero-cost; after T1–T8):**
1. `uv run pytest tests/ -k "schema or annotat or bakeoff or store or slate or viz or visualize"`
   — all green.
2. CSV: open the re-encoded `data/interim/bakeoff/bakeoff_labels.csv` (T4) and confirm cells are
   human-readable (no `""```json\n{\"...` nesting) and still load via `AnnotationStore`.
3. Slate: rebuild `proposed_slate.json` from the existing `bakeoff_results.json` (T5 local
   entrypoint); confirm it validates via `load_slate`/`Slate` and yields valid
   `(model_id, prompt_id)` pairs.
4. Figures: `uv run python scripts/10_visualize.py` → `bakeoff_summary.{pdf,svg,png}` at 300 DPI
   under the shared style; existing figures re-rendered consistently; stage-03 figures log clean
   skips; no font-substitution warnings.

**UCloud (only T9):** live vLLM schema-exact check; stage-02 re-run; `reason` non-null gate;
clean CSV at production path; optional small `--limit scripts/03_annotate.py` spot-check that
`reason` populates and `annotation_store.csv` stays clean at production scale.

**Enhancements (T10–T11, optional, after the sprint):** (T10, local) with a store holding arms
beyond the current YAML, `results.json`/slate/figure render the full additive set, no-provider
arms flagged and excluded from `recommended`, slate still validates. (T11, UCloud) one driver
invocation benchmarks ≥2 vLLM models one-at-a-time + OpenAI arms, producing one combined
figure/slate via T10. These do **not** gate the core bug-fix definition of done.

**Definition of done:** `reason` is reliably populated and human-readable across all arms in
both stage 02 and (spot-checked) stage 03; the label stores are clean standard CSV;
`proposed_slate.json` is a ranked, recommendation-bearing summary that still drives stage 03;
and all pipeline figures (existing + new bake-off/stage-03) render to paper-ready PDF/SVG/PNG
from current data, on demand, with no UCloud dependency for anything except the re-run.
