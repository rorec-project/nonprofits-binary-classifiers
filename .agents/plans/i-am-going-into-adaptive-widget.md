# Executable Work-Order Pack — Pipeline Simplification + Pedagogical Documentation

> **Audience: an orchestrator agent.** This is a self-contained plan. You (the
> orchestrator) read it in full, then implement it by spawning one fresh-context
> subagent per work order (§3), pasting that subagent the §1 Shared Context Pack +
> its own §3 work order. You verify every subagent's report yourself (§0.5) before
> marking a task done. No external runner, no `ralph.sh`, no harness assumptions —
> everything needed is in this document.

---

## 0. Orchestrator Protocol

### 0.1 Roles
- **Orchestrator (you):** own sequencing, spawning, verification, and the final
  gate. You do not write task code yourself; you dispatch and verify.
- **Subagent:** executes exactly ONE work order with surgical precision, then
  returns the §0.4 report. Starts with zero prior context — you must paste it
  everything (§0.3).

### 0.2 Execution DAG
```
Wave 0 (solo):      T1  (deps + lazy import; regenerates uv.lock)
                      │
Wave 1 (parallel):  T2  T3  T4  T5  T6  T7   (all depend only on T1)
                      │
Wave 2 (parallel):  T8  T9  T10              (docs; depend on T1–T4 = final code)
                      │
Wave 3 (you):       Final acceptance gate (§4)
```
Within a wave, tasks are file-disjoint and may run concurrently. Do not start a
wave until the previous wave's tasks are all `done` and verified.

### 0.3 Subagent spawn contract — paste ALL of the following into each spawn
1. The entire **§1 Shared Context Pack**.
2. The subagent's **§3 work order** verbatim (Owns / Must-not-touch / Operations /
   Acceptance / Report-back).
3. An explicit line: *"You may modify ONLY the files under 'Owns'. Treat every
   other file as read-only. If you believe you must touch another file, STOP and
   report it as a RISK instead."*
4. The §0.4 report-back format.
Subagents start fresh: paste content, never merely reference a path.

### 0.4 Subagent report-back contract (every subagent's final message MUST contain)
- **FILES**: each file created/modified/deleted, absolute path + one-line summary.
- **TESTS**: exact commands run + the pass/fail summary line(s).
- **DEVIATIONS**: any departure from the work order + justification, or "none".
- **RISKS**: anything affecting other tasks/waves, or "none".
- **ARTIFACTS**: any new/changed artifact schema, or "none".

### 0.5 Verification discipline
For each returned task, **re-run its Acceptance checks yourself** before marking it
done — never trust the subagent's pass claim. If a check fails, return the task to
a fresh subagent with the failure pasted in. Record the verified result.

### 0.6 Branch / commit / safety rules
- Work on one feature branch (e.g. `refactor/simplify-and-document`); branch from
  the current HEAD. **Never `git push`, never open/merge a PR** — stop at commit.
- Commit per wave with a Conventional Commit message; no attribution trailers
  unless the human adds them.
- **One owner per file** (enforced by §3 "Owns"/"Must-not-touch"). The shared,
  high-collision files (`config.py`, the YAMLs, `pyproject.toml`, `evaluate.py`)
  each have exactly one owner.
- **Never create or modify production human-gate artifacts** under
  `data/processed/gold/` (`gold_to_code.csv`, `production_slate.json`,
  `anchor_to_code.csv`, `selected_model.json`, `test_unlock.json`). Smoke runs use
  `config/smoke.yaml` and a scratch path only.

---

## 1. Shared Context Pack  *(paste into every subagent)*

### 1.1 Project + goal
A config-driven binary text classifier labeling US nonprofit missions as religious
(`1`) vs non-religious (`0`) via an LLM-as-primary weak-supervision ensemble, with
a downstream **population-prevalence estimate** (PPI++) as the research deliverable.
Stages 01–09 run via `scripts/run_pipeline.py`; 10–11 are script-only. The actual
research contribution is the *calibrated, uncertainty-quantified prevalence
estimate* — preserve everything that supports it.

### 1.2 Repo conventions (binding)
- **uv only:** `uv run python …`, `uv run ruff …`, `uv run ty …`. `pyproject.toml`
  + `uv.lock` are the source of truth.
- **Lint/format/type:** `uv run ruff check .`, `uv run ruff format .`,
  `uv run ty check`. Change rules in `pyproject.toml`, never inline `# noqa` /
  `# type: ignore`. Python 3.13.
- **Paths:** `pathlib.Path` only; prefer `PathRegistry` properties.
- **Docstrings:** Google style (match the package).
- **Comments** (`docs/agents/conventions/comments.md`): heavy,
  scan-friendly **section comments above code (never inline)** explaining the
  *why* / methodology / data provenance / non-obvious calculations — not syntax.

### 1.3 Confirmed design decisions (the "what" + "why")
Simplification principle: **one principled primary method per concern + minimal
robustness; push tertiary/diagnostic/dead-path machinery to optional extras.**
Research basis lives in `.agents/docs/2026060*` (cited per module in §1.5).

**DO (this pack implements):**
1. `crowd-kit`, `cleanlab`, `quapy` → optional extras (dead-in-default-path or
   tertiary; all already behind `ImportError` guards except the one fixed in T1).
2. Prevalence cross-checks default → `[emq]` (vendored SLD); KDEy opt-in. EMQ as
   the single quantification sensitivity check (matches the project's research).
3. Training arms default → `[hard, class_weighted]`; drop `pruned`. Soft vote-share
   stays the default target. Rationale: soft targets already down-weight the
   disagreement band that `pruned`/cleanlab removes (arXiv:2511.14117;
   arXiv:2605.20642). Keep `pruned_arm` code for opt-in.
4. Drop the Vickers-Elkin decision-curve (orthogonal to a prevalence study).
5. Sweep: drop the `{25,50,100}%` learning curve (→ `[1.0]`).

**DO NOT change (these ARE the fidelity — confirmed with the user):**
- Calibration stays **Platt + temperature** (compared, then deployed). Do NOT
  reduce to temperature-only; do NOT add isotonic (overfits at anchor n=500).
- `final_seeds` stays **5** (research mandates ≥5 seeds for reported variance).
- Acceptance gate stays **`max_ece`-only** for this pass (Brier/log-loss = future
  work, out of scope).
- PPI++ primary, frozen-test gate (G3), gold/silver discipline, global seeds,
  bootstrap CIs, **majority-vote** production aggregation, minority-class metrics,
  the LOW-tier rule layer, and the DeBERTa-v3 vs ModernBERT comparison: untouched.

### 1.4 Documentation standard (pedagogical — applies to every code file a task owns)
Bring each owned module to a standard where an external reader follows it without
reading call sites:
- **Module docstring**: the stage's role in the pipeline, data provenance, the
  methodology, and the key citations (author, year).
- **Google-style docstrings** on every public function/class (Args/Returns/Raises).
- **Section comments above code** explaining the *why* and any non-obvious
  statistics/transform — per §1.2.
- **Embed citations at the point of use** (in the relevant docstring/section
  comment). Pull exact DOIs from the `.agents/docs/2026060*` reports.

### 1.5 Citation map (module → references to embed)
- `evaluation/evaluate.py`, `metrics.py` — Davis & Goadrich (2006); Saito &
  Rehmsmeier (2015); Hernández-Orallo et al. (2012); Chicco & Jurman (2020, MCC).
- `evaluation/calibration.py` — Platt (1999); Zadrozny & Elkan (2002); Guo et al.
  (2017); Desai & Durrett (2020); Silva Filho et al. (2023); isotonic-overfit
  caveat for small calibration sets.
- `evaluation/thresholds.py` — Forman (2008, cost/threshold framing).
- `prevalence/estimate.py`, `prevalence/ppi.py` — Angelopoulos et al. (2023, PPI;
  Science adi6001 / arXiv:2311.01453); Hopkins & King (2010); Keith & O'Connor
  (2018); Gentzkow, Kelly & Taddy (2019); Meyer & Mittag (2017).
- `prevalence/composite.py` — Rogan & Gladen (1978); Forman (2008).
- `prevalence/quantify.py` — Saerens et al. (2002, EM/SLD); Bella et al. (2010);
  González et al. (2017); Esuli et al. (2023); Schumacher et al. (2025); González,
  Moreo & Sebastiani (2024, shift caveat); Moreo et al. (2021, QuaPy/KDEy).
- `prevalence/weights.py` — Horvitz–Thompson (1952); PPI difference-estimator
  framing (arXiv:2603.19160).
- `annotate/aggregate.py` — Dawid & Skene (1979); Goh, Mueller et al. (2022,
  CROWDLAB); Ratner et al. (2016/2017); majority-vote-for-LLM (arXiv:2511.15714;
  arXiv:2601.22336).
- `annotate/run_annotation.py` — Gilardi, Alizadeh & Kubli (2023); Pangakis,
  Wolken & Fasching (2023); Ziems et al. (2024).
- `qc/agreement.py` — Cohen (1960); Landis & Koch (1977); Krippendorff (alpha);
  SILICON (arXiv:2412.14461); Variance-Aware (arXiv:2601.02370).
- `train/arms.py` — soft labels: arXiv:2511.14117, arXiv:2605.20642, PMC12148080;
  class weighting: King & Zeng (2001), "Balancing the Scales" (arXiv:2409.19751);
  pruned/cleanlab (opt-in): Northcutt, Jiang & Chuang (2021).
- `train/encoder.py`, `train/sweep.py` — He et al. (2021/2023, DeBERTa-v3); Warner
  et al. (2024, ModernBERT); Reimers & Gurevych (2019) + Wang et al. (2020,
  MiniLM); noisy-label fine-tuning: Zhu et al. (2022), Wang et al. (2023).

---

## 2. Reference facts (verified against the current tree)
- `pyproject.toml` core deps currently include `crowd-kit>=1.4.2`,
  `cleanlab>=2.9.0`, `quapy>=0.2.0,<0.3`; `vllm` is already under
  `[project.optional-dependencies].serve`.
- `src/binary_classifier/train/arms.py:11` — `from cleanlab.filter import
  find_label_issues` (module-level; used in `pruned_arm` L98–132 and `prune_ein2s`
  L177–209).
- `src/binary_classifier/config.py` factories: `_default_calibration_methods`
  (~L162), `_default_prevalence_cross_checks` (~L220), `_default_training_arms`
  (~L398), `_default_curve_fractions` (~L403), `_default_final_seeds` (~L413).
- `src/binary_classifier/evaluation/evaluate.py:26` imports `net_benefit`;
  `:501` emits `"decision_curve": net_benefit(...)`; `:492` emits `"subgroups":
  subgroup_report(...)` (KEEP subgroups).
- `trainer.py:213` `_needs_oof` returns True iff `pruned` in arms OR `crowdlab` in
  comparison_arms → after T2 the default path skips training OOF.
- `quapy`/`crowd-kit` are already lazily imported behind `ImportError`
  (`quantify.py`, `aggregate.py`); only `cleanlab` (arms.py) needs the lazy fix.

---

## 3. Work Orders

> Every task: follow §1 conventions; bring owned code files to the §1.4 doc
> standard with §1.5 citations; return the §0.4 report.

### T1 — Dependencies → optional extras + lazy import  *(Wave 0, solo)*
- **Owns:** `pyproject.toml`, `src/binary_classifier/train/arms.py`.
- **Must-not-touch:** everything else.
- **Operations:**
  1. In `pyproject.toml`, remove `crowd-kit`, `cleanlab`, `quapy` from
     `[project].dependencies`. Add to `[project.optional-dependencies]`:
     `diagnostics = ["crowd-kit>=1.4.2", "cleanlab>=2.9.0"]` and
     `quant = ["quapy>=0.2.0,<0.3"]`. Keep `serve` as-is.
  2. In `arms.py`, delete the module-level `from cleanlab.filter import
     find_label_issues` (L11). Add a local import `from cleanlab.filter import
     find_label_issues` inside `prune_ein2s` (the only function that calls it),
     with a section comment: cleanlab is an optional `diagnostics` extra; the
     pruned arm is opt-in (cite Northcutt, Jiang & Chuang 2021).
  3. Run `uv lock` / `uv sync` to regenerate the lockfile lean.
  4. Bring `arms.py` to the §1.4 doc standard (soft/hard/class_weighted/pruned
     rationale + §1.5 citations).
- **Acceptance:**
  - `uv sync` (no extras) succeeds.
  - `uv run python -c "import binary_classifier.train.arms"` succeeds with
    cleanlab absent.
  - `uv run ruff check . && uv run ty check` green.
  - `uv sync --all-extras && uv run pytest tests/test_arms.py` green (cleanlab
    present).

### T2 — Config defaults  *(Wave 1)*
- **Owns:** `src/binary_classifier/config.py`, `config/religious_missions.yaml`,
  `config/smoke.yaml`, plus tests asserting these defaults
  (`tests/test_prevalence_stage.py`, `tests/test_sweep.py`, `tests/test_foundation.py`
  — only the assertions about the changed defaults).
- **Must-not-touch:** `arms.py`, `evaluate.py`, test files owned by T3/T4.
- **Operations:**
  1. `_default_prevalence_cross_checks` → return `["emq"]`. Update YAML
     `prevalence.cross_checks: [emq]`. Section comment: PPI++ primary; EMQ
     (vendored SLD) is the single quantification sensitivity check; KDEy opt-in
     under the `quant` extra (cite Saerens 2002; Angelopoulos 2023).
  2. `_default_training_arms` → return `["hard", "class_weighted"]`. Update YAML
     `training.arms: [hard, class_weighted]`. Comment: soft is the default target;
     `pruned` dropped as redundant with soft targets (cite arXiv:2605.20642),
     available opt-in.
  3. `_default_curve_fractions` → return `[1.0]`. Update YAML
     `training.curve_fractions: [1.0]`. Comment: learning curve dropped; single
     full-data run.
  4. **Leave unchanged** (assert in a comment WHY): `_default_calibration_methods`
     (`[platt, temperature]`), `_default_final_seeds` (5 seeds), the `max_ece`
     acceptance gate.
  5. Reconcile `config/smoke.yaml`: if it overrides any of the three changed keys,
     align it; otherwise leave.
  6. Update only the test assertions that encode the old defaults.
- **Acceptance:**
  - `uv run python -c "from binary_classifier.config import load_config;
    load_config('config/religious_missions.yaml'); load_config('config/smoke.yaml')"`.
  - `uv run pytest tests/test_prevalence_stage.py tests/test_sweep.py
    tests/test_foundation.py` green; `uv run ruff check . && uv run ty check` green.

### T3 — Evaluation slimming (remove decision-curve)  *(Wave 1)*
- **Owns:** `src/binary_classifier/evaluation/evaluate.py`,
  `src/binary_classifier/evaluation/decision_curve.py` (delete),
  `tests/test_decision_curve.py` (delete), `tests/test_evaluate_stage.py`.
- **Must-not-touch:** `calibration.py`, `thresholds.py`, `subgroups.py`,
  `metrics.py` (T6 owns those).
- **Operations:**
  1. In `evaluate.py`: remove the `net_benefit` import (L26) and the
     `"decision_curve": net_benefit(...)` report entry (~L501) and any
     now-unused threshold list feeding it. **Keep** `subgroup_report` (L492).
  2. Delete `evaluation/decision_curve.py` and `tests/test_decision_curve.py`.
  3. Update `tests/test_evaluate_stage.py` to drop any `decision_curve`
     assertions; keep all other metric assertions.
  4. Doc pass on `evaluate.py` per §1.4 (§1.5 metrics citations); add a one-line
     note that decision-curve was intentionally removed for a prevalence study.
- **Acceptance:**
  - `grep -rn "net_benefit\|decision_curve" src/ tests/` returns nothing.
  - `uv run pytest tests/test_evaluate_stage.py` green; ruff + ty green.

### T4 — Test hygiene for optional dependencies  *(Wave 1)*
- **Owns:** `tests/test_quantify.py`, `tests/test_aggregation_compare.py`,
  `tests/test_crossfit.py`, the `pruned` cases in `tests/test_arms.py`.
- **Must-not-touch:** source files; tests owned by T2/T3.
- **Operations:** add `pytest.importorskip("quapy" | "cleanlab" | "crowdkit")` at
  the top of each test (or per-test) that imports an optional dependency, so the
  lean suite skips them and the full suite runs them.
- **Acceptance:**
  - Lean: `uv sync && uv run pytest tests/test_quantify.py
    tests/test_aggregation_compare.py tests/test_crossfit.py tests/test_arms.py`
    → optional-dep tests **skipped**, rest green.
  - Full: `uv sync --all-extras && uv run pytest <same>` → all run, green.

### T5 — Pedagogical docs: prevalence subsystem  *(Wave 1, docs-only)*
- **Owns:** `prevalence/estimate.py`, `prevalence/ppi.py`, `prevalence/composite.py`,
  `prevalence/quantify.py`, `prevalence/weights.py`.
- **Must-not-touch:** config, tests, any non-prevalence module.
- **Operations:** §1.4 doc standard + §1.5 citations. No behavior change.
- **Acceptance:** `uv run ruff check . && uv run ty check` green;
  `uv run pytest tests/test_ppi.py tests/test_quantify.py
  tests/test_prevalence_weights.py tests/test_prevalence_stage.py` green
  (run after T2/T4 land or coordinate; docstring-only edits won't change behavior).

### T6 — Pedagogical docs: evaluation metrics/calibration  *(Wave 1, docs-only)*
- **Owns:** `evaluation/calibration.py`, `evaluation/thresholds.py`,
  `evaluation/subgroups.py`, `metrics.py`.
- **Must-not-touch:** `evaluate.py` (T3), prevalence (T5).
- **Operations:** §1.4 + §1.5. Emphasize in `calibration.py`: Platt+temperature
  comparison rationale and the isotonic-overfit caveat.
- **Acceptance:** ruff + ty green; `uv run pytest tests/test_calibration.py
  tests/test_thresholds.py tests/test_subgroups.py tests/test_metrics.py` green.

### T7 — Pedagogical docs: annotate / train / qc  *(Wave 1, docs-only)*
- **Owns:** `annotate/aggregate.py`, `annotate/run_annotation.py`,
  `train/encoder.py`, `train/sweep.py`, `train/crossfit.py`, `qc/agreement.py`.
- **Must-not-touch:** `arms.py` (T1), config, tests.
- **Operations:** §1.4 + §1.5. In `aggregate.py`, document why majority is the
  production default and DS/CROWDLAB are opt-in diagnostics.
- **Acceptance:** ruff + ty green; `uv run pytest tests/test_aggregate.py
  tests/test_agreement.py tests/test_sweep.py` green.

### T8 — `README.md` restructure (ELI5-first, zero information loss)  *(Wave 2)*
- **Owns:** `README.md`.
- **Hard rule:** **relocate, never delete** — every fact in the current README must
  appear somewhere in the new one (body or appendix). Diff the fact set before/after.
- **Operations — target structure:**
  1. What this is (plain paragraph + analogy). 2. Big picture (9-stage diagram +
  "cheap AI labels ⇄ small human gold check"). 3. Three core ideas (AI
  bulk-labels/humans spot-check; model never sees the final exam; we statistically
  *correct* counts for population estimates). 4. Quickstart (install **lean vs
  `--all-extras`**, smoke test, operator loop). 5. The 9 stages in plain English
  (one paragraph each → appendix pointer). 6. The 4 human checkpoints G1–G4.
  7. **How to run an evaluation and read the results** (acceptance gate
  `min_pr_auc 0.90` / `min_minority_f1_ci_lower 0.70` / `max_ece 0.05`; outputs
  `test_evaluation.json`, `prevalence_report.json`; how to read the prevalence CI +
  LOW-tier caveat). 8. Configuring in 60 seconds → link to `config/README.md`.
  **Appendices:** A Data layout & DVC · B Stage-by-stage technical reference (I/O,
  columns, thresholds) · C Methodology & citations (§1.5 + the intentional
  simplifications/skips) · D Sampling frame vs population · E UCloud runtime ·
  F Troubleshooting · G Legacy pipeline.
  Reflect the simplified state (optional extras; arms; cross-checks=emq; lean curve;
  decision-curve removed; calibration & 5 seeds retained).
- **Acceptance:** all prior README facts present; reads top-to-bottom for the
  target persona; markdown lints clean (no broken internal links).

### T9 — `config/README.md` decision-first rewrite (no traffic lights)  *(Wave 2)*
- **Owns:** `config/README.md`.
- **Operations:** keep all current reference content; reorder to:
  (a) **"What you actually decide"** — `entity`/`field`/`label_name`, confirm the
  model slate, the 4 gates, each with "if unsure, use the default";
  (b) **"What you can safely leave at default"**; (c) the full section-by-section
  tables (Pydantic line refs moved to the end of each section); (d) a worked
  **retasking walkthrough** (missions → activities). Update changed defaults
  (cross_checks, arms, curve). **No traffic-light symbols.**
- **Acceptance:** opens with the decision checklist; every current knob still
  documented; defaults match the YAML after T2.

### T10 — Thin-doc sync  *(Wave 2)*
- **Owns:** `AGENTS.md`, `docs/agents/pipeline.md`,
  `docs/agents/configuration.md`.
- **Operations:** keep them as thin pointers; update stale specifics: optional
  deps, arms default, cross-checks default, decision-curve removal, lean curve.
- **Acceptance:** no contradictions with README/config-README/YAML; ruff/ty N/A.

---

## 4. Final acceptance gate  *(orchestrator runs after Wave 2)*
1. `uv run ruff check . && uv run ruff format --check . && uv run ty check` — green.
2. **Lean integrity:** `uv sync` →
   `uv run python -c "import binary_classifier.train.arms, binary_classifier.train.sweep, binary_classifier.evaluation.evaluate"`
   succeeds with `cleanlab`/`quapy`/`crowd-kit` absent.
3. **Full suite:** `uv sync --all-extras && uv run pytest -m "not slow and not network"` — green.
4. **Synthetic smoke** (`config/smoke.yaml`, `allow_synthetic: true`, scratch path):
   `06_train.py --limit 200` (no pruned/OOF, no cleanlab import) · `07_evaluate.py`
   (no `decision_curve` key; calibration still compares platt+temperature) ·
   `09_prevalence.py` (cross-checks = EMQ only; no quapy import).
5. **Invariants:** `_needs_oof(cfg)` is `False`; `len(final_seeds) == 5`;
   `curve_fractions == [1.0]`; `calibration_methods == ["platt","temperature"]`.
6. **Docs:** README carries no information loss; config README is decision-first
   with no traffic lights.
7. Commit per wave (no push). Report a consolidated FILES/TESTS/DEVIATIONS/RISKS
   summary back to the human.

## 5. Out of scope (record as future work, do NOT implement)
- Brier/log-loss in the acceptance gate (calibration-selection metric).
- Multicalibrated-LLM prevalence (Linder et al. 2026) — an addition, not a
  simplification.
- Any change to PPI++, frozen-test gate, gold/silver split, majority-vote
  production aggregation, or the encoder comparison.
