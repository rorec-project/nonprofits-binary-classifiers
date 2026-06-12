# PR-1 — `feature/05-anchor-sample` (shared metrics + anchor stage + gates foundation)

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/05-anchor-sample` |
| Depends on | PR-0 — sentinels: `pyproject.toml` pins `transformers>=5.8.0,<6.0.0`; `[tool.pytest.ini_options]` registers `slow`/`network` markers; `vllm` only under `[project.optional-dependencies].serve` |
| Blocks | PR-2 … PR-7 |
| Spec blocks implemented | §6.1 `AnchorConfig`; §6.2 (ALL registry properties — one PR-wide pass); §6.3 (`anchor:` block + `config/smoke.yaml`) |
| Ralph state | `.agents/ralph/state/pr-1.md` + `pr-1.status` |

**Pre-flight (first iteration):** verify the PR-0 sentinels above; switch to / create
the branch.

**Sequencing note:** T1.2 lands before (or is coordinated with) T1.1 — T1.1's
silver-labels-path switch needs `registry.silver_labels` from T1.2 (stated in T1.1).

**Human checkpoints:** none mandatory — the PR gate's "manual orchestrator check"
(smoke stages 01,05) is run by the orchestrator; T1.3's script acceptance runs in a
scratch checkout (ORCHESTRATOR.md A.5).

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-1 — `feature/05-anchor-sample` (shared metrics + anchor stage + gates foundation)

**Objective**: promote the metric bundle to a shared module; implement stage 05 so the
human coding session can start; lay all registry/config/gate plumbing later PRs need.

**T1.1 — `metrics.py` promotion** (owns `src/binary_classifier/metrics.py`,
`src/binary_classifier/qc/agreement.py`, `tests/test_metrics.py`)
Operations: create `metrics.py` exposing
`compute_metric_bundle(y_true, y_pred, *, y_score=None, minority_class=1, seed=42,
n_resamples=1000, confidence_level=0.95) -> dict` and
`bootstrap_ci(y_true, y_pred, minority_class, seed, n_resamples=1000,
confidence_level=0.95) -> dict` by MOVING the bodies of `_compute_metrics`
(agreement.py:345–416) and `_bootstrap_ci` (457–511). Add `roc_auc` to the bundle when
`y_score` is provided (sklearn `roc_auc_score`). Rewire `agreement.py` to import and
delegate (keep its private names as thin aliases so its internal call sites and the
13 existing tests pass UNCHANGED). Also switch agreement.py's inline silver-labels
path to `registry.silver_labels` (after T1.2 lands — sequence T1.2 → T1.1 or
coordinate). New `tests/test_metrics.py`: numeric parity test (same fixture through
old API path and new module gives identical numbers), roc_auc presence/absence.
Acceptance: full Tier-1 green, including the untouched `tests/test_agreement.py`.

**T1.2 — registry + config plumbing** (owns `src/binary_classifier/paths.py`,
`src/binary_classifier/config.py`, `config/religious_missions.yaml`, `config/smoke.yaml`)
Operations: add ALL §6.2 properties + `ensure_dirs()` entries (one PR-wide pass, so
later PRs never touch paths.py; nested dirs like `predictions_dir / "shards"` use
`mkdir(parents=True, exist_ok=True)` as the existing code does). Add `AnchorConfig`
to config.py + root field + YAML block, and set
`model_config = ConfigDict(extra="ignore")` explicitly on `BinaryClassifierConfig`
(makes the already-default behavior visible; see §3.2). Create `config/smoke.yaml`
exactly per §6.3 — its training/evaluation/… blocks reference fields added in later
PRs; with extra="ignore" this loads fine today, and the acceptance check proves it.
Acceptance: `uv run python -c "from binary_classifier.config import load_config;
load_config('config/religious_missions.yaml'); load_config('config/smoke.yaml')"`;
Tier-1 green.

**T1.3 — anchor stage** (owns `src/binary_classifier/data/anchor.py`,
`scripts/05_build_anchor.py`, `tests/test_anchor.py`)
Operations: implement `build_anchor(cfg, registry, force=False) -> None`:
1. `df = load_missions(cfg)`; compute `Q = compute_quality_score(text)` and
   `tier = assign_tier(Q, cfg.q_thresholds)` for every row (vectorize via `.map`).
2. Exclusion: drop EIN2s present in `registry.silver_manifest` or
   `registry.gold_manifest` (string-normalized comparison per §5.5#5). Log the
   excluded count (~3.7% of frame) — this is a deliberate, documented estimand note.
3. Stratify by `tier × ntee_major_group`. Allocation: proportionate to stratum frame
   share, times `oversample_low_factor` for LOW strata, floor of 1 for any stratum
   with ≥ `min_stratum_frame` rows; renormalize to `cfg.anchor.n`. Draw with
   `np.random.default_rng(cfg.SEED)`. Compute and store per-stratum
   `sample_prob = n_drawn_k / n_frame_k` (the DSL/PPI design weight input).
4. Write `anchor_manifest` (§5.4 schema) — anchor.py writes this itself (do NOT
   reuse `sample._write_manifest`, whose schema differs): `stratum` is the composite
   `f"{tier}|{ntee_major_group}"` key and `ntee_major_group` is deliberately kept as
   its own column (stage-09 groups by it). Also write the coding template
   `anchor_to_code.csv` (`EIN2, tier, text, human_label` empty) with clobber
   protection: refuse if the template exists with any `human_label` filled, unless
   `force=True` (mirror stage-01's behavior).
Script: thin wrapper (`--config`, `--force`). Tests: allocation sums to n;
`sample_prob` consistent with draw counts; every tier present; LOW oversampled;
exclusion works incl. dtype drift; determinism (two runs identical); clobber
protection (with/without force); synthetic path (`allow_synthetic=True`) end-to-end
on `tiny_registry`.
Acceptance: Tier-1 green; `uv run python scripts/05_build_anchor.py --config
config/smoke.yaml` produces both artifacts in a scratch checkout.

**T1.4 — G4 gate + orchestrator wiring** (owns `src/binary_classifier/qc/preflight.py`,
`scripts/run_pipeline.py`, `tests/test_preflight.py` (extend),
orchestrator test (extend))
Operations: in preflight.py add `_validate_anchor_labels(cfg, registry) ->
list[str]` (template exists; every row has strict 0/1 `human_label`) and wire it into
`validate_gates` for stages `{"07", "09"}` (G4). Add `_STAGE_SPLITS` entries
`"06": "validation"`, `"07": "test"`. In run_pipeline.py: add `"05"` to
`_STAGE_MODULES` (§5.2); pass `force=` to stage 05 like stage 01; extend the G1 check
set from `{"02","04"}` to `{"02","04","06","07"}`; add the G4 check before stages
07/09 (G3 arrives in PR-3 — leave a marked TODO hook). Update the module docstring
gate narrative. Tests: G4 missing/partial/non-binary/complete; `_STAGE_SPLITS`
additions; `_STAGE_MODULES["05"]` import-resolves.
Acceptance: Tier-1 green.

**PR-1 gate**: all tasks green; `run_pipeline.py --config config/smoke.yaml
--stages 01,05` runs end-to-end on synthetic data (manual orchestrator check).

