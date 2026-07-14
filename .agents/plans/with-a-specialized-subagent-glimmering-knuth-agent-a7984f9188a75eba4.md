---
created: 2026-06-10
---

# Skeptical review of the harmonize-pipeline refinement plan

Verified against the actual code on branch `refactor/harmonize-pipeline`. Baseline: `pytest -q` = 69 passed.

## (a) Plan claims that are WRONG or UNVERIFIABLE

1. **`.gitignore` idiom — CORRECT (verified empirically).** Tested in a scratch repo: the proposed
   four-line idiom (`/data/*`, `!/data/processed/`, `/data/processed/*`, `!/data/processed/gold/`)
   un-ignores `data/processed/gold/` while keeping `data/raw/*.parquet` and
   `data/processed/silver_labels.csv` ignored. The current idiom (`.gitignore:28` `data` +
   `:31` `!data/processed/gold/`) DOES ignore gold (`git check-ignore` → `.gitignore:28:data`),
   and `git ls-files data/` = 0. Bug confirmed; fix works. **Caveat:** line 4 `*.parquet` is global —
   gold artifacts are `gold_to_code.csv` + `production_slate.json` (csv/json, paths.py:124-136), so
   parquet is not an issue *today*, but any future `.parquet` placed under `gold/` would still be
   ignored by line 4 and the plan's idiom would not save it. Worth a one-line note.

2. **pipeline.md `results/` claim — WRONG.** Plan E says "docs/agents/pipeline.md `results/`
   → current `data/` layout." `pipeline.md` (31 lines) contains NO `results/` string; it already uses
   `data/` (pipeline.md:24). This edit target does not exist as described.

3. **configuration.md fix is too narrow.** Plan E says change `silver_dir` → `interim_dir`. But
   `silver_dir` is NOT a config field at all — `config.py:40-43` only defines `raw_dir`, `interim_dir`,
   `processed_dir`, `models_dir`. `configuration.md:13-14` documents BOTH `gold_dir` AND `silver_dir`
   as `paths` YAML keys; both are wrong — `gold_dir` is a derived `PathRegistry` property
   (paths.py:79, `processed_dir/"gold"`), not a YAML knob. The doc fix must correct both entries, not
   just `silver_dir`.

4. **Function names / line numbers in workstream D drift from source.**
   - Plan D says "add param to `build_silver` (~sample.py:85)". The function is `build_silver_pool`
     (sample.py:51), not `build_silver`; the `QThresholdsConfig()` line is **sample.py:91**.
   - `build_gold_set` is at sample.py:169; its `QThresholdsConfig()` line is **sample.py:193**.
   - `build_sample` uses `cfg.q_thresholds` at sample.py:349 (plan says :349 — correct).
   The "audit found" section (lines 36) cites 91/193 correctly; the D workstream body cites 85/169.
   Signatures CONFIRMED to take no thresholds param.

## (b) MISSING items the plan should include

5. **Workstream D misses a THIRD `QThresholdsConfig()` site.** `assign_tier` itself constructs a
   default at **quality.py:593** (`if thresholds is None: thresholds = QThresholdsConfig()`). This is
   the legitimate fallback, so it can stay — but the plan's framing ("only `build_sample` respects
   YAML") implies two sites; there are three constructions total. After threading, sample.py:91/193
   should pass the param THROUGH to `assign_tier`, leaving 593 as the only default fallback. Should be
   stated so the implementer doesn't accidentally remove the fallback.

6. **Removing `_read_prompt_file` leaves an unused import.** base.py:80 is the ONLY user of
   `from pathlib import Path` (base.py:10). Removing the method must also remove the now-dead `Path`
   import or ruff/lint will flag it. Plan doesn't mention this.

7. **`load_missions` cannot use `registry.*` as written — under-specified.** Plan E: "`load_missions`
   (load.py:73-74) should use `registry.missions_parquet` / `registry.bmf_parquet`." But
   `load_missions(cfg)` (load.py:46) takes ONLY `cfg`, not a registry. Executing this requires EITHER
   (a) a signature change to `load_missions(cfg, registry)` — which breaks 4 call sites in
   tests/test_load.py (:22,:29,:39) plus sample.py:343 and the `data/__init__.py` export — OR (b)
   constructing a `PathRegistry` inside load.py (load.py currently doesn't import it). Also a
   behavioral difference: load.py uses `Path(paths.raw_dir).resolve()` (CWD-anchored) whereas the
   registry uses `self._root / raw_dir` (`_root`-anchored); under `PathRegistry.from_config(root=...)`
   these can resolve differently, which could shift where synthetic parquets are generated/read. The
   plan must pick an approach and account for the call-site/test fallout.

8. **`abstain_on_fabricated_positive` is the SAME class of defect as `guided_json`, and the plan
   misses it.** The flag (config.py:216, default False) + machinery (`qc/evidence.py:94`
   `abstain_fabricated_positives`) are wired ONLY in the standalone `scripts/04_quality_check.py:88-94`,
   NOT in the package `run_quality_check` (`qc/agreement.py`) that `run_pipeline.py` stage 04 actually
   calls (run_pipeline.py:31). So the orchestrator path silently skips evidence verification + the
   abstain rule. The review brief explicitly asked about this path. If the plan wants `guided_json`
   "honest", consistency demands at least noting (or fixing) that the documented QC abstain rule is a
   no-op via the orchestrator. The plan's verification step 6 (smoke via run_pipeline) will NOT
   exercise it.

9. **No test coverage exists for the symbols B/C/D touch — plan must ADD, not just adjust.**
   - `grep` of `tests/`: ZERO references to `dawid_skene`, `crowdlab`, `aggregate_labels`,
     `canary`/`CANARY`, `QThresholds`, `assign_tier`, `build_silver`, `build_gold`. So:
     - C (quarantine): no existing test asserts the aggregator behavior → plan's verification step 3
       requires a NEW test; none exists to "adjust."
     - B (canary): no existing canary test → step 4 requires a NEW fixture+test.
     - D (threading): step 5 needs a NEW non-default-threshold test (see #13 on why default config
       won't reveal it).
   - The good news: NO existing test will break from B/C/D, because none reference these symbols and
     the YAML thresholds equal the defaults (see #13). `test_schema.py:192` asserts vLLM passes
     `guided_json: <schema>` under the default `guided_json: true`; if E rewires the flag, that test
     must keep passing for the `true` branch and ideally gain a `false`-branch assertion.

10. **AGENTS.md is internally contradictory and the plan's doc-sync doesn't reconcile it.**
    AGENTS.md:43 states "**`data/` and `models/` are symlinks to cloud storage, not git-committed …
    They are gitignored.**" — directly contradicted by AGENTS.md:44 ("Gold-committed … `processed/gold/`
    … is committed"). After A1, line 43's blanket "all of data/ gitignored" becomes outright false, and
    its "`data/` is a symlink" claim is ALSO false on disk: `data/` is a real directory; only
    `data/processed/.../silver` is the symlink (`readlink` → Cloud/Sync/…). The plan's A2/E must FIX
    line 43, not merely "correct the false 'gold is committed' claim." (Committing files under a
    symlinked `data/` would also be fragile — another reason the real-dir reality matters.)

11. **README is internally inconsistent on the gold path.** README:83 still says default gold is
    `data/processed/train_test_datasets/gold/gold_to_code.csv` (stale), while README:126 already uses
    the NEW `data/processed/gold/production_slate.json`. Plan E lists only README:~83; it should
    reconcile the whole file for consistency.

12. **`.agents/plans/let-s-design-a-plan-idempotent-harbor.md`** repeatedly references `silver_dir`
    as a config key (lines 37,52,105,106,110,255). Plan correctly scopes dated `.agents/plans/*` as
    historical/leave-alone — fine — but worth confirming the human agrees these stay frozen even
    though they encode a path field that never shipped.

## (c) RISKS / ordering hazards

13. **HIGHEST-VALUE FINDING — the canary AND the existing stage-04 gate both depend on an
    UNGUARANTEED silver∩validation overlap; in production this is likely a latent freeze-gate bug, and
    the canary inherits it.** Traced end-to-end:
    - `run_annotation` annotates the **silver manifest only** (run_annotation.py:117); the merge at
      run_annotation.py:125 just joins text onto silver rows. NOTHING reads `validation_manifest` /
      `gold_manifest` into the annotation run (grep confirms: those manifests are consumed only by
      stage-01 reporting (01_build_sample.py:58-60) and preflight G1 (preflight.py:83-84) — never fed
      to `run_annotation`).
    - The stage-04 gate (`run_quality_check`) builds its silver side ONLY from the annotation store
      (agreement.py:91), then **inner-joins** aggregated silver labels against the validation split
      (agreement.py:111-116) and **raises `ValueError("No overlapping validation labels…")`** if the
      overlap is empty (agreement.py:117-122).
    - `build_silver_pool` (sample.py:51) and `build_gold_set` (sample.py:169) are **independent draws**
      from the same HIGH+MEDIUM pool. Nothing enforces validation ⊆ silver.
    - Magnitude: overlap ≈ (silver/pool) × |validation|. With silver=20k and validation≈175 drawn from
      a large production pool, expected overlap is **near zero** → the freeze gate would raise and
      freeze nothing on a real run. The 69 passing tests + the synthetic smoke don't catch this because
      a tiny synthetic pool forces high incidental overlap (do NOT "verify" this by running the
      synthetic sampler — it will mislead you into thinking overlap is reliable).

    Consequence for B: even if the canary loader reads validation EIN2s correctly, `canary_only=True`
    filters the SILVER work-items (run_annotation.py:195-197) by that set — so it runs on
    validation∩silver, which is the same near-empty set. So B is at best a partial run on the incidental
    overlap and at worst a silent no-op; the verification step-4 fixture (manifest EIN2s also placed in
    the input df) masks both. **The real fix is upstream of B:** validation EIN2s must be guaranteed
    into the annotation store (e.g. annotate silver ∪ validation, or constrain the sampler so validation
    ⊆ silver). The canary should then either inject validation rows' text into the matrix or hard-fail
    on an empty canary∩pool. Treat this as a blocking design question (Q1) — it is bigger than the
    canary itself because the existing production freeze gate appears to share the defect.

14. **Canary/QC contamination — REAL, as the brief suspected.** The QC agreement gate
    (`run_quality_check`, agreement.py:110,328) loads the **validation** split (`split=="validation"`)
    as ground truth at the 0.85 threshold (config `qc.agreement_threshold`). Seeding the canary from
    the same `validation_manifest` means the ongoing drift-monitor and the one-shot freeze gate observe
    the SAME held-out rows. If the monitor is ever used to decide prompt tweaks (the plan says "must not
    be tuned on it"), validation silently degrades into dev data and the 0.85 gate is no longer
    independent. Plan B's "monitoring-only, prompts must not be tuned on it" docstring is the only
    guardrail and it is advisory, not enforced. Consider sourcing the canary from a dedicated slice or a
    held-out subset of validation reserved for drift only (the plan defers "dedicated monitoring slice" —
    that deferral is the crux of the contamination risk).

15. **A1 depends on a user move that creates an order trap.** `git check-ignore`/`git add -n` in
    A1's verification only pass AFTER the user has physically placed gold at `data/processed/gold/`.
    On disk today: `data/processed/gold/` does NOT exist; legacy `data/processed/train_test_datasets/gold/`
    is EMPTY and `…/silver` is a cloud symlink. If A1 lands before the move, the verification commands
    report nothing to add and the "fix" looks broken. Plan acknowledges "user's move" but should make
    the ordering an explicit precondition/checklist item, and warn that `data/` being a real dir (not a
    symlink, despite AGENTS.md:43) is what makes committing under it possible at all.

16. **`ensure_dirs()` does not create `raw_dir`.** paths.py:162-172 creates interim/manifests, bakeoff,
    models, gold, interim, processed — but NOT `raw_dir`. Plan A3 tells the user to create
    `data/{raw,interim,models}` manually; good, but note that `load_missions` hard-fails on missing raw
    parquet unless `allow_synthetic:true` (load.py:77-86), so a user who skips the raw step gets a
    FileNotFoundError at stage 01, not a friendly setup message. Worth surfacing in A3.

17. **D threading is SAFE for current tests but verification step 5 is mandatory.** YAML
    `q_thresholds` (HIGH 5.0/MEDIUM 3.0/LOW 0.0, config/religious_missions.yaml:40-43) EQUALS
    `QThresholdsConfig()` defaults (config.py:179-181). So threading `cfg.q_thresholds` changes nothing
    with the shipped config → no seeded `test_sample` drift today. The risk only materializes if/when
    someone sets non-default thresholds; the plan's step-5 non-default test is therefore the only thing
    that proves the wiring. Keep it.

18. **C quarantine is safe — confirmed no live callers.** Only caller of `aggregate_labels` is
    agreement.py:99 with `method="majority"`. `aggregate_dawid_skene`/`aggregate_crowdlab` have no other
    references in src/scripts/tests, and no `__init__` re-exports them. Raising `NotImplementedError`
    breaks nothing. (Minor: keeping them in the dispatch dict means import-time is fine; the error only
    fires on explicit selection — as intended.)

19. **validation_manifest existence/timing — OK, with a guard gap (answers the brief's question).** The
    manifest IS produced at stage 01 (`build_sample` → `_write_split_manifest(validation, …)`,
    sample.py:389) and it DOES carry an `EIN2` column (`_write_manifest` cols, sample.py:455-464), so the
    canary loader can read it; no human coding is needed (only the EIN2 list). BUT: it is written at
    stage 01 and lives under `interim_dir/manifests/` (paths.py:114) — i.e. the cloud-symlinked, GIT-
    IGNORED tree, not a committed artifact. A standalone `scripts/03_annotate.py --canary` run without a
    prior stage 01 (or without the interim symlink mounted) hits a missing-file; the loader needs an
    explicit FileNotFoundError guard with a "run stage 01 first" hint (the plan doesn't specify one).

## (d) Open questions for the human

- **Q1 (B + stage-04, BLOCKING):** How are validation EIN2s guaranteed into the annotation store?
  Nothing currently annotates them (`run_annotation` is silver-only), yet the stage-04 freeze gate
  inner-joins against them and raises if the overlap is empty. With silver=20k vs validation≈175 from a
  large pool the overlap is ~0, so the existing gate looks like a latent production bug masked by
  synthetic-smoke overlap. Should the annotation run cover silver ∪ validation, or should the sampler
  constrain validation ⊆ silver? The canary cannot work until this is resolved. And should the canary
  loader *inject* validation text into the matrix (vs filter), and hard-fail on an empty canary∩pool?
- **Q2 (B/contamination):** Are you comfortable that the drift canary and the 0.85 freeze gate share the
  exact same validation rows? If the canary will ever inform prompt iteration, do you want a reserved
  drift-only sub-slice instead?
- **Q3 (E, decision):** `load_missions(cfg)` has no registry. De-duplicate via signature change
  `load_missions(cfg, registry)` (touches 4 test call sites + sample.py + the export) or by building a
  registry inside load.py? And do you accept the `.resolve()` vs `_root` anchoring change?
- **Q4 (E scope):** Should `abstain_on_fabricated_positive` (wired only in scripts/04, absent from the
  orchestrator's `run_quality_check`) be brought into scope alongside the `guided_json` honesty fix, or
  explicitly deferred? Right now `run_pipeline.py` stage 04 silently skips evidence verification.
- **Q5 (docs):** Confirm AGENTS.md:43 ("data/ is a symlink, all gitignored") should be rewritten — it's
  false on two counts after A1 (data/ is a real dir; gold/ is committed). And confirm README should be
  made internally consistent (line 83 stale vs line 126 new).
- **Q6 (gitignore):** Do you want a guard/comment that future `.parquet` files placed under `gold/`
  would still be caught by the global `*.parquet` (line 4)?
