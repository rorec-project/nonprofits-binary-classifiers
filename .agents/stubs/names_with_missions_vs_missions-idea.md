---
created: 2026-06-21
---

# Evaluating names vs. missions vs. names+missions as classifier input

## Context

The religious-vs-nonreligious classifier today reads **mission text only** (`LONGEST_MISSION`).
Organization names are never seen by the model — they are explicitly dropped at the data join
(`src/binary_classifier/data/load.py` keeps only `EIN2` + `NTEE_IRS` from the BMF). The question
is whether we should instead/also feed the **best EIN2-level name** from the `NonProfitData` final
panel, whether those names are too short to carry signal, whether to combine name+mission, and
whether the pipeline is flexible enough to run all three input modes. This plan records the
evidence-based answer and the concrete work to act on it.

The motivation is not a few F1 points. Churches/congregations are largely exempt from Form 990
(IRC §6033), so they are systematically missing from the mission corpus — which means the
mission-only pipeline structurally cannot classify a large, disproportionately religious slice of
the sector, and stage-09 prevalence is biased **low** on religion. Names exist for that slice.

## Findings (the answer)

All figures measured directly from `../NonProfitData` (read-only), not from the codebook.

### 1. Can the pipeline use the best EIN2-level names? — Yes.
`panel_final.parquet` already ships engineered name columns that are **already unique at EIN2 level**
(0% of EIN2s have >1 distinct value, so no dedup needed):
- `BEST_NAME_CASED` / `BEST_NAME_BARE_CASED` — proper-cased; the *bare* variant strips legal
  suffixes ("Inc/Incorporated"). **Use `BEST_NAME_BARE_CASED` for the LLM** (casing + de-noised).
- `CANONICAL_NAME` — all-caps and sometimes truncated → not suitable for the model.
- Coverage: **90.5%** of the 1,537,078 panel EIN2s have a `BEST_NAME` (vs **36.5%** with a mission).

### 2. Are the names too short? — No, not prohibitively.
`BEST_NAME` length: median **5 words / 36 chars** (p10=3, p90=8; only 0.5% are a single word;
76% are ≥4 words). Missions are longer (median 22 words). The short-text literature
(`docs/research/20260605-literature-short-text-classification.md`) warns of sparsity/ambiguity, but a
4–5-word nonprofit name is information-dense for *this* task: a crude keyword test already flags
**64% of NTEE-X (Religion) names** as religious vs 1–5% in other groups ("Trinity Baptist Church",
"First Assembly of God"). Length is not the risk — the **failure modes** named in
`docs/research/20260605-literature-religious-nonprofit-classification.md` (points 4–5) are:
secularized faith-founded names, faith-named secular orgs ("St. Mary's Hospital"), and
non-Christian / non-English under-coding.

### 3. Combine, or keep missions only? — Combine where both exist; names also unlock a population missions can't.
The repo's own literature separates two religiosity dimensions (Sider & Unruh typology):
**names carry religious *identity/affiliation*** (denomination, "church/temple/synagogue", saint
names); **missions carry religious *purpose/activity***. They are complementary, and the
metadata-fusion literature supports concatenation. Two distinct questions on two populations:
- **(a) Incremental** — on orgs that already have a mission, does adding the name help? Likely a
  small effect; this is where false positives like "Trinity Health" (religious identity, secular
  mission) live.
- **(b) Coverage** — for orgs with a **name but no mission**, can name-only classify them at all?
  This is the strategic win. **829,989 panel orgs** (and ~2.88M in the full BMF registry; NTEE-X
  only 9.4% mission-covered → ~394K religion-related orgs reachable only by name) have a name and
  no mission today.

Note: "name+mission" **degenerates to name-only** on the 63.5% of panel orgs without a mission. The
honest production framing is **"name + mission-when-available."**

### 4. Is the pipeline flexible enough? — Yes for any single text column; modest lift for the rest.
- The `cfg.field` mechanism (`config.py:571`, `config/religious_missions.yaml:11`) cleanly selects
  **one** upstream text column and renames it to `text` everywhere downstream
  (`load.py` field rename; `run_annotation.py:296-304`; annotators send `text` as the user message,
  `openai_annotator.py:75-103`, `vllm_annotator.py:77-124`). So **missions** work today and a
  **name** column works as soon as it is exposed upstream.
- Gaps to close: (i) names aren't joined into the input frame; (ii) prompts hardcode "mission
  statement" framing (`prompts/v1.txt`/`v2.txt`/`v3.txt`); (iii) **name+mission needs a derived
  concatenated field** (single-field design has no native multi-field path); (iv) outputs write to
  fixed paths so each arm needs its own config namespace; (v) the gold/silver pipeline was built on
  missions, so the coverage arm needs new labels. None require architectural change.

## Recommendation

1. **Keep mission-only as the baseline for mission-covered orgs**; treat name as a **coverage
   extension**, productionized as **"name + mission-when-available."**
2. **Decide the estimand first** (see Open Decisions) — this governs labeling and makes the arms
   comparable rather than measuring three different targets.
3. Run a **two-population evaluation** before any production rewire:
   - Arm (a) incremental, on **existing** gold (mission-covered).
   - Arm (b) coverage, on a **new** stratified human-coded sample of **no-mission** orgs.

## Implementation plan

### A. Data layer — expose names (`src/binary_classifier/data/load.py`)
- Read `BEST_NAME_BARE_CASED` (+ `EIN2`) from
  `../NonProfitData/data/processed/panel/merged/panel_final.parquet`, taking distinct EIN2 rows
  (already unique). Left-join onto the missions cross-section / BMF frame on `EIN2`.
- Carry the name column through alongside `mission_text` (do **not** overwrite it).
- Add a derived **`name_mission`** column built from a fixed template, e.g.
  `"Organization name: {name}\nMission statement: {mission}"`, that **gracefully omits the mission
  block when absent** (so this field self-degrades to name-only). This is what makes a single
  `cfg.field` serve the name+mission arm.

### B. Prompt generalization (`src/binary_classifier/annotate/prompts/`)
- Add one **input-agnostic** prompt variant (e.g. `v4_fielded.txt`) that classifies "a US nonprofit"
  from a labelled evidence block (Name / Mission, either may be absent) instead of assuming a
  mission statement. Keep the JSON output schema identical (`annotate/schema.py`) so scoring,
  bakeoff, and downstream stages are unchanged. Reuse this one prompt across all three arms.

### C. Config namespacing (one config per arm)
- `config/religious_missions.yaml` stays the missions-only baseline.
- Add `config/religious_names.yaml` (`field: BEST_NAME_BARE_CASED`) and
  `config/religious_name_mission.yaml` (`field: name_mission`), each with distinct
  `paths.processed_dir` / `paths.interim_dir` so stage outputs don't collide.

### D. Evaluation — reuse existing harness, two populations
- **Arm (a) incremental:** run the existing **bakeoff** scorer (`annotate/bakeoff_prompts.py`,
  metrics via `metrics.py`: κ, minority-F1 CI) on `prompt_dev + validation` (~200 rows, **keep
  `test` locked**) for three inputs: mission-only / name-only / name+mission. The existing gold
  already has `EIN2`, so names join in directly. Caveat to document: gold labels were coded from
  missions, so name-only is *penalized when the name is right and the thin mission was coded
  negative* — this arm bounds incremental value, it does **not** validate coverage.
- **Arm (b) coverage:** build a **new** stratified gold sample drawn from **no-mission** panel orgs
  (reuse the stage-01 sampling + `gold_to_code.csv` template pattern). Stratify/oversample
  religion-likely name tokens, non-Christian traditions, and identity-vs-purpose conflicts
  ("St. Mary's Hospital", "Trinity Health", "YMCA"), per the audit guidance in
  `docs/research/20260605-literature-religious-nonprofit-classification.md`. Code labels under the
  chosen estimand, then score name-only on this set.
- Produce one comparison table (arm × input × metric) — no auto-comparison exists today, so add a
  small read-only aggregation over each arm's `test_evaluation.json` / bakeoff results.

### E. Escalation (only if arm (a)/(b) are promising)
- Encoder refit reusing silver labels with name-augmented `text`, then the full stage 06→09 path on
  a namespaced config; quantify the prevalence correction from the newly reachable religious orgs.

## Verification

- **Data join:** spot-check that `BEST_NAME_BARE_CASED` is populated for ~90% of frame EIN2s after
  the join and that `name_mission` omits the mission block when the mission is null.
- **Prompt:** dry-run the new prompt through `openai_annotator`/`vllm_annotator` on ~10 records per
  input mode; confirm valid JSON against `annotate/schema.py`.
- **Arm (a):** bakeoff completes on ~200 mission-covered rows for all three inputs; inspect the
  name-vs-mission disagreement cases by hand (these are the substantive finding).
- **Arm (b):** new no-mission gold codes cleanly under the estimand; name-only metrics reported with
  bootstrap CIs; manually review false positives/negatives on the oversampled conflict strata.
- **No regression:** the baseline missions-only config reproduces current metrics unchanged.

## Open decisions

- **Estimand:** does positive mean religious **auspice/identity** (what names capture), religious
  **purpose/activity** (what missions capture), or **either signal positive**? This must be fixed
  before coding the new gold; it determines whether "Trinity Health" is a true or false positive and
  whether the three arms (and their stage-09 prevalence numbers) are comparable.
- **Scope:** panel filers (1.54M EIN2; lead numbers) vs. the full BMF registry (3.44M; upper-bound
  coverage win) — the latter extends prevalence beyond filers but needs the BMF name path.
- **How far this round:** stop at the two-population evaluation + memo, or continue into production
  rewire (sections A–C + E).
