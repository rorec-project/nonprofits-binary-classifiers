# Pipeline Review Against Best Practices

**Reviewed:** 2026-06-09
**Branch:** `refactor/harmonize-pipeline` (`1dd6e7f`)
**Scope:** The *built* pipeline — stages 01–04 (`src/binary_classifier/`, `scripts/`,
`config/religious_missions.yaml`) — reviewed against (a) the project's own research
synthesis in `.agents/docs/` and (b) external authoritative sources (OpenAI API docs via
Context7; the primary literature cited by the synthesis: Pangakis & Wolken, Gilardi et al.,
Snorkel/WRENCH, scikit-learn `model_evaluation`, QuaPy).
**Type:** Read-only review. The only artifact produced is this document; no source or data
was modified.
**Out of scope as "defects":** training, evaluation, inference-at-scale, and visualization
(roadmap points 3–6) are explicitly *not built*. They are not criticized here as missing;
§5 raises only forward-looking guidance for when they are built.

---

## 1. Headline

Run end-to-end today (`uv run python scripts/run_pipeline.py`), the pipeline executes
without error but **all three of its quality-control mechanisms are silently inert**, and
it freezes an unvalidated silver-label file:

1. **Bake-off scoring (stage 02) produces no scores.** `run_pipeline._run_stage` calls
   every stage as `func(cfg, registry)` and only ever forwards `limit` to stage 03
   (`scripts/run_pipeline.py:69-72`). `human_labels_path` is never passed, so
   `run_bakeoff` falls to the `{"note": "no human labels provided — skipping scoring"}`
   branch (`bakeoff_prompts.py:113-118`). Model/prompt *selection* — the stated purpose of
   the stage — has nothing to select on.
2. **Prompt provenance collapses (stage 03).** Every annotation is stored with an empty
   `prompt_id`, so the three prompt variants are indistinguishable and resume is broken
   (D1 below).
3. **The agreement gate never runs (stage 04).** `human_validation_path` is never passed
   either, so `run_quality_check` takes the `agreement = None` branch
   (`agreement.py:103-105`) and then freezes the file *unconditionally*
   (`agreement.py:108-110`). Even when a human file *is* supplied, a sub-85 % result only
   logs a warning and still freezes (`agreement.py:94-99`).

Compounding this: per `AGENTS.md`, the gold/validation manifests are "EIN2 lists +
sampling metadata, **not** text/labels." So the human-label artifact that stages 02 and 04
depend on is **neither produced by the pipeline nor given a defined path/schema** anywhere.
The validation-first workflow that the entire `.agents/docs/` synthesis is built on
(Pangakis & Wolken 2024/2025: "Automated Annotation with Generative AI Requires
Validation") is designed in but not wired up.

This is the most important result of the review: the scaffolding is good, but the loop that
makes weak supervision trustworthy is currently open.

---

## 2. What the pipeline does well

Credit where due — these are real strengths and match the research synthesis:

- **Config-driven and typed.** A single Pydantic `BinaryClassifierConfig` →
  `PathRegistry` is the source of truth for seeds, paths, thresholds, and the model slate
  (addresses legacy audit R-14). Retasking is a YAML copy.
- **Long/tidy label store, one row per `(EIN2, source_id)`** (`schema.py`) — exactly the
  weak-supervision-ready shape the synthesis prescribes; the natural input for majority
  vote, Dawid–Skene, and CROWDLAB.
- **Abstention is a first-class label** (`ambiguous_review`, `insufficient_information` →
  NaN). This matches Snorkel-style `ABSTAIN` and the "never guess" guidance in the prompts.
- **Schema captures evidence spans + domain codes**, aligning with the codebook design in
  `20260606-tech-synthesis-map.md` ("require exact evidence spans for any positive").
- **Seeds and `EIN2` are carried through** (addresses R-02, R-04); `raw_response` is
  persisted (good for reproducibility/debugging).
- **Stage 01 ships assertions** (`scripts/01_build_sample.py:62-110`): EIN2 uniqueness,
  all 26 strata present, positive-share band, disjoint human splits. This is the kind of
  guard the rest of the pipeline lacks.
- **Drop-in comparison arms** for Dawid–Skene (`crowd-kit`) and CROWDLAB (`cleanlab`) are
  stubbed in `aggregate.py` — the right libraries, honestly marked as stubs.
- **`TrainingConfig.metric_for_best_model = "pr_auc"`** already encodes the correct
  imbalanced-metric default from `20260606-tech-imbalanced-text-evaluation.md`.

---

## 3. Defects in built code (stages 01–04)

Tiered by severity. Each: location · consequence · one-line fix.

### High

**D1 — Stage 03 drops `prompt_id`, collapsing the model×prompt matrix and breaking resume.**
`run_annotation`'s factory hardcodes `prompt_id=""` (`run_annotation.py:102,109`), so every
stored record has `source_id = "{model_id}__"` and `prompt_id = ""`, regardless of which
prompt produced it. But the resume filter reconstructs the key as `f"{m}__{p}"` with the
real prompt stem (`run_annotation.py:184`) — a guaranteed mismatch.
*Consequence:* (a) resume never matches → every re-run re-annotates everything **and**
appends duplicates (`already_done` collides across prompts); (b) the three prompt variants
are indistinguishable in the store, destroying the inter-prompt reliability signal the
synthesis calls for ("run at least two prompt variants … use disagreement as an uncertainty
feature", `20260606-tech-llm-weak-supervision-noisy-labels.md`). Note `bakeoff_prompts.py:92`
does this correctly (`source_id = f"{model_id}__{prompt_id}"`), proving it's an oversight.
*Fix:* thread the real `prompt_id` (= prompt stem) into the annotator factory and
`LabelRecord`.

**D2 — Validation and the agreement gate are inert in every default invocation.**
See §1. Beyond the unwired orchestrator: even the standalone scripts default
`--human-labels` / `--human-validation` to `None` (`02_…py:42-47`, `04_…py:31-36`), there is
no defined artifact carrying `human_label`, and the `run_quality_check` docstring claims
`human_validation_path` "Defaults to the validation manifest" while the code leaves it
`None` (doc/code mismatch). The gate also warns-but-freezes on failure.
*Fix:* define a human-coding artifact (path + schema, `EIN2,human_label`), wire it through
`run_pipeline`, and make the gate *block* the freeze (raise / non-zero exit) when agreement
< threshold or when no human labels are present.

**D3 — Silent synthetic-data fallback can masquerade as a real run.**
`load_missions` silently swaps in ~1 000 rows of fake data when the upstream parquet is
absent, logging only at `INFO` (`load.py:52-54`). A stage-01 run then builds a "20 000-row
silver pool" from 1 000 synthetic rows and writes manifests indistinguishable from a real
build. The temp dir is also never cleaned (the "caller may delete" comment has no caller).
Stage 03 has *no* such fallback (`run_annotation.py:87` reads the parquet directly and will
raise), so the two stages disagree about whether synthetic data is allowed.
*Fix:* gate synthetic data behind an explicit `--synthetic` flag (or env var); otherwise
fail loudly. Stamp manifests with a `data_source` field. Clean up the temp dir.

### Medium

**D4 — Agreement is raw percent, not the imbalanced bundle; docstring overclaims.**
`run_quality_check` computes only `agree / len(valid)` (`agreement.py:86-87`), yet the
module docstring and the stage-04 CLI help both advertise "Krippendorff alpha"
(`agreement.py:3`, `04_…py:16-17`) — none is computed (doc/code mismatch). Raw agreement on
a rare-positive task is misleading: `20260606-tech-imbalanced-text-evaluation.md` is
explicit that accuracy/agreement alone hides minority-class failure.
*Fix:* report the imbalanced bundle on the validation overlap — confusion matrix,
minority precision/recall, MCC, balanced accuracy, PR-AUC — with bootstrap CIs (the config
already promises "bootstrap confidence intervals", `config.py:80`). Add Cohen's κ /
Krippendorff α *as coder-agreement diagnostics*, noting both are prevalence-sensitive for
classifier scoring (per that same doc) so they supplement, not replace, the confusion
matrix.

**D5 — The closed-reference annotator does not honor `guided_json: true`.**
`OpenAIAnnotator` uses `response_format={"type": "json_object"}` (`openai_annotator.py:77`).
Per OpenAI's docs, `json_object` is "older JSON mode" — it guarantees *valid JSON* but
**not** schema conformance; `json_schema` with `strict: true` is the Structured Outputs
mode that "ensures the model's output matches a supplied JSON schema" and is "preferred for
models that support it." Meanwhile `VLLMAnnotator` *does* constrain output
(`extra_body={"guided_json": …}`, `vllm_annotator.py:81`). So the closed and open paths are
**not** equivalently constrained, contradicting both the config flag and the
`build_json_schema` docstring ("passed to both OpenAI `response_format` and vLLM
`guided_json`"). `build_json_schema` itself is dead code — defined but never called, so the
inline vLLM schema and this function can drift.
*Fix:* switch the OpenAI path to `response_format={"type":"json_schema","json_schema":{…,
"strict":true}}` and source both annotators from the single `build_json_schema()`.

**D6 — The annotation store is O(n²) and won't scale to the full matrix.**
`AnnotationStore._save` rewrites the *entire* CSV on every `append_many`
(`schema.py:249-255`), and `run_annotation_matrix` checkpoints every 100 records. For the
full run (~20 000 × 3 models × 3 prompts ≈ 180 000 rows, each carrying `raw_response`),
that is ~1 800 full rewrites of a growing multi-hundred-MB file. Resume also rebuilds the
"already done" set with `.iterrows()` over the whole frame (`run_annotation.py:178-180`).
*Fix:* append-only writes (open in `"a"` mode / parquet partitions), and build the resume
key set vectorised (`set(zip(df.EIN2, df.source_id))`).

**D7 — Hardcoded CWD-relative paths bypass the `PathRegistry`.**
`data/annotation_store.csv` (`run_annotation.py:77`), `data/bakeoff_labels.csv` and
`results/bakeoff_results.json` (`bakeoff_prompts.py:65-67`) are written relative to the
current working directory, not via `registry.data_dir` / `registry.results_dir`. This
contradicts the "no string concatenation … single source of truth for paths" design and
can land the stores *outside* the cloud-synced symlinked locations the manifests use.
*Fix:* route every output through the registry (add `annotation_store` / `bakeoff_*`
properties).

**D8 — Reproducibility of annotations is overstated.**
The config and docstrings describe temperature-0 + seed as giving "reproducible /
deterministic annotations" (`config.py:90-94`). For closed APIs this is best-effort only:
OpenAI's own docs say `system_fingerprint` exists "to track backend changes that might
affect the determinism of model outputs" used together with `seed` — i.e. determinism is
not guaranteed across backend changes, and `20260606-tech-llm-weak-supervision-noisy-labels.md`
explicitly flags "reproducibility risk from model updates; pin exact model version."
`closed_reference: gpt-4o-mini` is a *floating alias*, not a pinned snapshot.
*Fix:* pin model IDs to dated snapshots and record `system_fingerprint` per call; soften
"deterministic" to "low-variance / best-effort"; keep persisting `raw_response` (already
done). (Separately, verify each open-weight ID in the slate resolves before a run — the
slate should reference released checkpoints.)

**D11 — Evidence spans are collected but never enforced (a missing hallucination guard).**
The prompts demand verbatim evidence (`v1.txt:43`: "`evidence_spans` must be verbatim
substrings; at least one span for `religious` or `nonreligious`") and the schema stores
`evidence_spans` + `domains_present` — but **no stage consumes either column** (confirmed:
they appear only in prompts, `schema.py`, and the annotators). The cheap, high-value QC the
synthesis explicitly calls for ("require exact evidence spans for any positive",
`20260606-tech-synthesis-map.md`) is therefore absent: nothing checks that each span is a
real substring of the text. A model can assert `religious` with a fabricated quote and the
label is accepted at full weight.
*Fix:* at aggregation/QC, verify every `evidence_span` is a verbatim substring of the
source text; abstain or flag-for-review any positive/negative label whose spans don't match,
and cross-check `domains_present` against the label.

### Low

**D9 — No tests.** There is no `tests/` directory. The deterministic, easily-unit-tested
core — `compute_quality_score`, `assign_tier`, `majority_vote` tie/all-abstain handling,
the resume key, `_allocate_stratum_targets` floor/cap math — has no regression coverage,
which is what let D1 land undetected. *Fix:* add `pytest` cases for these pure functions.

**D10 — Dead fields and minor inconsistencies.**
- `reason` is parsed from the LLM response (`openai_annotator.py:123`) but **no prompt asks
  for it**, so it is always `None`; the prompts emit `boundary_notes` instead.
- `is_positive_enriched` is `lexicon OR rescue` in the silver pool but `lexicon`-only in the
  gold set (`sample.py:102-108` vs `:200`) — intentional? It should be documented.
- `CANARY_EIN2` is synthetic placeholders (`run_annotation.py:35-41`), so drift detection
  is a no-op against real data.
- `_score_vs_human` in the bake-off reports accuracy/precision/recall/F1 only
  (`bakeoff_prompts.py:217-225`), missing the MCC / balanced-accuracy already mandated for
  the imbalanced task.

---

## 4. Design risks baked in now (not yet defects, but they compound)

Framed per the repo persona — flag, don't silently "fix" what may be intentional.

**R1 — The Q≥3 tier filter + positive-enrichment create a train/deploy distribution gap
that nothing downstream corrects.** Silver and gold are drawn only from HIGH+MEDIUM missions
(`sample.py:96`, `:196`) and enriched to ~35 % positive (`_TARGET_POSITIVE_SHARE`), while
deployment must score the *whole* population, including LOW/bare-label missions (handed to a
separate rule layer). So the labeled distribution differs from the target distribution in
both text richness and class prevalence. `inclusion_prob` is computed and written to every
manifest (`sample.py:160-161`, `:249-250`) — but **no code consumes it**. The stated
downstream goal of prevalence estimation
(`20260606-tech-calibration-quantification-prevalence.md`) will be biased unless the labeled
sample is reweighted by `inclusion_prob` and corrected with quantification (classify-and-
count / PACC / QuaPy). *Recommendation:* document this as a deliberate design choice and
commit to the mitigation (reweighting + quantification) before any prevalence number is
reported.

**R2 — Lexicon-driven enrichment risks circularity in evaluation.** The same
`RELIGIOUS_LEXICON` that enriches the positive pool is also the basis of `apply_rule_label`
(the high-precision rule label) and shapes which positives reach the gold set. A gold set
over-weighted toward lexicon-positive cases will *inflate* apparent agreement/accuracy on
easy cases and under-sample the hard borderline negatives (saint-named secular,
faith-heritage, "spiritual"-not-religious) that the prompts and `_classify_boundary` go to
some lengths to capture. *Recommendation:* ensure the frozen test set's positives are not
selected by the lexicon (or stratify and report lexicon-hit vs lexicon-miss slices
separately), and honor `inclusion_prob` when scoring.

**R3 — Abstention/disagreement is dropped rather than reported.** `majority_vote` routes
ties and all-abstain to `NaN` (`aggregate.py:39-61`) and QC logs an abstain rate, but there
is no Snorkel-style coverage/overlap/conflict report and no concrete human-review queue for
abstains/ties/LLM-vs-rule conflicts. The synthesis stresses "human-in-the-loop should target
disagreement, not random cleanup." *Recommendation:* emit a conflict report and write
abstain/tie/conflict rows to a review file.

---

## 5. Forward-looking guidance for roadmap stages

Not defects — guidance for when points 3–6 are built, drawn from `.agents/docs/`:

- **Evaluation:** report the full imbalanced bundle with bootstrap CIs on the frozen test;
  treat ROC-AUC as secondary. (`pr_auc` default already correct.)
- **Calibration before prevalence:** Platt / temperature / isotonic on a held-out
  calibration split (`netcal`, `gpleiss/temperature_scaling`).
- **Prevalence:** classify-and-count correction / QuaPy (CC/ACC/PCC/PACC), report
  uncertainty; combine with `inclusion_prob` reweighting (R1).
- **Weak supervision:** actually wire the `crowd-kit` / `cleanlab` arms and compare against
  majority vote and a Snorkel `LabelModel`; report LF/prompt coverage, overlap, conflict.
- **Downstream inference:** if classifier outputs feed regressions/estimates, apply
  DSL or prediction-powered inference (Egami et al.; Angelopoulos et al.) — high accuracy
  alone does not guarantee unbiased estimates.
- **Encoder training:** plain fine-tune baseline first with fixed seeds + early stopping on
  *human* validation labels; add noisy-label methods only if they beat that on held-out
  F1/MCC (Zhu et al. 2022; Pangakis & Wolken 2024).

---

## 6. Prioritized actions

1. **Close the validation loop (D1 + D2).** Fix `prompt_id` provenance; define and wire the
   human-label artifact; make the agreement gate block the freeze. Without this, every
   silver label the pipeline emits is unvalidated.
2. **Make synthetic data opt-in and loud (D3).** A real run must never silently use fake data.
3. **Honor `guided_json` on the closed model + de-dupe the schema (D5).**
4. **Strengthen QC: add the imbalanced metric bundle + bootstrap CIs, enforce
   evidence-span verbatim matching, and drop the false "Krippendorff alpha" claim
   (D4 + D11).**
5. **Add tests for the pure core (D9)** — the cheapest insurance against another D1.

Then address the scale (D6), path-registry (D7), and reproducibility (D8) items before the
first full 20 k run, and document the sampling-bias mitigation (R1) before reporting any
prevalence figure.
