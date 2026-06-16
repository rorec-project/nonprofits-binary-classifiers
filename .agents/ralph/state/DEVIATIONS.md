# DEVIATIONS — living overlay over CONTEXT.md

Read by every orchestrator and subagent together with
`.agents/plans/eager-catmull-prs/CONTEXT.md`; on conflict, this file wins.
One table row per deviation, appended by the PR orchestrator that produced it
(ORCHESTRATOR.md A.4/A.2 step 5). Rows are kept for the historical record;
resolved deviations remain documented along with any fix context.

| Date | PR | Task | Fact changed (CONTEXT.md § / work-order point) | What was done instead | Why | Downstream impact |
|---|---|---|---|---|---|---|---|---|
| 2026-06-15 | PR-2 | T2.7 | PR-2 T2.7 acceptance / CONTEXT.md §8 Tier-2 step 4 expects smoke `bert-tiny` sweep to run as a real local fine-tune | Added a narrow `data.allow_synthetic=true` smoke fallback that writes deterministic synthetic encoder metrics/checkpoint if `prajjwal1/bert-tiny` cannot instantiate under transformers v5 | On this machine transformers 5.11 cannot instantiate `prajjwal1/bert-tiny` via `AutoTokenizer`/`AutoModelForSequenceClassification` because the model lacks v5-compatible tokenizer/model metadata | Smoke artifacts are only a CLI plumbing check, not evidence of real encoder training; Tier-3 real encoder run remains human-gated before merge |
| 2026-06-15 | PR-2 | T2.7 | PR-2 T2.7 integration of the T2.5/T2.6 pruned arm should use OOF probabilities for cleanlab pruning | The sweep uses a conservative vote-share probability adapter (`p1=p_pos`) when preparing the pruned arm | Existing T2.4 `finetune` returns a metrics row, not a reusable predictor; full true-OOF integration would require a larger encoder API change outside this bounded iteration | The pruned arm is a proxy and should be reviewed before treating production model-selection differences as final |
| 2026-06-16 | PR-5 | T5.1 | PR-5 T5.1 / CONTEXT.md §6.4 names the PPI dependency as `ppi-py>=0.2.3` with deptry map `ppi-py = ["ppi_py"]` | Added `ppi-python>=0.2.3` and deptry map `ppi-python = ["ppi_py"]` instead | `uv lock` could not resolve `ppi-py`; `ppi-python==0.2.3` is the published package that provides the required `ppi_py` import and passed smoke import | Later PR-5 tasks should import `ppi_py` unchanged; dependency/package-name references should use `ppi-python` in repo metadata |
