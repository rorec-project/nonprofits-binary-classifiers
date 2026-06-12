# PR-0 — `chore/deps-transformers-v5` (dependency modernization)

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `chore/deps-transformers-v5` |
| Depends on | nothing (first PR) — but §9.9 applies: the local toolchain arch fix is T0.1 step (2) |
| Blocks | every later PR (PR-1 … PR-7) |
| Spec blocks implemented | §6.4 (PR-0 part: transformers/torch/sentencepiece pins, vllm → `serve` extra, `tracking` extra, pytest markers) |
| Ralph state | `.agents/ralph/state/pr-0.md` + `pr-0.status` |

**Pre-flight (first iteration):** confirm the existing suite is green as a baseline
(`uv run pytest` — the `slow`/`network` markers do not exist until T0.1 lands);
switch to / create the branch.

**Human checkpoints:** T0.2 needs HF Hub network access once
(`uv run pytest -m network`). Any torch-floor fallback in T0.1 is a DEVIATION the
human must review at the PR boundary — it changes the UCloud expectations
(`docs/RUNNING_ON_UCLOUD.md`) for every later PR.

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-0 — `chore/deps-transformers-v5` (dependency modernization)

**Objective**: move the environment to the June-2026 stack with zero behavior change
to stages 01–04.

**T0.1 — pyproject + lock** (owns `pyproject.toml`, `uv.lock`)
Operations: (1) Pre-check: `grep -rn "import vllm" src/ scripts/` MUST return zero
matches (re-verifying §3.2's claim) before moving vllm to the extra. (2) Environment
arch check: `uv run python -c "import platform; print(platform.machine())"` — on
Apple Silicon this MUST print `arm64`; if it prints `x86_64`, the local uv/python is
a Rosetta x86_64 build and **torch ≥2.3 has no macOS x86_64 wheels — resolution will
fail** (observed on this machine on 2026-06-12: uv resolved platform
`macosx_26_0_x86_64` and torch errored). Fix the toolchain first (`uv python install`
a native arm64 CPython and recreate `.venv`) — do NOT paper over it with
`tool.uv.required-environments`. (3) Apply §6.4 PR-0 edits exactly (transformers,
torch, sentencepiece, vllm → `serve` extra, `tracking` extra, pytest markers). Run
`uv lock`, `uv sync`. If torch 2.7 still fails to resolve on a correctly-arch'd local
platform, apply the documented fallback (floor 2.6.0) and record it as a DEVIATION.
Acceptance: `uv lock` clean; `uv sync` clean; `uv sync --extra serve` resolves;
`uv run pytest -m "not slow and not network"` green (existing 13 test files);
`uv run python -c "import transformers, torch, sentencepiece;
print(transformers.__version__, torch.__version__)"` prints 5.8+ / 2.7+ (or fallback).

**T0.2 — tokenizer regression guard** (owns `tests/test_tokenizer_sanity.py`) [parallel-ok]
Operations: new test file, `@pytest.mark.network`, asserting for BOTH
`microsoft/deberta-v3-base` and `answerdotai/ModernBERT-base`:
`ids = tok("hello world")["input_ids"]; ids[0] == tok.cls_token_id and
ids[-1] == tok.sep_token_id` (guards the v5.0–5.3 DeBERTa regression, §4.5).
Acceptance: `uv run pytest -m network tests/test_tokenizer_sanity.py` green (network
required, run once locally).

**T0.3 — UCloud doc note** (owns `docs/RUNNING_ON_UCLOUD.md`) [parallel-ok]
Operations: add a "GPU environment" note: B200 = sm_100 ⇒ torch ≥2.7 from the cu128
index; verify with `python -c "import torch; print(torch.version.cuda,
torch.cuda.is_bf16_supported())"`; `uv sync` without `--extra serve` for training jobs;
`--extra serve` only when serving annotation models.
Acceptance: doc renders; no other file touched.

**PR-0 gate**: T0.1–T0.3 reports green; full Tier-1 (§8) green.

