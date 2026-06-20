# Plan: Serve Gemma (annotation) + a coding model on UCloud B200, flawlessly

## Context

We run the classifier pipeline's open-weight arm (stage 02 bake-off, stage 03
annotation) on a UCloud **NVIDIA B200** (Blackwell, `sm_100a`, 183 GB) by
serving `google/gemma-3-27b-it` through a local vLLM OpenAI-compatible server.
The vLLM server **fails to start**, which blocks both stages. Separately, we
also want an interactive **coding model** served on the same node for dev
assistance (Opencode), but the serving scripts only understand a single model.

### Root cause (verified from `vllm_annotation.log`)

vLLM loads Gemma and finishes `torch.compile`, then dies in engine warmup:

```
[1/4] /opt/cuda/bin/nvcc ... -gencode=arch=compute_100a,code=sm_100a ... sampling.cu ...
/bin/sh: 1: /opt/cuda/bin/nvcc: not found
RuntimeError: Engine core initialization failed.
```

- FlashInfer has **no prebuilt sampling kernel for `sm_100a`**, so it
  **JIT-compiles** `sampling.cu`/`renorm.cu` at warmup. JIT needs **`nvcc`**.
- PyTorch/vLLM wheels ship the CUDA *runtime* but **not the compiler**.
  FlashInfer falls back to a hardcoded `/opt/cuda/bin/nvcc` that does not exist
  here, and `CUDA_HOME` is empty.
- The node **does** have a complete CUDA 12.8 toolkit (supports `sm_100`),
  exposed the **UCloud-native way** as an Lmod/EasyBuild module: `module load
  CUDA/12.8.0` sets `CUDA_HOME`/`PATH`/`CPATH`/`LD_LIBRARY_PATH` and puts nvcc
  12.8 on `PATH`. It's just not loaded by default. (`uv.lock` is now vLLM 0.22.1
  / flashinfer-python 0.6.11 / torch 2.11 **cu128**, so 12.8 is the matching nvcc.)

### Platform alignment (verified live on this B200 node + UCloud docs)
- **Lmod is active** (`LMOD_CMD=/opt/lmod/lmod/libexec/lmod`); UCloud's
  documented pattern is to select the **easybuild** folder via the job's
  optional **"Modules path"** parameter, which adds the EasyBuild tree to
  `MODULEPATH`. We don't have to rely on that being set — `module use
  /opt/easybuild/ubuntu-24.04/<amd|intel>/modules/all` adds it ourselves.
- `module -t avail CUDA` here lists `12.1.1, 12.4.0, 12.6.0, 12.8.0`. Only
  **12.8** supports Blackwell `sm_100`; 12.6 and older cannot compile
  `compute_100a`. The GPU node is **AMD EPYC + 8× B200 (192 GB)**, so the `amd`
  EasyBuild tree is the correct one. Verified: `module use … && module load
  CUDA/12.8.0` → `CUDA_HOME=…/CUDA/12.8.0`, `nvcc` release 12.8.
  (First load may need `module --ignore_cache load` if the Lmod cache is stale —
  exactly what the error told us the first time.)
- **`$HOME` is ephemeral overlay; `/work` is persistent wekafs** (confirmed via
  `mount`). FlashInfer's JIT cache (`~/.cache/flashinfer`) and vLLM's
  `torch.compile` cache (`~/.cache/vllm`) are therefore **wiped every job**, so
  the ~5-min Blackwell compile repeats on every launch unless redirected to
  `/work`. Persisting them on the data drive is part of the fix, not a nicety.

### Decisions taken
- **Fix:** provide nvcc and keep FlashInfer (fast kernels; one-time JIT, cached).
- **Topology:** serve both models **concurrently on two ports** on one B200.
- **Scope:** unblock Gemma **and** generalize the serving layer + update docs.

---

## Part A — Unblock vLLM on the B200 (provide nvcc, keep FlashInfer)

Goal: load the CUDA 12.8 module so FlashInfer's JIT finds nvcc, persist the
compiled kernels on `/work`, and keep everything pinned to torch's cu128.

1. **Load CUDA via the module system in `utils/init.sh`** (where `env.sh` is
   generated, `init.sh:106-115`). Add a small, idempotent block that the
   generated `env.sh` will replay (so every shell, `run.sh`, and `serve-llm.sh`
   inherit a correct CUDA env without re-deriving paths):
   ```sh
   # Blackwell (B200/sm_100) FlashInfer JIT needs nvcc; the torch/vLLM wheels
   # don't ship it. Load EasyBuild CUDA (matches torch's cu128). UCloud's
   # "Modules path" job param may already add the EB tree; we add it defensively.
   CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"   # overridable in .env
   if [ -f /opt/lmod/lmod/init/bash ]; then
     source /opt/lmod/lmod/init/bash
     for p in /opt/easybuild/ubuntu-24.04/amd/modules/all \
              /opt/easybuild/ubuntu-24.04/intel/modules/all; do
       [ -d "$p" ] && module use "$p"
     done
     module load "$CUDA_MODULE" 2>/dev/null || module --ignore_cache load "$CUDA_MODULE" || true
   fi
   [ -z "${CUDA_HOME:-}" ] && echo "WARNING: $CUDA_MODULE not loaded; vLLM on B200 will fail FlashInfer JIT." >&2
   ```
   The module sets `CUDA_HOME`/`PATH`/`CPATH`/`LD_LIBRARY_PATH` correctly
   (verified), so we don't hand-maintain include/lib paths. Add a `.env` knob
   `CUDA_MODULE` (default `CUDA/12.8.0`) for forward-compat with newer CUDA.

2. **Persist GPU caches on `/work` and wire the annotation URL** — also write
   into `env.sh` (these are why warm starts are fast and the pipeline targets the
   right port):
   ```sh
   export FLASHINFER_CACHE_DIR="${DATA}/flashinfer-cache"  # JIT kernels persist
   export VLLM_CACHE_ROOT="${DATA}/vllm-cache"             # torch.compile persists
   export VLLM_BASE_URL="http://127.0.0.1:${LLM_ANNOTATE_PORT}/v1"  # see Part B
   ```
   `mkdir -p` the two cache dirs alongside the existing `mkdir -p` at
   `init.sh:142`. Verify the exact FlashInfer cache env-var name against the
   installed flashinfer-python 0.6.11 at implementation time; **fallback** if it
   differs: symlink `~/.cache/flashinfer` → `${DATA}/flashinfer-cache` in `init.sh`.

3. **Optional portable fallback (non-UCloud only):** for an environment without
   the EasyBuild CUDA module, add `nvidia-cuda-nvcc-cu12>=12.8,<13` to the
   `serve` extra (`pyproject.toml:28`). On UCloud the module path is primary and
   sufficient — it ships the full runtime/headers FlashInfer needs; the pip nvcc
   alone does not, so keep it documented as fallback-only.

4. First start spends ~5 min JIT-compiling FlashInfer's sm_100 kernels into
   `FLASHINFER_CACHE_DIR`; subsequent starts and **future jobs** reuse the cache
   (which is why it must live on persistent `/work`, not ephemeral `$HOME`).

---

## Part B — Generalize serving: two roles, two ports, one B200

Today `utils/serve-llm.sh` only knows `LLM_CODING_MODEL` on `LLM_CODING_PORT`,
and `factory.make_annotator` builds `VLLMAnnotator` with no `base_url`, hardcoding
the pipeline client to `http://127.0.0.1:8000/v1`
(`annotators/vllm_annotator.py:48`, `factory.py:70-79`).

1. **Config knobs — `.env.example` (and live `.env`)** `:19-26`. Replace the
   single coding block with role-based vars (keep `LLM_API_KEY`, `LLM_TENSOR_PARALLEL`):
   ```sh
   # Annotation model — served for pipeline stages 02/03 (the vLLM open-weight arm).
   # Model id itself stays the single source of truth in config/religious_missions.yaml.
   LLM_ANNOTATE_PORT=8000
   LLM_ANNOTATE_MAX_MODEL_LEN=8192
   # Coding model — interactive dev assistant (Opencode).
   LLM_CODING_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
   LLM_CODING_PORT=8001
   LLM_CODING_MAX_MODEL_LEN=8192
   # When both run on one GPU, each server's share of VRAM:
   LLM_GPU_MEM_UTIL=0.45
   ```
   Note: the **annotation** model id is *not* duplicated here — `serve-llm.sh
   annotate` reads it from the YAML's vLLM candidate (same one-liner `init.sh:168`
   already uses), which also fixes the "served model must match `spec.id`"
   fragility, since one source feeds both the server and the annotator.

2. **`utils/serve-llm.sh` → role-parameterized.** Refactor into a reusable
   `serve_one <role> <model> <port> <max_len> <gpu_mem_util>` that keeps the
   current idempotent nohup-launch + `/health` wait, but per-role log
   (`vllm-<role>.log`) and per-role pgrep match. CLI:
   - `serve-llm.sh annotate` → model from YAML, port `LLM_ANNOTATE_PORT`,
     `--gpu-memory-utilization` high (~0.9, sole tenant).
   - `serve-llm.sh coding` → `LLM_CODING_MODEL`, port `LLM_CODING_PORT`, ~0.9.
   - `serve-llm.sh both` → start annotate then coding, each at `LLM_GPU_MEM_UTIL`
     (~0.45) so both fit in 183 GB on different ports.
   - `--stop [role]` / `--status [role]` operate per-role (default: all known roles).
   - Pass `--served-model-name` (the served id) and `--gpu-memory-utilization`.
   - Source `env.sh` first so CUDA_HOME/PATH from Part A are present.
   - Keep the existing Public-IP endpoint printout, generalized per role/port.

3. **Thread the annotation base URL into the client.** Make
   `factory.make_annotator` pass `base_url` to `VLLMAnnotator` from
   `os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")`
   (`factory.py:70-79`). `VLLMAnnotator` already accepts `base_url`
   (`vllm_annotator.py:48`); the default keeps non-UCloud/local use working.
   `env.sh` (Part A) exports `VLLM_BASE_URL` from `LLM_ANNOTATE_PORT`, so the
   pipeline targets the annotation server regardless of which port it's on.

4. **`utils/devenv.sh` / Opencode** already key off `LLM_CODING_MODEL` /
   `LLM_CODING_PORT` (`devenv.sh:148-167,194`) — they keep working once the port
   default moves to 8001. Verify the generated `opencode.json` model name and
   `UCLOUD_LLM_BASE_URL` reflect the coding port.

5. **Docs.** Update `docs/RUNNING_ON_UCLOUD.md` (§ around the `serve-llm.sh` row
   `:165`, the B200/cu128 note `:215`, and the serve step `:250`) and
   `docs/UCLOUD_CODING_LLM.md` (single-model assumptions, `:14-16,104`) for: the
   `module load CUDA/12.8.0` requirement on B200 (and the optional **"Modules
   path" → easybuild** job-submission parameter, which our scripts also handle
   defensively), the role-based `serve-llm.sh`, concurrent two-port serving +
   VRAM split, and the new `.env` knobs (`CUDA_MODULE`, `LLM_ANNOTATE_*`,
   `LLM_CODING_*`, `LLM_GPU_MEM_UTIL`).

---

## Files to modify
- `utils/init.sh` — CUDA detection + `env.sh` exports + cache dirs (Part A).
- `pyproject.toml` — optional `nvidia-cuda-nvcc-cu12` in `serve` extra (Part A.2).
- `.env.example` (and the live project-drive `.env`) — role-based serving knobs
  + optional `CUDA_MODULE` override.
- `utils/serve-llm.sh` — role-parameterized serving (`annotate|coding|both`).
- `src/binary_classifier/annotate/annotators/factory.py` — pass `base_url` from
  `VLLM_BASE_URL` to `VLLMAnnotator`.
- `docs/RUNNING_ON_UCLOUD.md`, `docs/UCLOUD_CODING_LLM.md` — document the above.

## Reused, not reinvented
- YAML-derived vLLM model id one-liner (`init.sh:168`) → also drives `annotate`.
- `VLLMAnnotator(base_url=...)` already exists (`vllm_annotator.py:40-73`).
- Existing idempotent launch / `/health` wait / Public-IP printout in
  `serve-llm.sh` — kept, just parameterized by role.

## Verification (on the B200 job)
1. `bash utils/init.sh` (or `source env.sh`); confirm `module list` shows
   `CUDA/12.8.0`, `echo "$CUDA_HOME"` is the EB toolkit path, and `nvcc
   --version` → release **12.8**. Then `uv sync --extra serve`.
2. `bash utils/serve-llm.sh annotate` → first run JIT-compiles FlashInfer (watch
   `outputs/.../vllm-annotate.log`), then `/health` returns; cache lands in
   `${DATA}/flashinfer-cache`. `curl -s localhost:8000/v1/models` lists
   `google/gemma-3-27b-it`.
3. Pipeline smoke: a tiny `--limit` stage-03 (or stage-02 bake-off) run that hits
   the vLLM arm and returns parsed `LabelRecord`s — confirms `VLLM_BASE_URL`
   wiring end-to-end.
4. `bash utils/serve-llm.sh both` → ports 8000 **and** 8001 both healthy;
   `nvidia-smi` shows both models resident within 183 GB. `opencode` connects to
   `:8001`.
5. Restart `serve-llm.sh annotate` → comes up in well under a minute (no
   recompile), proving the persistent cache works across restarts/jobs.
6. `bash utils/serve-llm.sh --status` / `--stop` behave per-role.
