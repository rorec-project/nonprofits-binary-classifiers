# Coding LLM in UCloud via vLLM

AI-assisted coding for interactive UCloud sessions, layered onto the base workflow via the `devenv.sh` overlay. The LLM stack is never touched by batch jobs.

## Architecture

```
Opencode (laptop)                  Opencode (UCloud SSH session)
       │                                       │
       │ UCloud Public IP                      │ localhost
       │ (static, allocated once in §2 step 5) │
       └───────────────────┬───────────────────┘
                           ▼
        vLLM :LLM_CODING_PORT (default 8001)  on UCloud job
        LLM_CODING_MODEL (.env) — weights in HF_HOME

  (The annotation arm — Gemma for stages 02/03 — runs separately on
   LLM_ANNOTATE_PORT, default 8000. `serve-llm.sh both` runs both at once,
   GPU permitting -- see "Co-locating with the annotation pipeline" below.)
```

**No Tailscale, no ssh tunnel, no dynamic IPs.** UCloud's native **Public IP** resource (Resources → IP addresses) provides a static address that is attached to a job at submission time and reused every time — the endpoint URL never changes between jobs.

## Per-job workflow

Launch a Terminal or PyTorch job. At job submission:

- Attach **both Drives** as usual.
- Attach the **Public IP** resource (allocated once beforehand).
- Enable **SSH** and attach `utils/init.sh` as **Initialization**.

Machine type: `u1-gpu-1` (single A40 or H100) is sufficient for 7B models. On a B200, its 183 GB fits the coding model alongside the annotation arm (`serve-llm.sh both`) **only when the coding model is small** (e.g. the `Qwen/Qwen3.6-35B-A3B` default). It does not fit `deepseek-ai/DeepSeek-V4-Flash` (~146 GiB alone) — see "Co-locating with the annotation pipeline" below before assuming `both` works with whatever `LLM_CODING_MODEL` is currently configured.

```bash
# 1. Already done by Initialization param (also loads the CUDA module):
bash utils/init.sh

# 2. Install tooling + write Opencode config (idempotent; ~2 min first time):
bash utils/devenv.sh

# 3. Start the coding vLLM, confirm health, print endpoint:
bash utils/serve-llm.sh coding        # or: serve-llm.sh both (only if LLM_CODING_MODEL is small enough)

# 4. Code from within the job (UCLOUD_LLM_BASE_URL=localhost set by devenv.sh):
opencode

# 5. Code from your laptop (UCLOUD_LLM_BASE_URL set to the static IP — see below):
opencode
```

**Full-node job requirements.** For `deepseek-ai/DeepSeek-V4-Pro` or `moonshotai/Kimi-K2.7-Code` you must submit a full 8× B200 job (UCloud product `gpu-nvidia-b200`). All 8 GPUs on the node are required — neither model fits a single GPU, and no other role (e.g. annotation) can share the job.

## One-time laptop setup

Done once after allocating the Public IP.

**Add to `~/.bashrc` or `~/.zshrc` on your laptop:**

```bash
# UCloud coding LLM — static IP, never changes between jobs
export UCLOUD_LLM_BASE_URL="http://<UCLOUD_PUBLIC_IP>:8001/v1"   # LLM_CODING_PORT
export UCLOUD_LLM_KEY="sk-ucloud"   # must match LLM_API_KEY in .env
```

**Merge into `~/.config/opencode/opencode.json` on your laptop:**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ucloud-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "UCloud LLM",
      "options": {
        "baseURL": "{env:UCLOUD_LLM_BASE_URL}"
      },
      "models": {
        "Qwen/Qwen3.6-35B-A3B": {
          "name": "Qwen3.6-35B-A3B (UCloud)"
        },
        "deepseek-ai/DeepSeek-V4-Pro": {
          "name": "DeepSeek-V4-Pro (UCloud)"
        },
        "moonshotai/Kimi-K2.7-Code": {
          "name": "Kimi-K2.7-Code (UCloud)"
        }
      }
    }
  }
}
```

The `{env:UCLOUD_LLM_BASE_URL}` placeholder resolves only when the env var is set. When no UCloud job is running and the var is unset, the provider is inert — your existing Anthropic/other providers work normally.

**Add the auth token to `~/.config/opencode/auth.json`:**

```json
{ "ucloud-llm": "sk-ucloud" }
```

Or: `opencode auth login` → Other → provider ID: `ucloud-llm` → key: `sk-ucloud`.

## Switching models

Single switch point in `.env` on the project drive — nothing in `serve-llm.sh` needs editing to change models or to switch back:

```bash
LLM_CODING_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
LLM_TENSOR_PARALLEL=2
```

Re-run `devenv.sh` (updates Opencode config) and `serve-llm.sh coding` (restarts the coding vLLM). Update the model name in the laptop's `opencode.json` to match. Weights cache under `HF_HOME` on the DATA_DRIVE — shared across projects, no re-download on restart.

### `deepseek-ai/DeepSeek-V4-Flash` as the coding model

Tested and working at `LLM_TENSOR_PARALLEL=1` on a single B200 (verified: healthy, correct completions, ~150 tokens/sec). Sizing is tight but workable for one interactive session:

|                         |                                                                   |
| ----------------------- | ----------------------------------------------------------------- |
| Weights                 | ~146 GiB                                                          |
| `LLM_GPU_MEM_UTIL_SOLO` | 0.95 (not the smaller-model default of 0.90 — see `serve-llm.sh`) |
| KV cache budget at 0.95 | ~14 GiB → ~40K tokens, ~4.8x concurrency at 8K-token requests     |

`serve-llm.sh` detects this exact model id and automatically adds the required vLLM flags (`--tokenizer-mode deepseek_v4`, `--reasoning-parser deepseek_v4`, `--tool-call-parser deepseek_v4`, `--enable-auto-tool-choice`, `--kv-cache-dtype fp8`) plus `PYTHONPATH=utils/vllm_compat` — a workaround for a broken `nvidia-cutlass-dsl` wheel (see `docs/RUNNING_ON_UCLOUD.md`'s "DeepSeek-V4-Flash setup" section and the comments in `utils/vllm_compat/sitecustomize.py` for the full diagnosis). No other config changes are needed beyond `LLM_CODING_MODEL` in `.env`.

**Tradeoff:** this fully occupies one B200 (146 of 183 GiB just for weights), so `serve-llm.sh both` cannot co-locate it with the annotation arm on the same GPU — run it solo, or put `annotate` on a different GPU via manual `CUDA_VISIBLE_DEVICES` invocation (see "DeepSeek-V4-Flash setup" in `docs/RUNNING_ON_UCLOUD.md`).

### `deepseek-ai/DeepSeek-V4-Pro` as the coding model

1.6T total / 49B activated parameters, FP4+FP8 mixed precision. **Requires a full 8× B200 node** (UCloud product `gpu-nvidia-b200` — not a single-GPU terminal job).

|                   |                                                     |
| ----------------- | --------------------------------------------------- |
| Architecture      | 1.6T total / 49B activated, FP4+FP8 mixed precision |
| Weights on disk   | ~960 GB                                             |
| Node              | 8× B200 192 GB each (1,536 GB total VRAM)           |
| vLLM              | ≥0.20.0, `deepseek_v4` flag family                  |
| Output throughput | ~912 tok/s on HGX B200                              |
| TTFT              | ~1.2 s                                              |

vLLM command (currently `serve-llm.sh` does not auto-detect this model id; set flags via `VLLM_EXTRA_ARGS` in `.env` or run your own vLLM job):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention_config.use_fp4_indexer_cache=True \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2}' \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --kv-cache-dtype fp8 \
  --api-key ${LLM_API_KEY}
```

The `utils/vllm_compat/sitecustomize.py` cutlass-dsl workaround applies (same `deepseek_v4` kernel stack as Flash). vLLM recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro

**Tradeoff:** Fully occupies an 8× B200 node. Cannot co-locate with the annotation pipeline under any configuration. Submit a dedicated `gpu-nvidia-b200` job.

### `moonshotai/Kimi-K2.7-Code` as the coding model

1T total / 32B activated parameters, native INT4 quantization. Same architecture as Kimi K2.5/K2.6 (MLA attention, MoonViT vision encoder). **Requires 8× B200 (full node) with TP=8.**

|                 |                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------- |
| Architecture    | 1T total / 32B activated, native INT4                                                        |
| Weights on disk | ~595 GB                                                                                      |
| Node            | 8× B200, TP=8 (full node)                                                                    |
| vLLM            | ≥0.19.1                                                                                      |
| Thinking mode   | **Forced** — cannot be disabled. `temperature=1.0`, `top_p=0.95` locked                      |
| Multi-turn      | `preserve_thinking` always on — must carry `reasoning_content` forward in assistant messages |

vLLM command:

```bash
vllm serve moonshotai/Kimi-K2.7-Code \
  --tensor-parallel-size 8 \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --api-key ${LLM_API_KEY}
```

The `utils/vllm_compat/sitecustomize.py` workaround is **NOT needed** — Kimi uses MLA, not DeepSeek CSA/HCA. vLLM recipe: https://recipes.vllm.ai/moonshotai/Kimi-K2.7-Code

**Opencode client side:** The model forces thinking mode with `preserve_thinking`. Assistant messages include `reasoning_content` fields that must be forwarded in multi-turn interactions. May require custom opencode provider config.

**Kimi Code CLI** is the native agent framework: https://www.kimi.com/code

**API pricing** (alternative to self-hosted): $0.19/M tokens (cache hit), $0.95/M (cache miss), $4.00/M output.

**Tradeoff:** Fully occupies an 8× B200 node. Cannot co-locate with the annotation pipeline. Submit a dedicated `gpu-nvidia-b200` job.

## Co-locating with the annotation pipeline

The coding model defaults to `LLM_CODING_PORT=8001` and the annotation arm to `LLM_ANNOTATE_PORT=8000`, so they never clash on a port. To run both on one B200 in a single command:

```bash
bash utils/serve-llm.sh both     # Gemma :8000 + coding :8001, split VRAM (LLM_GPU_MEM_UTIL each)
```

**This only works when the coding model is small enough to share the GPU.** `both` splits VRAM via `LLM_GPU_MEM_UTIL` (default 0.45 each), so it has no chance of fitting `deepseek-ai/DeepSeek-V4-Flash` (~146 GiB alone) alongside Gemma (~54 GiB) on one 183 GiB B200 — run DeepSeek solo (`serve-llm.sh coding`), or give each role its own GPU with a manual `CUDA_VISIBLE_DEVICES`-pinned invocation.

For laptop access to the coding model, open port `8001` TCP on the Public IP resource (the annotation arm is job-local and needs no public port).

## Security

> From the UCloud docs: _"Enabling this feature allows anyone with the IP to access > the application. Measures must be implemented to ensure that the application is > adequately protected."_

`vllm serve --api-key ${LLM_API_KEY}` (set in `serve-llm.sh`) requires a Bearer token on every request. Use a genuine secret in `LLM_API_KEY` rather than the default `sk-ucloud`.

## Stopping

```bash
bash utils/serve-llm.sh --stop coding   # stop just the coding model
bash utils/serve-llm.sh --stop          # stop all roles (annotate + coding)
```

The Public IP resource stays allocated after the job ends — detach it in the UCloud UI when not in use, and reattach it at the next job submission.

`serve-llm.sh --status [role]` and `--stop [role]` first load `.env` so they only inspect or stop the models this project configures (matched by exact model id). If `.env` is unavailable, `--status` reports `unknown` and `--stop` refuses to kill any process rather than matching an unrelated vLLM server.
