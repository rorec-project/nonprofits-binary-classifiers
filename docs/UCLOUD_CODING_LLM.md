# Coding LLM (optional appendix)

AI-assisted coding for interactive UCloud sessions, layered onto the base workflow via the `devenv.sh` overlay. The LLM stack is never touched by batch jobs.

## Architecture

```
Opencode (laptop)                  Opencode (UCloud SSH session)
       │                                       │
       │ UCloud Public IP                      │ localhost
       │ (static, allocated once in §2 step 5) │
       └───────────────────┬───────────────────┘
                           ▼
               vLLM :8000  on UCloud job
               LLM_CODING_MODEL (.env) — weights in HF_HOME
```

**No Tailscale, no ssh tunnel, no dynamic IPs.** UCloud's native **Public IP** resource (Resources → IP addresses) provides a static address that is attached to a job at submission time and reused every time — the endpoint URL never changes between jobs.

## Per-job workflow

Launch a Terminal or PyTorch job. At job submission:

- Attach **both Drives** as usual.
- Attach the **Public IP** resource (allocated once beforehand).
- Enable **SSH** and attach `utils/init.sh` as **Initialization**.

Machine type: `u1-gpu-1` (single A40 or H100) is sufficient for 7B models. Reserve B200 for pipeline stages.

```bash
# 1. Already done by Initialization param:
bash utils/init.sh

# 2. Install tooling + write Opencode config (idempotent; ~2 min first time):
bash utils/devenv.sh

# 3. Start vLLM, confirm health, print endpoint:
bash utils/serve-llm.sh

# 4. Code from within the job (UCLOUD_LLM_BASE_URL=localhost set by devenv.sh):
opencode

# 5. Code from your laptop (UCLOUD_LLM_BASE_URL set to the static IP — see below):
opencode
```

## One-time laptop setup

Done once after allocating the Public IP.

**Add to `~/.bashrc` or `~/.zshrc` on your laptop:**

```bash
# UCloud coding LLM — static IP, never changes between jobs
export UCLOUD_LLM_BASE_URL="http://<UCLOUD_PUBLIC_IP>:8000/v1"
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
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
          "name": "Qwen2.5-Coder-7B (UCloud)"
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

Single switch point in `.env` on the project drive:

```bash
LLM_CODING_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"   # stronger; needs B200
LLM_TENSOR_PARALLEL=2
```

Re-run `devenv.sh` (updates Opencode config) and `serve-llm.sh` (restarts vLLM). Update the model name in the laptop's `opencode.json` to match. Weights cache under `HF_HOME` on the DATA_DRIVE — shared across projects, no re-download on restart.

## Co-locating with the annotation pipeline

If running the coding LLM on the same B200 job as stages 03/06/07/08, set `LLM_CODING_PORT=8001` in `.env` to avoid clashing with the annotation vLLM on `:8000`. Open port `8001` TCP on the Public IP resource in addition to (or instead of) `8000`.

## Security

> From the UCloud docs: _"Enabling this feature allows anyone with the IP to access > the application. Measures must be implemented to ensure that the application is > adequately protected."_

`vllm serve --api-key ${LLM_API_KEY}` (set in `serve-llm.sh`) requires a Bearer token on every request. Use a genuine secret in `LLM_API_KEY` rather than the default `sk-ucloud`.

## Stopping

```bash
bash utils/serve-llm.sh --stop
```

The Public IP resource stays allocated after the job ends — detach it in the UCloud UI when not in use, and reattach it at the next job submission.
