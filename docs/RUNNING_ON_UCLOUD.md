# Running on UCloud

This runbook describes how to run the binary classifier pipeline on the
UCloud SDU/DeiC Interactive HPC platform.

## Prerequisites

- A UCloud project with access to the **gpu-nvidia-b200** SKU.
- Your SSH public key uploaded in **Resources → SSH keys**.

## Job submission

1. Open the UCloud interface and start a **Terminal** job.
2. Select machine type **`gpu-nvidia-b200`** (full GPUs, not the `-mig` fractional variant).
3. Enable the **SSH server** at submission time.
4. (Optional) Attach the **Initialization** script `utils/init.sh` so the environment is ready before you log in.

## Connecting

```bash
ssh ucloud@ssh.cloud.sdu.dk -p <PORT>
```

The `<PORT>` is displayed in the UCloud job details after the job starts.

## Persistence

Only `/work` survives job termination. Mount a private Drive at `/work`
and keep **all** persistent data there:

- Repository clone
- Input `.parquet` files
- `uv` caches (`/work/.uv-cache`, `/work/.uv-python`)
- Output artifacts
- The `.venv` created by `uv sync`

## Environment setup

`uv` is pre-installed (v0.11.3). The system Python is 3.12, but `uv` self-manages Python 3.13 as required by `pyproject.toml`. The `utils/init.sh` script handles installation and syncing:

```bash
cd /work/nonprofits-binary-classifiers
bash utils/init.sh
```

After `utils/init.sh` completes, load the shell environment manually:

```bash
set -a
source /work/.env
set +a
```

> **Caveat:** Variables exported inside `utils/init.sh` do **not** propagate to the interactive shell. Always source `.env` after connecting.

### GPU environment

B200 GPUs are Blackwell `sm_100` devices and require PyTorch 2.7 or newer from the CUDA 12.8 (`cu128`) wheel index. For training jobs, install the base environment only:

```bash
uv sync
```

Use `uv sync --extra serve` only on jobs that serve annotation models with vLLM.

Before production runs, verify the CUDA build and bf16 support:

```bash
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_bf16_supported())"
```

## Secrets

Store `OPENAI_API_KEY` (and any other secrets) in a `.env` file inside a **private `/work` Drive**. Never commit secrets to the repository.

## vLLM serving

The open-weight annotator is served locally inside the job via vLLM:

```bash
uv run vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tensor-parallel-size 8 \
    --port 8000
```

The annotator calls `http://127.0.0.1:8000/v1`. No public port forwarding is required.

## GPU compatibility check

Before the first run, verify that the CUDA driver and PyTorch build are compatible with the Blackwell B200:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If the device name is not recognized or CUDA errors appear, pin a B200-compatible vLLM / PyTorch build in `pyproject.toml` and re-run `uv sync`.

## Caveats

- `utils/init.sh` export propagation: environment variables set inside the init script are isolated. Source `.env` manually after SSH login.
- Wall-time limits: interactive jobs have a finite duration. Save checkpoints frequently and resume via the per-stage `--limit` / resume-by-`EIN2` mechanisms.
- The `uv` cache and Python install directories are redirected to `/work` so they persist across jobs.
