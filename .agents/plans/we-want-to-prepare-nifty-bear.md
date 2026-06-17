# UCloud Execution Setup for the Binary-Classifier Pipeline

## Context

We run ML pipelines on **UCloud (SDU/DeiC Interactive HPC)** as our main workhorse.
The current project (`nonprofits-binary-classifiers`, this repo) needs a clean way to
run on UCloud: an init script for terminal/batch jobs, a unified storage strategy for
data/outputs/models/caches, reusable open-weight LLM caches, ruff/ty made runnable,
Git/GitHub management, and clear instructions for both interactive (SSH-from-local) and
batch use.

The prior setup (`~/Documents/Projects/omarchy-dotfiles/ucloud/{init.sh,batch.sh}`) was
built for an R/renv project and **conflated three concerns** — dev environment, agent
harness, and pipeline runtime — all under a personal Member Files drive. It also has a
duplicated setup block (lines 44–92) and assumes manual `.env` creation. The project
already carries a *minimal* `utils/init.sh` and `docs/RUNNING_ON_UCLOUD.md`, but they
miss HF cache redirection, have a stale model reference (Qwen-235B vs the authoritative
Gemma in config), and don't address the multi-drive storage strategy.

**Goal:** a simple, layered setup — *storage layout first, scripts just point the
ephemeral job at it* — that keeps reusable assets (raw corpus, HF/LLM caches) on a
shared cross-project drive, the repo + outputs on a project drive, and cleanly separates
lean runtime from an optional dev/agent overlay.

**Decisions locked with the user:**
- Storage: **shared cross-project drive** for raw data + reusable caches; **project
  drive** for the repo + outputs.
- Annotator model: **`config/religious_missions.yaml` is authoritative** →
  `google/gemma-3-27b-it` (vLLM), switchable by editing the YAML + the serve command.
- Dev/agent layer: **optional overlay**, decoupled from runtime.
- Workflow: **run stage 01 + human gates locally** (deterministic, one-time), upload
  only what's needed; UCloud runs the heavy GPU stages.
- Results: **simple `rsync` over SSH**, no SSHFS/IDE-remote requirement.

---

## Storage design (the spine)

Two UCloud Drives, both mounted into every job (each appears at `/work/<DriveName>`):

**Drive A — shared corpus & cache** (cross-project; reused by future repos e.g. UCLOUD-BERTOPIC)
```
/work/<DATA_DRIVE>/
  raw/            # uploaded ONCE, manually: missions_cross_section.parquet, bmf_unified_processed.parquet
  hf-cache/       # HF_HOME — transformers encoders AND vLLM open-weight downloads (Gemma) share this
  uv-cache/       # UV_CACHE_DIR
  uv-python/      # UV_PYTHON_INSTALL_DIR
```

**Drive B — project** (this repo; SEPARATE from the shared corpus drive)
```
/work/<PROJECT_DRIVE>/           # repo root (clone directly into the drive mount)
  .env                             # secrets (OPENAI/HF/GITHUB/ANTHROPIC) — private, never committed
  .git/                            # repo (cloned in place)
  utils/   src/   docs/   ...      # repo code
  outputs/                         # heavy pipeline outputs
    interim/                       # <- data/interim
    models/                        # <- data/models
    processed/{silver_labels.csv,evaluation,predictions,prevalence,figures}  # heavy processed outputs
```

**Wiring = symlinks** (so the authoritative YAML's *relative* `data/*` paths resolve
unchanged; `PathRegistry` anchors at `cwd`). Created idempotently by the init script:
- `data/raw`            → `/work/<DATA_DRIVE>/raw`
- `data/interim`        → `/work/<PROJECT_DRIVE>/outputs/interim`
- `data/models`         → `/work/<PROJECT_DRIVE>/outputs/models`
- `data/processed/{silver_labels.csv,evaluation,predictions,prevalence,figures}` →
  `/work/<PROJECT_DRIVE>/outputs/processed/...`
- `data/processed/gold/` **stays in the repo** (git-committed human labels + slate).

> **Watch-item (gitignore):** the symlinked paths under `data/processed/` sit inside a
> git-tracked tree (because `gold/` is tracked). Git records a symlink as content, so
> after the init script runs, `git status` would otherwise show machine-specific symlinks
> (→ `/work/<PROJECT_DRIVE>/…`) as files to commit. During implementation, confirm
> `.gitignore` already covers every symlinked path (`git check-ignore data/processed/figures`
> etc.) and add entries if not, so the working tree stays clean.

> **Watch-item (drive names are load-bearing):** the `/work/` prefix is confirmed, but the
> exact mounted directory *name* is the risk — a member-files drive mounted with a `#NNNN`
> suffix (`/work/AlessandroPizzigolotto#6144`), while a user-created drive may mount as
> `/work/<Title>` (possibly sanitized/suffixed). The user must set `DATA_DRIVE` /
> `PROJECT_DRIVE` in each script's config block to the **observed** names on a live job,
> and the init script must **validate each mount and fail loud** if absent (carry over the
> old script's `if [ ! -d "${DRIVE}" ]; then … exit 1` check). The example paths below are
> placeholders, not known-correct.

A single `HF_HOME=/work/<DATA_DRIVE>/hf-cache` covers both the encoder models
(`deberta-v3-base`, `ModernBERT-base`, `all-MiniLM-L6-v2`) and the vLLM Gemma download,
so encoders and the open-weight annotator share one reusable cache across projects.

---

## One-time MANUAL setup (before any script) — to be documented for the user

1. **Project + grant.** In the `bertopic-test` project, ensure a **Storage** allocation
   exists (project **Allocations** panel). Add members via **Invite** if collaborating.
2. **Create the two Drives** in **Files → "Create Drives"** (`<DATA_DRIVE>`, `<PROJECT_DRIVE>`).
3. **Upload raw parquet** into `/work/<DATA_DRIVE>/raw/` (UCloud web uploader, one time).
   Also upload the stage-01 outputs that are gitignored but needed downstream — the
   `data/interim/manifests/*.csv` (esp. `silver_manifest.csv`, read by stage 03) — into
   `/work/<PROJECT_DRIVE>/outputs/interim/manifests/` (see Local-first workflow).
4. **Upload SSH public key** in **Resources → SSH keys**.
5. **Gemma is gated:** accept the license on huggingface.co for `google/gemma-3-27b-it`
   and create an **HF token**.
6. **Create the secrets `.env`** (private; never committed) with `OPENAI_API_KEY`,
   `HF_TOKEN`, `GITHUB_TOKEN`, and (for the overlay) `ANTHROPIC_API_KEY`.
   Place it on the **project drive** at `/work/<PROJECT_DRIVE>/.env` — which is a
   **separate drive from the shared corpus drive** (no secrets ever go on the corpus drive,
   since other repos/people mount it). `init.sh` sources it from there.
7. **Clone the repo (manual, one-time) into `/work/<PROJECT_DRIVE>/`** — and **document the
   exact steps from the UCloud platform**: start a Terminal job with the project drive
   attached → open the web terminal (or SSH in) → authenticate git **without leaking the
   token into shell history / `.git/config`** (use a fine-grained PAT via
   `git config --global credential.helper store` seeded from `.env`, or `gh auth login`)
   → `git clone https://github.com/<user>/nonprofits-binary-classifiers.git`. After this
   first clone, `utils/init.sh` takes over (pull + sync + symlinks) on every subsequent job.

---

## Local-first workflow (one-time, deterministic)

Run the cheap/deterministic front of the pipeline **locally**, commit the small tracked
artifacts, then let UCloud do the heavy lifting:
- Local: `01_build_sample` (manifests + `gold_to_code.csv`), human **G1** coding, optional
  `02_bakeoff_prompts` via OpenAI API, human **G2** → `production_slate.json`.
- These gold/slate files are **git-committed** (under `data/processed/gold/`), so they
  travel to UCloud via `git pull` — no manual upload of labels needed.
- **But the interim manifests are NOT git-tracked** (`data/interim` is symlinked-to-drive /
  gitignored). Stage 03 on UCloud reads `silver_manifest.csv` as input, so the
  `data/interim/manifests/*.csv` produced locally by stage 01 **must be uploaded once** to
  the project drive's `outputs/interim/manifests/` (manual setup step 3). **Do not just
  re-run stage 01 on UCloud to regenerate them** — verify first whether `01_build_sample`
  clobbers an existing `gold_to_code.csv`; if it does, re-running would destroy the
  human labels. Upload the manifests instead of re-running.
- UCloud (GPU) then runs the heavy stages: `03` annotation *iff* the open-weight Gemma
  arm is used (vLLM), and `06`/`07`/`08` (train/eval/infer). CPU-only stages
  (`04`,`05`,`09`,`10`,`11`) run on a cheap `cpu-amd-zen5` node or locally.

This keeps B200 hours spent only on `02/03` (open-weight) + `06/07/08`.

---

## Scripts (layered) — files to create/modify under `utils/`

Each script opens with a small **config block** (`DATA_DRIVE`, `PROJECT_DRIVE`, `REPO_DIR`,
git identity) — the same edit-one-block pattern the user already knows from the old
`MEMBER_FILES_NAME`.

### 1. `utils/init.sh` — lean RUNTIME init (rewrite of current file)
App-agnostic (works as the **Initialization** Bash script for Terminal / PyTorch /
JupyterLab, and as a manual `bash utils/init.sh`). Steps:
- Wait for `/work`; assert both Drives mounted.
- Export runtime env: `HF_HOME`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PATH`; source
  `/work/<PROJECT_DRIVE>/.env` (`set -a … set +a`).
- **Write `env.sh` + auto-source it in `~/.bashrc`/`~/.profile`** (fixes the documented
  "exports don't propagate to the interactive shell" caveat — keep the old pattern).
- Ensure the **symlinks** above (idempotent).
- `uv python install 3.13` (system Python is 3.12) and `uv sync` (base env;
  `--extra serve` only on vLLM jobs).
- Optional **model pre-pull** driven by the config (`uv run hf download <id>` into HF_HOME).
- Reuse existing knobs already in `utils/init.sh`: `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`.

### 2. `utils/devenv.sh` — OPTIONAL dev/agent overlay (new)
Opt-in for interactive SSH sessions only. Ports the *good* parts of the old monolith
**minus R/renv**: Neovim + dotfiles overlay, Claude Code + opencode (the "independent LLM
review" agent harness, driven by `ANTHROPIC_API_KEY`), XDG dirs, git config. Installs into
the project drive so it persists. Never invoked by batch/GPU jobs.

### 3. `utils/run.sh` — batch/segment runner (new; replaces the R `batch.sh`)
`source env.sh` → `git pull --ff-only` → `uv run python scripts/run_pipeline.py
--stages "$1" --config config/religious_missions.yaml`. **Stage-range aware** because the
human gates (G1/G2) break the run into segments; document which segment runs on CPU vs
GPU nodes. Honor per-stage resume (`--limit`, resume-by-`EIN2`) for wall-time safety.

### 4. `utils/sync-results.sh` — simple local pull (new, tiny)
A documented one-liner wrapper: `rsync -av -e "ssh -p <PORT>"
ucloud@ssh.cloud.sdu.dk:/work/<PROJECT_DRIVE>/outputs/processed/ ./local-results/`.
No SSHFS / IDE-remote dependency.

---

## ruff & ty

Already **configured** in `pyproject.toml` (`[tool.ruff]`, `[tool.ty.*]`) — "configure"
here means **make them runnable**. Add `ruff` and `ty` to the `dev` dependency group so
`uv sync` installs them; document `uv run ruff check` / `uv run ty check`. (Alternative:
`uvx ruff` / `uvx ty` with no manifest change — but adding to `dev` is more reproducible.)

## Model / vLLM (config-authoritative + switchable)

- Serve the authoritative model: `uv run vllm serve google/gemma-3-27b-it --port 8000`
  at `--tensor-parallel-size 1` (or 2) — ~54 GB bf16 fits a single B200 (192 GB).
  **Correct the stale `Qwen-235B / TP=8` lines** in `docs/RUNNING_ON_UCLOUD.md`.
- **Switching models:** edit the `vllm` candidate id in `config/religious_missions.yaml`,
  re-pull weights, and update the `vllm serve` id (+ raise `--tensor-parallel-size` for
  larger models). Document this as the single switch point.

## GitHub management

Repo cloned once on the project drive; `GITHUB_TOKEN` from `.env` via git's credential
store (HTTPS) so `pull`/`commit`/`push` work in place. Set `user.name`/`user.email` in the
init config block. Prefer the credential helper over embedding the token in the remote URL
(the old script's `https://$TOKEN@…` pattern risks leaking the token into logs).

## Documentation

Rewrite `docs/RUNNING_ON_UCLOUD.md` to cover: two-drive layout, one-time manual setup,
local-first workflow, init/overlay/batch scripts, GPU vs CPU node choice per stage, model
switching, and the `rsync` result-sync recipe. Extend `.env.example` with `HF_TOKEN`,
`GITHUB_TOKEN`, `ANTHROPIC_API_KEY` (currently only `OPENAI_API_KEY`).

## Operational notes (gaps surfaced while defining manual vs automated)

- **Bootstrap order (manual clone).** First-ever job: Terminal + project drive → create
  `.env` → authenticate git → clone → `bash utils/init.sh`. Every later job: `init.sh`
  detects the existing checkout and only `git pull`s + `uv sync`s. Document both paths.
- **Cost / wall-time hygiene (B200 is expensive).** Run CPU-only stages on `cpu-amd-zen5`,
  not the GPU node; right-size wall-time; **stop the job when idle** (UCloud bills running
  jobs, including idle interactive ones). Use checkpoint/resume for long stages.
- **Outbound network assumption.** Jobs need outbound internet for OpenAI, HF Hub, GitHub,
  npm, and Anthropic (overlay). The old R scripts relied on this and it worked — state it
  as an assumption to confirm on first run.
- **Stage → node map (put a table in the runbook).** CPU node / local: 01, 02 (OpenAI
  arm), 04, 05, 09, 10, 11. GPU node: 03 (only if the open-weight vLLM arm is used), 06,
  07, 08. vLLM serving needed only for the open-weight annotation arm.
- **Repeatable submission.** Save the UCloud job parameters (machine type, both drives,
  Initialization script, SSH) as a reusable job template so you don't re-enter them.
- **Disk/quota growth.** HF cache + uv cache + checkpoints + `.venv` accumulate on the
  drives; note quota awareness and an occasional cleanup of stale checkpoints.
- **Secrets location (settled).** `.env` lives on the **project drive**
  (`/work/<PROJECT_DRIVE>/.env`), which is separate from the shared corpus drive. No
  secrets on the corpus drive.

---

## Critical files

- `utils/init.sh` — rewrite (lean runtime; symlinks; HF/uv env; env.sh auto-source).
- `utils/devenv.sh` — new (optional dev/agent overlay; no R).
- `utils/run.sh` — new (segment batch runner).
- `utils/sync-results.sh` — new (rsync pull).
- `pyproject.toml` — add `ruff`, `ty` to `[dependency-groups] dev`.
- `.env.example` — add `HF_TOKEN`, `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`.
- `docs/RUNNING_ON_UCLOUD.md` — rewrite.
- Reference (read-only): `src/binary_classifier/paths.py`, `config.py`,
  `config/religious_missions.yaml` (path/model resolution — do not edit the YAML's infra).

## Verification (end-to-end on a fresh job)

1. Start a Terminal/PyTorch job, attach both Drives, attach `utils/init.sh` as Initialization, enable SSH.
2. SSH in; confirm env auto-loaded: `echo $HF_HOME` points at the data drive; symlinks
   resolve (`ls -l data/raw` → `/work/<DATA_DRIVE>/raw`).
3. `uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_bf16_supported())"` (GPU job).
   If this reports a non-cu128 build or bf16 unsupported on B200 (sm_100), pin the cu128
   torch index for `uv` and re-sync (anticipated failure mode, not a surprise).
4. `uv run ruff check .` and `uv run ty check` succeed.
5. Run one CPU stage end-to-end (e.g. `bash utils/run.sh 10` → figures appear under the
   symlinked `data/processed/figures` on the project drive).
6. **Exercise the 01→03 boundary:** confirm `data/interim/manifests/silver_manifest.csv`
   resolves through the symlink (uploaded, not regenerated), then run a `--limit`'d stage
   03 to prove it reads the manifest — catches the "manifest not found" footgun.
7. GPU smoke: short `06_train` (or `--limit`) writes a checkpoint under `data/models`.
8. `git pull`/`commit`/`push` works in the project-drive checkout.
9. From local: `bash utils/sync-results.sh` pulls outputs down via rsync.
10. (If open-weight arm) `vllm serve google/gemma-3-27b-it` comes up; annotator reaches
   `http://127.0.0.1:8000/v1`; weights cached under `HF_HOME` (no re-download on restart).
