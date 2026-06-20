#!/usr/bin/env bash
# utils/devenv.sh — OPTIONAL dev/agent overlay for interactive SSH sessions.
#
# Opt-in only. Layers an editor + agent harness on top of the lean runtime
# (utils/init.sh): Neovim + LazyVim (+ optional private dotfiles), Claude Code,
# opencode (the independent-LLM-review harness, driven by ANTHROPIC_API_KEY),
# XDG dirs, git config, and Opencode provider configuration for the UCloud vLLM
# coding assistant (see utils/serve-llm.sh and §14 of RUNNING_ON_UCLOUD.md).
# Everything installs into the PROJECT drive so it
# persists across jobs. It is NEVER invoked by batch/GPU jobs and writes its own
# sourced file (devenv.sh) so the lean runtime env.sh stays free of harness vars.
#
# Run AFTER utils/init.sh, over SSH:  bash utils/devenv.sh
#
# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail
shopt -s nullglob

# Refuse non-TTY contexts (e.g., UCloud Initialization or batch jobs). A script
# invoked as `bash utils/devenv.sh` is non-interactive in Bash's `$-` sense, so
# TTY presence is the reliable signal for an SSH/web-terminal session.
if [ ! -t 0 ] || [ ! -t 1 ]; then
  echo "ERROR: utils/devenv.sh is for interactive SSH sessions only." >&2
  echo "       Do not use it as an Initialization script or in batch jobs." >&2
  exit 1
fi

# ─── Auto-discover PROJECT_DRIVE by scanning for .env under /work/ ────────────
ENV_FILE=""
for d in /work/*/; do
  if [ -f "${d}.env" ]; then
    ENV_FILE="${d}.env"
    PROJECT_DRIVE="$(basename "${d}")"
    echo "--- Found .env on /work/${PROJECT_DRIVE} ---"
    break
  fi
done

if [ -z "${ENV_FILE}" ]; then
  echo "ERROR: .env not found under /work/." >&2
  echo "       Create it on the project drive before starting the job." >&2
  exit 1
fi

set -a && . "${ENV_FILE}" && set +a

# Git identity + the HTTPS credential helper are established by utils/init.sh in
# the default ~/.gitconfig on every job; the overlay deliberately does NOT set
# GIT_CONFIG_GLOBAL (that would replace ~/.gitconfig and drop the credential
# helper, breaking git auth in the dev session).

# Private dotfiles repo (set to "" to skip the dotfiles overlay).
DOTFILES_REPO="https://github.com/chickymonkeys/omarchy-dotfiles.git"

# Feature flags (1 = enable, 0 = disable).
INSTALL_NEOVIM=1     # Neovim + LazyVim starter
INSTALL_DOTFILES=1   # overlay private dotfiles on LazyVim (needs INSTALL_NEOVIM=1 + GITHUB_TOKEN)
INSTALL_OPENCODE=1   # opencode (independent LLM review harness)
INSTALL_CLAUDE=1     # Claude Code
CONFIGURE_OPENCODE=1 # Write Opencode provider config pointing at the UCloud LLM

PROJECT="/work/${PROJECT_DRIVE}"
TOOLS_DIR="${PROJECT}/devtools"
BIN_DIR="${TOOLS_DIR}/bin"
DEVENV_SH="${PROJECT}/devenv.sh"

if [ ! -d "${PROJECT}" ]; then
  echo "ERROR: project drive not found at ${PROJECT}. Run utils/init.sh first." >&2
  exit 1
fi

mkdir -p "${BIN_DIR}"
export PATH="${BIN_DIR}:${PATH}"

# ─── XDG dirs (persistent on the project drive) ───────────────────────────────

export XDG_CONFIG_HOME="${TOOLS_DIR}/config"
export XDG_DATA_HOME="${TOOLS_DIR}/share"
export XDG_STATE_HOME="${TOOLS_DIR}/state"
export XDG_CACHE_HOME="${TOOLS_DIR}/cache"
mkdir -p "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${XDG_STATE_HOME}" "${XDG_CACHE_HOME}"

# ─── Editor-support system packages (no R / GIS toolchain) ────────────────────

echo "--- System packages (fd, ripgrep, fzf) ---"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  fd-find ripgrep fzf

# ─── Neovim + LazyVim ─────────────────────────────────────────────────────────

if [ "${INSTALL_NEOVIM}" -eq 1 ]; then
  if [ ! -x "${BIN_DIR}/nvim" ]; then
    echo "--- Installing Neovim ---"
    curl -fsSL "https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz" |
      tar xz -C "${TOOLS_DIR}"
    ln -sf "${TOOLS_DIR}/nvim-linux-x86_64/bin/nvim" "${BIN_DIR}/nvim"
  fi

  NVIM_CONFIG="${XDG_CONFIG_HOME}/nvim"
  if [ ! -d "${NVIM_CONFIG}" ]; then
    echo "--- Installing LazyVim starter ---"
    git clone --depth=1 https://github.com/LazyVim/starter "${NVIM_CONFIG}"
    rm -rf "${NVIM_CONFIG}/.git"
  fi

  if [ "${INSTALL_DOTFILES}" -eq 1 ] && [ -n "${DOTFILES_REPO}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
    DOTFILES_DIR="${TOOLS_DIR}/dotfiles"
    if [ ! -d "${DOTFILES_DIR}" ]; then
      echo "--- Cloning dotfiles ---"
      git clone --depth=1 "${DOTFILES_REPO}" "${DOTFILES_DIR}"
    else
      git -C "${DOTFILES_DIR}" pull --ff-only 2>/dev/null || true
    fi
    [ -d "${DOTFILES_DIR}/dotfiles/.config/nvim" ] &&
      cp -r "${DOTFILES_DIR}/dotfiles/.config/nvim/." "${NVIM_CONFIG}/"
  fi
fi

# ─── opencode + Claude Code (npm prefix on the project drive) ─────────────────

NPM_DIR="${TOOLS_DIR}/npm-global"
mkdir -p "${NPM_DIR}"
npm config set prefix "${NPM_DIR}"

if [ "${INSTALL_OPENCODE}" -eq 1 ] && [ ! -x "${BIN_DIR}/opencode" ]; then
  echo "--- Installing opencode ---"
  npm install -g opencode-ai
  [ -f "${NPM_DIR}/bin/opencode" ] && ln -sf "${NPM_DIR}/bin/opencode" "${BIN_DIR}/opencode" ||
    echo "WARNING: opencode binary not found after install" >&2
fi

if [ "${INSTALL_CLAUDE}" -eq 1 ] && [ ! -x "${BIN_DIR}/claude" ]; then
  echo "--- Installing Claude Code ---"
  npm install -g @anthropic-ai/claude-code
  [ -f "${NPM_DIR}/bin/claude" ] && ln -sf "${NPM_DIR}/bin/claude" "${BIN_DIR}/claude" ||
    echo "WARNING: claude binary not found after install" >&2
fi

# ─── Opencode provider configuration (UCloud vLLM coding assistant) ────────────

if [ "${CONFIGURE_OPENCODE}" -eq 1 ] && [ "${INSTALL_OPENCODE}" -eq 1 ]; then
  OC_CONFIG_DIR="${XDG_CONFIG_HOME}/opencode"
  mkdir -p "${OC_CONFIG_DIR}"

  MODEL_SHORT="$(basename "${LLM_CODING_MODEL}")"

  cat >"${OC_CONFIG_DIR}/opencode.json" <<OCEOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ucloud-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "UCloud LLM (${MODEL_SHORT})",
      "options": {
        "baseURL": "{env:UCLOUD_LLM_BASE_URL}"
      },
      "models": {
        "${LLM_CODING_MODEL}": {
          "name": "${MODEL_SHORT}"
        }
      }
    }
  },
  "model": "ucloud-llm/${LLM_CODING_MODEL}"
}
OCEOF

  cat >"${OC_CONFIG_DIR}/auth.json" <<AUTHEOF
{
  "ucloud-llm": "${LLM_API_KEY}"
}
AUTHEOF
  echo "--- Opencode config → ${OC_CONFIG_DIR}/ ---"
fi

# ─── Write the overlay env file and auto-source it (separate from env.sh) ─────

cat >"${DEVENV_SH}" <<EOF
# devenv.sh — dev/agent overlay environment (interactive sessions only).
# Auto-generated by utils/devenv.sh. Kept separate from the runtime env.sh so
# batch/GPU jobs never pick up the editor/agent harness vars.
export PATH="${BIN_DIR}:\$PATH"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME}"
export XDG_DATA_HOME="${XDG_DATA_HOME}"
export XDG_STATE_HOME="${XDG_STATE_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
# Opencode LLM provider: vLLM on localhost (job) or Public IP (laptop).
# {env:UCLOUD_LLM_BASE_URL} in opencode.json resolves this at runtime.
# On your laptop override this in ~/.bashrc / ~/.zshrc with the static Public IP
# (set UCLOUD_PUBLIC_IP in .env — see §14 of RUNNING_ON_UCLOUD.md).
export UCLOUD_LLM_BASE_URL="http://localhost:${LLM_CODING_PORT}/v1"
export UCLOUD_LLM_KEY="${LLM_API_KEY}"
EOF

SOURCE_LINE="[ -f '${DEVENV_SH}' ] && source '${DEVENV_SH}'"
for rc in "${HOME}/.bashrc" "${HOME}/.profile" "${HOME}/.bash_profile"; do
  grep -qF "${DEVENV_SH}" "${rc}" 2>/dev/null || echo "${SOURCE_LINE}" >>"${rc}"
done

echo "=== devenv.sh complete $(date) ==="

echo "Open a new shell (or 'source ${DEVENV_SH}') to pick up nvim / claude / opencode."
echo ""
echo "  Next steps on this job:"
echo "    bash utils/serve-llm.sh coding   # start the coding LLM (or 'both' with Gemma)"
echo "    opencode                   # launch agent — uses localhost:${LLM_CODING_PORT}"
echo ""
echo "  Laptop one-time setup (add to ~/.bashrc / ~/.zshrc):"
if [ -n "${UCLOUD_PUBLIC_IP:-}" ]; then
  echo "    export UCLOUD_LLM_BASE_URL=\"http://${UCLOUD_PUBLIC_IP}:${LLM_CODING_PORT}/v1\""
else
  echo "    export UCLOUD_LLM_BASE_URL=\"http://<UCLOUD_PUBLIC_IP>:${LLM_CODING_PORT}/v1\""
  echo "    (set UCLOUD_PUBLIC_IP in .env — see §14 of RUNNING_ON_UCLOUD.md)"
fi
echo "    export UCLOUD_LLM_KEY=\"${LLM_API_KEY}\""
echo "  Also merge the ucloud-llm provider block from:"
echo "    ${XDG_CONFIG_HOME}/opencode/opencode.json"
echo "  into your laptop's ~/.config/opencode/opencode.json (or XDG equivalent)."
