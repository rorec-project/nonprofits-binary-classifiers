#!/usr/bin/env bash
# utils/devenv.sh — OPTIONAL dev/agent overlay for interactive SSH sessions.
#
# Opt-in only. Layers an editor + agent harness on top of the lean runtime
# (utils/init.sh): Neovim + LazyVim (+ optional private dotfiles), Claude Code,
# opencode (the independent-LLM-review harness, driven by ANTHROPIC_API_KEY),
# XDG dirs, and git config. Everything installs into the PROJECT drive so it
# persists across jobs. It is NEVER invoked by batch/GPU jobs and writes its own
# sourced file (devenv.sh) so the lean runtime env.sh stays free of harness vars.
#
# Run AFTER utils/init.sh, over SSH:  bash utils/devenv.sh
#
# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

. "$(dirname "$0")/config.sh"

# Git identity + the HTTPS credential helper are established by utils/init.sh in
# the default ~/.gitconfig on every job; the overlay deliberately does NOT set
# GIT_CONFIG_GLOBAL (that would replace ~/.gitconfig and drop the credential
# helper, breaking git auth in the dev session).

# Private dotfiles repo (set to "" to skip the dotfiles overlay).
DOTFILES_REPO="https://github.com/chickymonkeys/omarchy-dotfiles.git"

# Feature flags (1 = enable, 0 = disable).
INSTALL_NEOVIM=1    # Neovim + LazyVim starter
INSTALL_DOTFILES=1  # overlay private dotfiles on LazyVim (needs INSTALL_NEOVIM=1 + GITHUB_TOKEN)
INSTALL_OPENCODE=1  # opencode (independent LLM review harness)
INSTALL_CLAUDE=1    # Claude Code

# Refuse to run in non-interactive shells (e.g., as a UCloud Initialization
# script or batch job). This overlay is for interactive SSH sessions only.
case "$-" in
  *i*) ;;
  *)
    echo "ERROR: utils/devenv.sh is for interactive SSH sessions only." >&2
    echo "       Do not use it as an Initialization script or in batch jobs." >&2
    exit 1
    ;;
esac

PROJECT="/work/${PROJECT_DRIVE}"
TOOLS_DIR="${PROJECT}/devtools"
BIN_DIR="${TOOLS_DIR}/bin"
ENV_FILE="${PROJECT}/.env"
DEVENV_SH="${PROJECT}/devenv.sh"

if [ ! -d "${PROJECT}" ]; then
  echo "ERROR: project drive not found at ${PROJECT}. Run utils/init.sh first." >&2
  exit 1
fi

mkdir -p "${BIN_DIR}"
export PATH="${BIN_DIR}:${PATH}"

# Load secrets (needed for dotfiles clone + agent auth).
[ -f "${ENV_FILE}" ] && set -a && . "${ENV_FILE}" && set +a

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
    curl -fsSL "https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz" \
      | tar xz -C "${TOOLS_DIR}"
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
    [ -d "${DOTFILES_DIR}/dotfiles/.config/nvim" ] \
      && cp -r "${DOTFILES_DIR}/dotfiles/.config/nvim/." "${NVIM_CONFIG}/"
  fi
fi

# ─── opencode + Claude Code (npm prefix on the project drive) ─────────────────
# npm global installs into the system prefix can fail silently; point npm at a
# writable, persistent prefix and symlink the binaries onto PATH.

NPM_DIR="${TOOLS_DIR}/npm-global"
mkdir -p "${NPM_DIR}"
npm config set prefix "${NPM_DIR}"

if [ "${INSTALL_OPENCODE}" -eq 1 ] && [ ! -x "${BIN_DIR}/opencode" ]; then
  echo "--- Installing opencode ---"
  npm install -g opencode-ai
  [ -f "${NPM_DIR}/bin/opencode" ] && ln -sf "${NPM_DIR}/bin/opencode" "${BIN_DIR}/opencode" \
    || echo "WARNING: opencode binary not found after install" >&2
fi

if [ "${INSTALL_CLAUDE}" -eq 1 ] && [ ! -x "${BIN_DIR}/claude" ]; then
  echo "--- Installing Claude Code ---"
  npm install -g @anthropic-ai/claude-code
  [ -f "${NPM_DIR}/bin/claude" ] && ln -sf "${NPM_DIR}/bin/claude" "${BIN_DIR}/claude" \
    || echo "WARNING: claude binary not found after install" >&2
fi

# ─── Write the overlay env file and auto-source it (separate from env.sh) ─────

cat > "${DEVENV_SH}" <<EOF
# devenv.sh — dev/agent overlay environment (interactive sessions only).
# Auto-generated by utils/devenv.sh. Kept separate from the runtime env.sh so
# batch/GPU jobs never pick up the editor/agent harness vars.
export PATH="${BIN_DIR}:\$PATH"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME}"
export XDG_DATA_HOME="${XDG_DATA_HOME}"
export XDG_STATE_HOME="${XDG_STATE_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
EOF

SOURCE_LINE="[ -f '${DEVENV_SH}' ] && source '${DEVENV_SH}'"
for rc in "${HOME}/.bashrc" "${HOME}/.profile" "${HOME}/.bash_profile"; do
  grep -qF "${DEVENV_SH}" "${rc}" 2>/dev/null || echo "${SOURCE_LINE}" >> "${rc}"
done

echo "=== devenv.sh complete $(date) ==="
echo "Open a new shell (or 'source ${DEVENV_SH}') to pick up nvim / claude / opencode."
