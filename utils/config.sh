#!/usr/bin/env bash
# utils/config.sh — shared UCloud drive configuration for the binary-classifier pipeline.
#
# Edit this file on first use. Sourced by utils/init.sh, utils/run.sh,
# utils/devenv.sh, and utils/sync-results.sh — one place to set drive names
# and git identity, instead of duplicating them across four scripts.
#
# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this block on first use
# ════════════════════════════════════════════════════════════════════════════

# Mounted directory NAMES under /work for the two Drives. These are LOAD-BEARING
# and must match what you observe in `ls /work` on a live job:
#   - a member-files drive often mounts with a #NNNN suffix
#     (e.g. "AlessandroPizzigolotto#6144")
#   - a user-created drive usually mounts as its (possibly sanitized) Title.
# The values below are PLACEHOLDERS — confirm them on the first job and edit.
DATA_DRIVE="CHANGE_ME_DATA_DRIVE"       # Drive A: shared corpus + reusable caches
PROJECT_DRIVE="CHANGE_ME_PROJECT_DRIVE" # Drive B: this repo + heavy outputs

# Git identity for commits made from UCloud. Change to your own name/email
# or use your GitHub noreply address for privacy.
GIT_NAME="YOUR_NAME"
GIT_EMAIL="YOUR_EMAIL"

# ════════════════════════════════════════════════════════════════════════════
# END CONFIGURATION — do not edit below unless you know what you are doing
# ════════════════════════════════════════════════════════════════════════════
