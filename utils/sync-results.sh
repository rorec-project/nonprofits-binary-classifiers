#!/usr/bin/env bash
# utils/sync-results.sh — pull UCloud processed outputs to your LOCAL machine.
#
# Run this on your laptop (NOT on the UCloud job). No SSHFS / IDE-remote needed.
#   bash utils/sync-results.sh <ssh-port> [dest-dir]
# The <ssh-port> is shown in the UCloud job details after the job starts and
# changes per job, so it is a positional argument; the drive name is config.

set -euo pipefail

. "$(dirname "$0")/config.sh"

PORT="${1:?usage: sync-results.sh <ssh-port> [dest-dir]}"
DEST="${2:-./local-results}"

mkdir -p "${DEST}"
rsync -av -e "ssh -p ${PORT}" \
  "ucloud@ssh.cloud.sdu.dk:/work/${PROJECT_DRIVE}/outputs/processed/" \
  "${DEST}/"
