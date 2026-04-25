#!/bin/bash
# Sync outputs/models/ between DGX nodes.
#
# Pull (default): remote → h2   (--ignore-existing, safe to run anytime)
# Push:           h2 → remotes  (overwrites, h2 is authoritative)
#
# Usage:
#   bash dgx/sync_outputs.sh [h3|b200|all]          # pull from remote(s)
#   bash dgx/sync_outputs.sh push [h3|b200|all]     # push from h2 to remote(s)
#
# Nodes:
#   h3   — dgx-H100-03  10.100.0.113
#   b200 — dgx-B200-1   10.100.0.121

set -uo pipefail

PROJECT=/raid/user_danielpedrozo/projects/info-gainme_dev
DEST="$PROJECT/outputs/models/"
LOGS="$PROJECT/logs"
mkdir -p "$LOGS"

IP_H3=10.100.0.113
IP_B200=10.100.0.121

# Parse args: optional "push" keyword, then target
MODE="pull"
if [ "${1:-}" = "push" ]; then
    MODE="push"
    shift
fi

TARGET="${1:-all}"
TARGET="${TARGET#--}"   # strip leading -- (--all → all, etc.)
TS=$(date +%Y%m%d_%H%M%S)

sync_node() {
    local name="$1"
    local ip="$2"
    local logfile="$LOGS/sync_${MODE}_${name}_${TS}.log"

    if [ "$MODE" = "push" ]; then
        echo "==> Pushing h2 → $name ($ip)"
        echo "    Log: $logfile"
        echo "    Started: $(date)"
        rsync -rlv --omit-dir-times \
            "$DEST" \
            "${ip}:${DEST}" \
            >> "$logfile" 2>&1
    else
        echo "==> Syncing from $name ($ip) → h2"
        echo "    Log: $logfile"
        echo "    Started: $(date)"
        rsync -rlv --omit-dir-times --ignore-existing \
            "${ip}:${DEST}" \
            "$DEST" \
            >> "$logfile" 2>&1
    fi

    local rc=$?
    local transferred
    transferred=$(grep -c '^s_' "$logfile" 2>/dev/null) || transferred=0

    if [ $rc -eq 0 ] || [ $rc -eq 23 ]; then
        echo "    Done: $(date) — ~${transferred} paths transferred (exit $rc)"
    else
        echo "    FAILED (exit $rc) — check $logfile"
    fi
    return 0
}

case "$TARGET" in
    h3)
        sync_node h3 "$IP_H3"
        ;;
    b200)
        sync_node b200 "$IP_B200"
        ;;
    all)
        sync_node h3 "$IP_H3"
        sync_node b200 "$IP_B200"
        ;;
    *)
        echo "Usage: $0 [push] [h3|b200|all]"
        exit 1
        ;;
esac
