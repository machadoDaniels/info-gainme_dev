#!/bin/bash
# Sync outputs/models/ from remote DGX nodes to this node.
# Usage: bash dgx/sync_outputs.sh [h3|b200|all]
#
# Nodes:
#   h3   — dgx-H100-03  10.100.0.113
#   b200 — dgx-B200-1   10.100.0.121
#   all  — both (sequential)

set -uo pipefail

PROJECT=/raid/user_danielpedrozo/projects/info-gainme_dev
DEST="$PROJECT/outputs/models/"
LOGS="$PROJECT/logs"
mkdir -p "$LOGS"

IP_H3=10.100.0.113
IP_B200=10.100.0.121

TARGET="${1:-all}"
TS=$(date +%Y%m%d_%H%M%S)

sync_node() {
    local name="$1"
    local ip="$2"
    local logfile="$LOGS/sync_${name}_${TS}.log"

    echo "==> Syncing from $name ($ip) → $DEST"
    echo "    Log: $logfile"
    echo "    Started: $(date)"

    rsync -rlv --omit-dir-times --ignore-existing \
        "${ip}:${DEST}" \
        "$DEST" \
        >> "$logfile" 2>&1
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
        echo "Usage: $0 [h3|b200|all]"
        exit 1
        ;;
esac
