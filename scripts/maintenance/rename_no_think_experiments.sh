#!/bin/bash
# Rename experiment directories where the Qwen3-8B oracle ran without thinking.
#
# Background:
#   When vLLM is started without --reasoning-parser qwen3 AND the oracle
#   request uses response_format={"type":"json_schema","strict":true},
#   constrained decoding forces the first generated token to be `{` — leaving
#   no room for a <think> block. The oracle answers without reasoning, while
#   the pruner (no response_format) still thinks. This affected 39 experiments
#   silently. Audit confirmed the first-turn assistant message has no <think>
#   in those 39 cases (check via reasoning_history of any conversation).
#
# This script renames each affected experiment directory by appending
# "_no_think" so:
#   - the existing data is preserved next to its peers (audit trail);
#   - re-running the benchmark with the parser-fix in dgx/run_full_benchmark.sh
#     creates a fresh directory under the original (canonical) name.
#
# Usage:
#   bash scripts/maintenance/rename_no_think_experiments.sh           # dry-run
#   bash scripts/maintenance/rename_no_think_experiments.sh apply     # rename
#
# After applying, regenerate any aggregate CSVs that index by experiment name:
#   python3 scripts/analysis/generate_unified_csv.py
#   python3 scripts/judge_eval/aggregate_judge_results.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUTS_DIR="${OUTPUTS_DIR:-${PROJECT_ROOT}/outputs/models}"
SUFFIX="${SUFFIX:-_no_think}"
APPLY="${1:-dry}"

PATHS=(
    # Gemma seeker (6) — no_cot variants
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/diseases_160_gemma12b_fo_no_cot"
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/diseases_160_gemma12b_po_no_cot"
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/geo_160_gemma12b_fo_no_cot"
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/geo_160_gemma12b_po_no_cot"
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/objects_158_gemma12b_fo_no_cot"
    "s_Gemma-3-12B-IT__o_Qwen3-8B__p_Qwen3-8B/objects_158_gemma12b_po_no_cot"
    # Llama-1B (5)
    "s_Llama-3.2-1B-Instruct__o_Qwen3-8B__p_Qwen3-8B/diseases_160_llama1b_po_no_cot"
    "s_Llama-3.2-1B-Instruct__o_Qwen3-8B__p_Qwen3-8B/geo_160_llama1b_fo_no_cot"
    "s_Llama-3.2-1B-Instruct__o_Qwen3-8B__p_Qwen3-8B/geo_160_llama1b_po_no_cot"
    "s_Llama-3.2-1B-Instruct__o_Qwen3-8B__p_Qwen3-8B/objects_158_llama1b_fo_no_cot"
    "s_Llama-3.2-1B-Instruct__o_Qwen3-8B__p_Qwen3-8B/objects_158_llama1b_po_no_cot"
    # Llama-3B (3)
    "s_Llama-3.2-3B-Instruct__o_Qwen3-8B__p_Qwen3-8B/diseases_160_llama3b_fo_no_cot"
    "s_Llama-3.2-3B-Instruct__o_Qwen3-8B__p_Qwen3-8B/diseases_160_llama3b_po_no_cot"
    "s_Llama-3.2-3B-Instruct__o_Qwen3-8B__p_Qwen3-8B/geo_160_llama3b_fo_no_cot"
    # Nemotron-Cascade-8B-Thinking seeker (1)
    "s_Nemotron-Cascade-8B-Thinking__o_Qwen3-8B__p_Qwen3-8B/geo_160_nemotron8b_po_cot_with_kickoff"
    # Nemotron-Cascade-8B seeker (6) — _with_kickoff variants
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/diseases_160_nemotron8b_po_cot_with_kickoff"
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/diseases_160_nemotron8b_po_no_cot_with_kickoff"
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/geo_160_nemotron8b_po_cot_with_kickoff"
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/geo_160_nemotron8b_po_no_cot_with_kickoff"
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/objects_158_nemotron8b_po_cot_with_kickoff"
    "s_Nemotron-Cascade-8B__o_Qwen3-8B__p_Qwen3-8B/objects_158_nemotron8b_po_no_cot_with_kickoff"
    # Olmo-Instruct (6)
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/diseases_160_olmo3_32b_instruct_fo_no_cot"
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/diseases_160_olmo3_32b_instruct_po_no_cot"
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/geo_160_olmo3_32b_instruct_fo_no_cot"
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/geo_160_olmo3_32b_instruct_po_no_cot"
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/objects_158_olmo3_32b_instruct_fo_no_cot"
    "s_Olmo-3.1-32B-Instruct__o_Qwen3-8B__p_Qwen3-8B/objects_158_olmo3_32b_instruct_po_no_cot"
    # Olmo-Think (6)
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/diseases_160_olmo3_32b_think_fo_cot"
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/diseases_160_olmo3_32b_think_po_cot"
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/geo_160_olmo3_32b_think_fo_cot"
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/geo_160_olmo3_32b_think_po_cot"
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/objects_158_olmo3_32b_think_fo_cot"
    "s_Olmo-3.1-32B-Think__o_Qwen3-8B__p_Qwen3-8B/objects_158_olmo3_32b_think_po_cot"
    # Qwen3-0.6B with_kickoff (6)
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/diseases_160_0.6b_po_cot_with_kickoff"
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/diseases_160_0.6b_po_no_cot_with_kickoff"
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/geo_160_0.6b_po_cot_with_kickoff"
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/geo_160_0.6b_po_no_cot_with_kickoff"
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/objects_158_0.6b_po_cot_with_kickoff"
    "s_Qwen3-0.6B__o_Qwen3-8B__p_Qwen3-8B/objects_158_0.6b_po_no_cot_with_kickoff"
)

cd "${OUTPUTS_DIR}"

echo "Outputs dir: ${OUTPUTS_DIR}"
echo "Suffix:      ${SUFFIX}"
echo "Mode:        ${APPLY}"
echo "Total:       ${#PATHS[@]} experiments"
echo "---"

ok=0; missing=0; collision=0
for path in "${PATHS[@]}"; do
    new="${path}${SUFFIX}"
    if [ ! -d "${path}" ]; then
        echo "MISSING:    ${path}"
        missing=$((missing + 1))
        continue
    fi
    if [ -e "${new}" ]; then
        echo "COLLISION:  ${new} already exists — skipping"
        collision=$((collision + 1))
        continue
    fi
    if [ "${APPLY}" = "apply" ]; then
        mv "${path}" "${new}"
        echo "RENAMED:    ${path} -> ${new}"
    else
        echo "WOULD MV:   ${path} -> ${new}"
    fi
    ok=$((ok + 1))
done

echo "---"
echo "OK: ${ok}  MISSING: ${missing}  COLLISIONS: ${collision}"
[ "${APPLY}" != "apply" ] && echo "(dry-run; rerun with 'apply' to execute)"
