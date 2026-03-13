#!/bin/bash
# Submete todos os configs test ao Slurm.
# Uso: ./slurm/run_all_tests.sh

set -e
cd "$(dirname "$0")/.."

sbatch slurm/run_benchmark.sh configs/diseases_test_no_cot.yaml
sbatch slurm/run_benchmark.sh configs/diseases_test_fo_no_cot.yaml
sbatch slurm/run_benchmark.sh configs/diseases_test_cot.yaml
sbatch slurm/run_benchmark.sh configs/diseases_test_fo_cot.yaml
sbatch slurm/run_benchmark.sh configs/objects_test_fo_no_cot.yaml
sbatch slurm/run_benchmark.sh configs/objects_test_po_no_cot.yaml
sbatch slurm/run_benchmark.sh configs/objects_test_po_cot.yaml
sbatch slurm/run_benchmark.sh configs/objects_test_fo_cot.yaml
