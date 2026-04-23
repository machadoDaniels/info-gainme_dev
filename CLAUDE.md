# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Info Gainme is a benchmark that measures **information gain** in LLM conversations using a three-agent architecture:

- **Seeker** — asks yes/no questions to identify a secret target (city, object, or disease)
- **Oracle** — knows the target and answers truthfully
- **Pruner** — eliminates candidates from the pool based on each Q&A pair

Information gain is Shannon entropy reduction: `H = log2(N)` where N is the number of active candidates. The benchmark records win rate, total IG, avg IG/turn, and compliance rate.

## Development setup

**Python version:** `pyproject.toml` requires Python ≥3.12.

**Dependencies:** Install via pip or `uv`:
```bash
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt
```

**Environment:** Create a `.env` file with:
```
OPENAI_API_KEY=sk-...  # Required even for local vLLM endpoints
HF_TOKEN=hf_...        # Required for vLLM to download models from HuggingFace
```

## Testing

There is no automated test suite. Validate changes manually using the demo scripts (see below).

## Local testing & debugging

**Run a single game** to test locally:
```bash
python3 demo_single_game.py                    # generic test
python3 demo_objects_game.py                   # objects domain
python3 demo_diseases_game.py                  # diseases domain
```

These demos use a small subset of candidates and don't require large-scale infrastructure. Useful for testing agent logic or configs.

**Run a single benchmark config:**
```bash
python3 benchmark_runner.py --config configs/full/8b/geo_160_8b_fo_cot.yaml
```

Results write to `outputs/` and are resumable — re-running the same config skips completed `(target_id, run_index)` pairs.

**Run the human baseline** (interactive CLI — you play as the Seeker):
```bash
python3 human_benchmark_runner.py --config configs/human/geo_20_human_fo.yaml
python3 human_benchmark_runner.py --config configs/human/geo_20_human_fo.yaml --num-games 5 --seed 42
```

Configs in `configs/human/` cover all three domains × FO/PO. Oracle and Pruner remain LLM-powered (Qwen3-8B). Results are saved to `outputs/` in the same structure as automated benchmarks and are compatible with the full analysis pipeline. Use `--num-games 0` to cycle through all targets. Ctrl+C stops after the current game finishes.

## Running benchmarks

**Automated (recommended): Single SLURM job with vLLM + benchmarks**

Start vLLM and run all configs in one job. `CONFIGS_TARGET` is passed via `--export`, not as a positional arg:

```bash
# pasta inteira
sbatch --partition=h100n2 --gres=gpu:2 \
  --export=ALL,MODEL1=Qwen/Qwen3-4B-Thinking-2507,MODEL1_NAME=Qwen3-4B-Thinking-2507,MODEL2=Qwen/Qwen3-8B,MODEL2_NAME=Qwen3-8B,MODE=dual,CONFIGS_TARGET=configs/full/4b/cot/ \
  dgx/run_full_benchmark.sh
# config individual: aponte CONFIGS_TARGET para o .yaml específico
```

This script:
1. Allocates the GPUs requested via `--gres` (usually `gpu:2` for dual mode)
2. Starts vLLM servers in background with `wait_vllm_ready` (abort if process dies or readiness times out)
3. Creates `.servers_override_<JOBID>.yaml` with the real node IP — passed to `benchmark_runner.py` via `--servers-override`
4. Runs all configs under `CONFIGS_TARGET` sequentially (with resume)
5. Kills vLLM and cleans up the override file when done

Defaults in the script: `Qwen3-4B-Thinking-2507` (seeker) + `Qwen3-8B` (oracle/pruner) in `dual` mode. **Always override via `--export=ALL,...`** — the old "positional arg" style is not supported.

Key overridable vars (all via `--export=ALL,KEY=VAL,...`):
- `MODEL1`, `MODEL1_NAME` — seeker HF id (for vLLM download) + served-model-name. **⚠️ `MODEL1_NAME` MUST match the `seeker.model` string in the target YAML verbatim** (usually short form, no `org/` prefix). The script generates `.servers_override_<JOBID>.yaml` keyed by `MODEL1_NAME`; if the YAML asks for `"Llama-3.2-1B-Instruct"` and you pass `MODEL1_NAME=meta-llama/Llama-3.2-1B-Instruct`, `config_loader.py` won't find the key, falls back to `OPENAI_API_KEY` → hits `api.openai.com` → 404 loop for hours (non-fatal, silently retries). Same rule for `MODEL2_NAME` ↔ `oracle.model`/`pruner.model`.
- `MODEL2`, `MODEL2_NAME` — oracle/pruner HF id + served-model-name (same caveat)
- `MODE=single|dual` (default `single`; use `dual` when seeker ≠ oracle model)
- `CONFIGS_TARGET` — folder or single `.yaml`
- `MODEL1_PORT`, `MODEL2_PORT` — override auto-assigned ports (base formula `8000 + (JOB_ID % 500) * 10`; script also probes with `ss -tln` and advances if busy)
- `MODEL1_MAX_LEN`, `MODEL2_MAX_LEN` — vLLM `--max-model-len` per model (default 32000)
- `MODEL1_GPU_MEM`, `MODEL2_GPU_MEM` — `--gpu-memory-utilization` (default 0.90)
- `MODEL1_REASONING_PARSER`, `MODEL2_REASONING_PARSER` — optional `--reasoning-parser` flag
- `VLLM_MAX_NUM_SEQS` (default 32), `VLLM_ENFORCE_EAGER` (auto by partition), `VLLM_ENGINE_READY_TIMEOUT_S` (default 1800)
- `RUNS_PER_TARGET` — forwarded to `benchmark_runner.py --runs-per-target` (override `dataset.runs_per_target` in the YAML)

**External seeker** (seeker endpoint already in `configs/servers.yaml`): brings up only oracle/pruner locally.
```bash
sbatch dgx/run_external_seeker_benchmark.sh configs/full/235b/no_cot/
sbatch dgx/run_external_seeker_benchmark.sh configs/full/llama-70b/no_cot/
```
To add a new external-seeker model: (1) add endpoint to `configs/servers.yaml`; (2) create configs in `configs/full/<model>/no_cot/` (and `cot/` if the model supports thinking); (3) submit as above.

Monitor with: `watch squeue -u $USER` and `tail -f logs/info-gainme-full-<JOBID>.out`

**Manual vLLM + screen (alternative):**
```bash
sbatch dgx/run_vllm_single_model.sh          # single model (Singularity)
sbatch dgx/run_vllm_multimodel.sh            # two models on same node (Singularity)
sbatch dgx/run_vllm_conda.sh                 # single model via Conda (not Singularity; includes SSH tunnel instructions for local access)
screen -dmS benchmarks bash -c 'bash dgx/run_benchmarks_screen.sh configs/full/8b/ 2>&1 | tee logs/screen-8b-all.out; exec bash'
```

**SLURM-only benchmarks** (when vLLM is already running):
```bash
bash dgx/run_benchmarks_slurm.sh configs/full/8b/            # folder
bash dgx/run_benchmarks_slurm.sh configs/full/8b/foo.yaml    # single config
```

**Single config locally (for testing):**
```bash
python3 benchmark_runner.py --config configs/full/8b/geo_160_8b_fo_cot.yaml
```

**Resumability:** Benchmarks detect completed `(target_id, run_index)` pairs from the existing `runs.csv` and skip them automatically. Useful for recovering from crashes or extending a partial run. The `BenchmarkRunner` reads `runs.csv`, filters targets that haven't been fully run, and continues where it left off. To restart from scratch, remove or rename the output directory.

## Job sanity check — is it actually producing?

A SLURM job in `R` state ≠ job producing runs. Always confirm with this checklist:

1. **vLLMs alive**: `curl -s http://localhost:<port>/v1/models` returns JSON with the expected `served-model-name` (both seeker and oracle/pruner). `ps | grep vllm.*api_server` shows both processes.
2. **Override file exists**: `.servers_override_<JOBID>.yaml` in the project root, keys matching `MODEL*_NAME`.
3. **`runs.csv` growing**: `wc -l outputs/.../<exp>/runs.csv` + `stat` mtime ≤ few minutes old. A CSV with **1 line (header only)** after hours of runtime = job is broken, not running slow.
4. **No OpenAI fallback in logs**: `grep api.openai.com logs/info-gainme-full-<JOBID>.out` should return nothing. Any hit means `MODEL_NAME` ↔ `seeker.model`/`oracle.model`/`pruner.model` mismatch → config_loader fell back to `OPENAI_API_KEY`.
5. **No `"died before readiness"`**: script's `wait_vllm_ready` aborts if vLLM subprocess exits early. If you see this in `.out`, check the `.log` of that vLLM — usually OOM, bad path, or CUDA arch mismatch (B200 needs `enforce-eager=false`).

**Common failure patterns:**
- `runs.csv` only header, mtime progressing every ~7h → `MODEL_NAME` mismatch; each config exhausts 50 retries × ~60s backoff then moves on. Cancel and resubmit with `MODEL_NAME` matching YAML exactly.
- `runs.csv` mtime frozen for hours while job still `R` → vLLM died silently but the bash script keeps polling. Check the vLLM `.log`, likely OOM or a downstream dependency (the other vLLM) crashed.
- Oracle/pruner port collision between consecutive JOB_IDs → was the old `%1000` bug, fixed to `%500 * 10 + ss probe`. If you see `"model X does not exist"` in the seeker's log pointing to the OTHER job's port, the fix didn't take effect — verify `dgx/run_full_benchmark.sh` is synced.
- Pruner returning invalid JSON (Nemotron/Olmo sometimes emit control chars) → `Pruner JSON parse error` in logs; benchmark skips pruning that turn (`active=N, pruned=0`). Run still completes but produces degenerate data (30 turns, IG=0). Consider changing oracle/pruner model.

**Auditing what's left across nodes:**
```bash
# DONE / INCOMPLETE / MISSING per config, grouped by folder:
python3 scripts/audit_experiments.py                                                          # current node
ssh 10.100.0.113 'cd /raid/user_danielpedrozo/projects/info-gainme_dev && python3 scripts/audit_experiments.py'  # other node
```
The script reads each YAML under `configs/full/**`, computes `expected = num_targets × runs_per_target`, and counts unique `(target_id, run_index)` pairs in the corresponding `outputs/models/s_<triple>/<exp>/runs.csv`. Take the max across nodes when a config is split.

## Analysis pipeline

Run in order after benchmarks finish:

**Step 1: Compute experiment metrics**
```bash
bash dgx/run_analyze_results.sh                         # all experiments in outputs/
bash dgx/run_analyze_results.sh outputs/models/.../runs.csv  # specific CSV
```
Outputs: `summary.json` (metrics: win rate, avg IG, turns), `variance.json` (per-target variance), and `outputs/unified_experiments.csv` (all experiments merged).

**Step 2: Synthesize reasoning traces** (CoT experiments only)
```bash
bash dgx/run_synthesize_traces.sh                                          # use gpt-4o-mini
MODEL=Qwen3-8B BASE_URL=http://localhost:8020/v1 bash dgx/run_synthesize_traces.sh  # local model
```
For each CoT game, extracts `<think>` blocks from `seeker.json` and synthesizes structured reasoning (options considered, choice rationale) via LLM. Idempotent — skips if `seeker_traces.json` exists. Output: `seeker_traces.json` per conversation.

Parallelism is two-level: `WORKERS` (default 8) = conversations in parallel per experiment; `TURN_WORKERS` (default 4) = LLM calls parallelized within a conversation. Max concurrent LLM calls ≈ `workers × turn_workers`. Override via `--turn-workers` flag or `TURN_WORKERS` env var.

**Step 3: Analyze reasoning traces**
```bash
bash dgx/run_analyze_traces.sh                  # default outputs/
bash dgx/run_analyze_traces.sh /custom/outputs  # custom directory
```
Aggregates all `seeker_traces.json` (CoT only) to generate `reasoning_traces_analysis.json` with question frequency, decision patterns, and per-model aggregations.

## Configuration

**Experiment configs** live in `configs/full/<model>/` as YAML files. Human baseline configs live in `configs/human/`. Each specifies:
- Models for seeker, oracle, pruner (agent LLM configs); set `model: "human"` on the seeker for the human baseline — no vLLM endpoint needed
- Dataset type (`geo`, `objects`, `diseases`) and CSV path
- Observability mode (`FULLY_OBSERVABLE` / `PARTIALLY_OBSERVABLE`)
- Experiment name (used in output folder naming)

**Server endpoints:** Centralized in `configs/servers.yaml` (model name → vLLM URL). `config_loader.py` walks up from any config file to find this automatically. **Never hardcode URLs in individual configs.** Update `servers.yaml` when vLLM endpoints move.

`config_loader.py` also resolves model URLs via environment variables as a fallback: for a model named `Qwen3-8B`, it checks `VLLM_QWEN3_8B` (uppercased, non-alphanumeric → `_`). Priority: `base_url` in config > env var > `servers.yaml`.

**Server override file:** `run_full_benchmark.sh` creates `.servers_override_<JOBID>.yaml` at runtime with the real node IP. Pass it to `benchmark_runner.py` via `--servers-override` if running manually after the job starts.

**CoT (Chain-of-Thought):** Enabled on the seeker via:
```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
use_reasoning: true
```
No-CoT configs omit both fields entirely.

**Observability modes:**
- `FULLY_OBSERVABLE` — Seeker sees the full candidate list each turn
- `PARTIALLY_OBSERVABLE` — Seeker sees only the Q&A history

## Code architecture

```
benchmark_runner.py        ← CLI entrypoint; loads config, dataset, runs BenchmarkRunner
human_benchmark_runner.py  ← Interactive CLI runner for the human seeker baseline
src/
  benchmark_config.py      ← BenchmarkConfig dataclass (agent configs + game settings)
  benchmark.py             ← BenchmarkRunner: ThreadPoolExecutor, incremental CSV writes
  orchestrator.py          ← Single game loop: Seeker→Oracle→Pruner per turn
  candidates.py            ← CandidatePool (flat list with active/pruned tracking)
  graph.py                 ← KnowledgeGraph (Node, Edge) with hierarchical pruning for geo domain
  entropy.py               ← Shannon entropy: H=log2(N), info_gain = H_before - H_after
  data_types.py            ← TurnState, Question, Answer, PruningResult, ObservabilityMode
  agents/
    llm_config.py          ← LLMConfig dataclass
    llm_adapter.py         ← OpenAI-compatible HTTP adapter (history, reasoning capture)
    seeker.py              ← SeekerAgent
    human_seeker.py        ← HumanSeekerAgent (drop-in replacement; reads questions from stdin)
    oracle.py              ← OracleAgent
    pruner.py              ← PrunerAgent (returns keep_labels via structured JSON)
  domain/
    types.py               ← DomainConfig + GEO_DOMAIN / OBJECTS_DOMAIN / DISEASES_DOMAIN
    geo/loader.py          ← loads cities CSV → KnowledgeGraph (hierarchical: region→country→city)
    objects/loader.py      ← loads objects CSV → CandidatePool
    diseases/loader.py     ← loads diseases CSV → CandidatePool
  analysis/
    data_types.py          ← GameRun, CityStats, ExperimentResults dataclasses
    loader.py              ← load_experiment_results: runs.csv → ExperimentResults (with token counts)
    writer.py              ← save_summary / save_city_variance → summary.json, variance.json
    reasoning_synthesis.py ← extracts <think> traces; synthesizes structured reasoning via LLM
    question_evaluator.py  ← evaluate_seeker_choices: re-simulates Oracle+Pruner for each
                             considered question to rank them by info gain (read-only)
  utils/config_loader.py   ← YAML → BenchmarkConfig; resolves model names via servers.yaml
  logging_config.py        ← centralized logging setup (called at process start)
  prompts/                 ← Markdown system prompts for each agent (templated)
scripts/
  analyze_results.py              ← runs.csv → summary.json + variance.json (wraps analysis/writer)
  generate_unified_csv.py         ← merges all experiments into outputs/unified_experiments.csv
  generate_model_summary_csv.py   ← per-model aggregation CSV
  aggregate_metrics_by_city.py    ← city-level metric aggregation
  aggregate_ig_over_time.py       ← IG-over-turns aggregation
  aggregate_cold_start.py         ← aggregate cold-start results
  plot_aggregated_ig.py           ← plot IG-over-turns curves
  compute_optimal_baseline.py     ← optimal-play upper-bound baseline
  extract_top_cities_by_population.py  ← data prep helper
  prepare_diseases_csv.py         ← prepare diseases CSV for dataset creation
  remove_duplicates_runs.py       ← de-dup runs.csv by (target_id, run_index)
  validate_oracle_answers.py      ← re-check Oracle answers against the ground truth
  delete_affected_runs.py         ← remove runs affected by oracle bugs
  delete_evaluations_with_connection_errors.py
  recalculate_question_evaluation_se.py
  download_from_hf.py / upload_to_hf.py  ← HuggingFace dataset sync (see also dgx/ shell wrappers)
  reasoning_traces/               ← CoT trace synthesis + question-choice evaluation
    synthesize_traces.py              ← batch synthesize traces (--all / --runs / --seeker-file)
    analyze_traces.py                 ← seeker_traces.json → reasoning_traces_analysis.json
    evaluate_all_seeker_choices.py    ← batch question-choice evaluation from runs.csv
    evaluate_seeker_choices.py        ← single-conversation question-choice evaluation
    generate_question_evaluations_csv.py  ← flatten question_evaluation.json → CSV
    summary_table.py                  ← decision-quality summary table (Table 2)
  question_classification/        ← post-hoc classification of seeker questions
    classify_questions.py
    analyze_question_classifications.py
    flatten_question_classifications.py
```

Note: the `dgx/` shell wrappers (e.g. `run_synthesize_traces.sh`, `run_analyze_traces.sh`) still work — they were updated to point at the new `scripts/reasoning_traces/` locations.

**Key flow:** `benchmark_runner.py` / `human_benchmark_runner.py` → `BenchmarkRunner.run()` → per game: `Orchestrator.from_target()` → loop: Seeker asks → Oracle answers → Pruner prunes → entropy computed → `TurnState` appended → results written to `runs.csv`.

**Human seeker integration:** `Orchestrator.from_target()` checks `seeker_config.model == "human"` and instantiates `HumanSeekerAgent` instead of `SeekerAgent` + `LLMAdapter`. `HumanSeekerAgent` implements the same interface (`question_to_oracle`, `add_oracle_answer_and_pruning`, etc.) and uses a `_MockLLMAdapter` so `export_conversation()` still produces a valid `seeker.json`.

**Parallelism:** `BenchmarkRunner` runs games concurrently via `ThreadPoolExecutor(max_workers=N)`. CSV writes are serialized with a `threading.Lock`. Each thread gets a `copy.deepcopy(pool)` to avoid shared state.

**Geo vs flat domains:** The geo domain uses `KnowledgeGraph` (hierarchical tree: region→subregion→country→state→city). When all cities under a parent are pruned, the parent is also pruned recursively (`apply_pruning` walks up via `has_child`/`contains` edges). Objects and diseases use the flat `CandidatePool` instead.

**Question-choice evaluation** (post-hoc, CoT only): `scripts/reasoning_traces/evaluate_all_seeker_choices.py` reads a `runs.csv`, finds conversations with `seeker_traces.json`, then for each turn re-runs Oracle+Pruner on every question the Seeker considered to compute counterfactual info gains. Results saved as `question_evaluation.json` per conversation and `question_evaluations_summary.json` per experiment. This pipeline is read-only — it never modifies turns or conversation files.

## Utility scripts

**Post-processing & data maintenance:**
- `scripts/audit_experiments.py` — walks `configs/full/**/*.yaml`, reports DONE / INCOMPLETE / MISSING per config by counting unique `(target_id, run_index)` pairs in each `runs.csv`. Use to find gaps before resubmitting.
- `scripts/remove_duplicates_runs.py` — remove duplicate rows from runs.csv by `(target_id, run_index)` pair
- `scripts/reasoning_traces/evaluate_seeker_choices.py` — evaluate a single conversation's question choices (debug-friendly version of batch evaluator)
- `scripts/delete_evaluations_with_connection_errors.py` — clean up failed evaluation runs
- `scripts/recalculate_question_evaluation_se.py` — recalculate standard error for existing evaluations

## Output structure

```
outputs/
├── unified_experiments.csv
├── reasoning_traces_analysis.json
└── models/s_<seeker>__o_<oracle>__p_<pruner>/<experiment>/
    ├── runs.csv
    ├── summary.json
    ├── variance.json
    └── conversations/<target>/
        ├── seeker.json / oracle.json / pruner.json
        ├── turns.jsonl
        ├── metadata.json
        └── seeker_traces.json   ← generated by run_synthesize_traces.sh
```

## Infrastructure

**Compute:**
- Models served via **vLLM** (OpenAI-compatible API)
- Benchmarks & analysis run inside **Singularity** container (`/raid/user_danielpedrozo/images/vllm_openai_latest.sif`)
- Runs on DGX nodes: `h100n2` and `h100n3` (8× H100 80GB each) and `b200n1` (8× B200 180GB). `OPENAI_API_KEY` + `HF_TOKEN` must be in `.env`. Each node has its own independent `/raid` — see "Syncing between nodes" below.

**Singularity bind mounts:** The host path `/raid/user_danielpedrozo` is mounted as `/workspace` inside the container (`--bind /raid/user_danielpedrozo:/workspace --pwd /workspace`). Config file paths should use host paths; the scripts handle the translation.

**File permissions:**
- Shared group `sd22` — scripts wrap commands with `sg sd22 -c "..."` to ensure files created by Singularity are group-writable
- `umask 002` is set in dgx scripts so new files are group-writable by default

**vLLM orchestration (`run_full_benchmark.sh`):**
- Ports derived from `SLURM_JOB_ID` with gap of 10 between neighbours (`BASE_PORT = 8000 + (JOB_ID % 500) * 10`) + `ss -tln` check to avoid collisions when jobs land on the same node
- With 1 GPU all agents share one model; with 2+ GPUs seeker gets `MODEL1`, oracle/pruner get `MODEL2`
- `wait_vllm_ready` polls `GET /v1/models` with a timeout (default 1800s) and aborts early if the vLLM process dies (`kill -0` check) — avoids the classic "job stuck forever" when vLLM silently fails
- vLLM startup + download timeout is controlled by `VLLM_ENGINE_READY_TIMEOUT_S` (default 1800s; large models may need 3600s+)
- `VLLM_MAX_NUM_SEQS` tunes concurrent requests on the vLLM side (default 32)
- `VLLM_ENFORCE_EAGER` auto-detected by partition: **omitted on `b200*`** (CUDA graphs pay off on Blackwell), **enabled on `h100*`** (faster startup). Override via `--export=ALL,VLLM_ENFORCE_EAGER=true|false,...`
- Creates `.servers_override_<JOBID>.yaml` with the actual node IP; the script deletes it at the end, but a crash leaves stale files in the project root — safe to remove manually
- vLLM logs are at `logs/info-gainme-full-<JOBID>-vllm-<MODELNAME>.log` (stderr from both the host-side `singularity exec` and the in-container Python are merged there — useful when startup fails)

**Data creation scripts:** `data/create_geo_160.py` and `data/create_diseases_160.py` regenerate the dataset CSVs from external sources (GeoNames API, etc.) if the raw data needs to be rebuilt.

## Running on different GPU partitions

Each DGX node has its own `/raid` filesystem — the project dir, `outputs/`, `hf-cache/`, and `images/` are **not shared across nodes**. Clone/sync when moving between nodes (see below).

**Available partitions** (check with `sinfo`):

| Partition | Node | GPUs | Use case |
|---|---|---|---|
| `h100n2` | `dgx-H100-02` | 8× H100 80GB | default "daniel" partition |
| `h100n3` | `dgx-H100-03` | 8× H100 80GB | "julia" partition (see alias below) |
| `b200n1` | `dgx-B200-1` | 8× B200 180GB | Blackwell — largest VRAM |

**Partition choice via `sbatch --partition=<name>`** when submitting. The `run_full_benchmark.sh` script auto-detects `b200*` and adjusts vLLM flags (drops `--enforce-eager`).

**Examples:**

```bash
# H100 (daniel, partition h100n2)
sbatch --partition=h100n2 --gres=gpu:2 \
  --export=ALL,MODEL1=Qwen/Qwen3-4B-Thinking-2507,MODEL1_NAME=Qwen3-4B-Thinking-2507,MODEL2=Qwen/Qwen3-8B,MODEL2_NAME=Qwen3-8B,MODE=dual,CONFIGS_TARGET=configs/full/4b/cot/ \
  dgx/run_full_benchmark.sh

# B200 (daniel, partition b200n1 — CUDA graphs auto-enabled)
sbatch --partition=b200n1 --gres=gpu:2 \
  --export=ALL,VLLM_ENGINE_READY_TIMEOUT_S=3600,MODEL1=allenai/Olmo-3.1-32B-Think,MODEL1_NAME=Olmo-3.1-32B-Think,MODEL2=Qwen/Qwen3-8B,MODEL2_NAME=Qwen3-8B,MODE=dual,CONFIGS_TARGET=configs/full/olmo3-32b/cot/ \
  dgx/run_full_benchmark.sh
```

Large (30B+) models benefit from `VLLM_ENGINE_READY_TIMEOUT_S=3600` to cover the HuggingFace download + engine core init. Very large models (64GB+ weights) may need dual-GPU with tensor parallelism (`--tensor-parallel-size` — not currently exposed by `run_full_benchmark.sh`; use `dgx/run_vllm_single_model.sh` instead or add the flag manually).

## Running as another user via `asjulia` alias

The project is frequently split so that `user_juliadollis` runs on `h100n3` and `user_danielpedrozo` runs on `h100n2` / `b200n1`. An alias `asjulia` is defined in `~/.bashrc`:

```bash
asjulia() { ssh -t user_juliadollis@localhost "bash -i -lc $(printf '%q' "$*")"; }
```

It runs a command as `user_juliadollis` via local SSH. Julia has read access to the project directory under `/raid/user_danielpedrozo/` on each node. Writing requires the shared group `sd22` + `g+w` on target dirs (see File permissions above).

Usage pattern (Claude Code can invoke this via Bash):
```bash
bash -ic 'asjulia "cd /raid/user_danielpedrozo/projects/info-gainme_dev; sbatch --partition=h100n3 --gres=gpu:2 --export=ALL,...,CONFIGS_TARGET=... dgx/run_full_benchmark.sh"'
```

Gotchas:
- Bash interpretation: inside the outer double quotes, `&&` is interpreted by the outer shell. Use `;` between commands to keep them on the SSH side (or escape: `\&\&`).
- Job submitter must `cd` into the project directory; SSH starts in `/home/<user>` by default.
- Check julia's queue: `bash -ic 'asjulia "squeue -u user_juliadollis"'` — equivalent alias `sqj` exists in `.bashrc`.

## Syncing between nodes

Each DGX has its own `/raid` (separate filesystems) — use `rsync` to move configs/code/outputs:

```bash
# sync scripts + configs n2 → n3 (excluding large stuff)
rsync -av --exclude='outputs' --exclude='logs' --exclude='hf-cache' --exclude='.git' \
  --exclude='.venv' --exclude='*.sif' \
  /raid/user_danielpedrozo/projects/info-gainme_dev/ \
  user_danielpedrozo@10.100.0.113:/raid/user_danielpedrozo/projects/info-gainme_dev/

# use git pull in H100-03 if remote is tracked
ssh 10.100.0.113 'cd /raid/user_danielpedrozo/projects/info-gainme_dev; git fetch origin; git reset --hard origin/main'
```

**Node IPs** (internal network):
- `dgx-H100-02` — `10.100.0.112`
- `dgx-H100-03` — `10.100.0.113`
- `dgx-B200-1` — `10.100.0.121`

**Cross-user rsync:** if files on the target belong to another user, pull from the target as that user instead of pushing:
```bash
bash -ic 'asjulia "cd /raid/user_danielpedrozo/projects/info-gainme_dev; \
  rsync -a --update user_juliadollis@10.100.0.112:/raid/user_danielpedrozo/projects/info-gainme_dev/outputs/models/<triple>/ \
    outputs/models/<triple>/"'
```
