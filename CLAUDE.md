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

## Running benchmarks

**Automated (recommended): Single SLURM job with vLLM + benchmarks**

Start vLLM and run all configs in one job:
```bash
sbatch dgx/run_full_benchmark.sh configs/full/4b/                  # all 4b configs
sbatch dgx/run_full_benchmark.sh configs/full/4b/cot/geo_160_4b_thinking_fo_cot.yaml  # single config
```

This script:
1. Allocates N GPUs (one per model)
2. Starts vLLM servers in background (with automatic readiness polling)
3. Updates `configs/servers.yaml` with the real node IP
4. Runs all configs from the given folder sequentially
5. Kills vLLM when done

Models are configurable at the top of `dgx/run_full_benchmark.sh`. Defaults: `Qwen3-4B-Thinking-2507` (seeker) + `Qwen3-8B` (oracle/pruner). Override via `sbatch --export=ALL,MODEL1=...,MODEL2=...,CONFIGS_TARGET=configs/full/4b/ dgx/run_full_benchmark.sh`. Key overridable vars: `MODEL1`, `MODEL1_NAME`, `MODEL2`, `MODEL2_NAME`, `MODE` (`single`/`dual`), `CONFIGS_TARGET`.

Monitor with: `watch squeue -u $USER` and `tail -f logs/info-gainme-full-<JOBID>.out`

**Manual vLLM + screen (alternative):**
```bash
sbatch dgx/run_vllm_single_model.sh          # single model
sbatch dgx/run_vllm_multimodel.sh            # two models on same node
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

**Step 3: Analyze reasoning traces**
```bash
bash dgx/run_analyze_traces.sh                  # default outputs/
bash dgx/run_analyze_traces.sh /custom/outputs  # custom directory
```
Aggregates all `seeker_traces.json` (CoT only) to generate `reasoning_traces_analysis.json` with question frequency, decision patterns, and per-model aggregations.

## Configuration

**Experiment configs** live in `configs/full/<model>/` as YAML files. Each specifies:
- Models for seeker, oracle, pruner (agent LLM configs)
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
  analyze_reasoning_traces.py     ← reads seeker_traces.json → reasoning_traces_analysis.json
  multi_synthesize_reasoning_traces.py  ← batch synthesize traces across experiments
  evaluate_all_seeker_choices.py  ← batch question-choice evaluation from runs.csv
  evaluate_seeker_choices.py      ← single-conversation question-choice evaluation
  aggregate_metrics_by_city.py    ← city-level metric aggregation
  aggregate_ig_over_time.py       ← IG-over-turns aggregation
  generate_question_evaluations_csv.py  ← flatten question_evaluation.json → CSV
  plot_aggregated_ig.py           ← plot IG-over-turns curves
  prepare_diseases_csv.py         ← prepare diseases CSV for dataset creation
  download_from_hf.py / upload_to_hf.py  ← HuggingFace dataset sync (see also dgx/ shell wrappers)
```

**Key flow:** `benchmark_runner.py` → `BenchmarkRunner.run()` → per game: `Orchestrator.from_target()` → loop: Seeker asks → Oracle answers → Pruner prunes → entropy computed → `TurnState` appended → results written to `runs.csv`.

**Parallelism:** `BenchmarkRunner` runs games concurrently via `ThreadPoolExecutor(max_workers=N)`. CSV writes are serialized with a `threading.Lock`. Each thread gets a `copy.deepcopy(pool)` to avoid shared state.

**Geo vs flat domains:** The geo domain uses `KnowledgeGraph` (hierarchical tree: region→subregion→country→state→city). When all cities under a parent are pruned, the parent is also pruned recursively (`apply_pruning` walks up via `has_child`/`contains` edges). Objects and diseases use the flat `CandidatePool` instead.

**Question-choice evaluation** (post-hoc, CoT only): `scripts/evaluate_all_seeker_choices.py` reads a `runs.csv`, finds conversations with `seeker_traces.json`, then for each turn re-runs Oracle+Pruner on every question the Seeker considered to compute counterfactual info gains. Results saved as `question_evaluation.json` per conversation and `question_evaluations_summary.json` per experiment. This pipeline is read-only — it never modifies turns or conversation files.

## Utility scripts

**Post-processing & data maintenance:**
- `synthesize_from_runs_csv.py` — batch synthesize reasoning traces from runs.csv with custom settings (alternative to `run_synthesize_traces.sh`)
- `remove_duplicates_runs.py` — remove duplicate rows from runs.csv by `(target_id, run_index)` pair
- `scripts/evaluate_seeker_choices.py` — evaluate a single conversation's question choices (debug-friendly version of batch evaluator)
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
- Benchmarks & analysis run inside **Singularity** container (`vllm_openai_latest.sif`)
- Typically on DGX H100 nodes with `OPENAI_API_KEY` in `.env`

**Singularity bind mounts:** The host path `/raid/user_danielpedrozo` is mounted as `/workspace` inside the container (`--bind /raid/user_danielpedrozo:/workspace --pwd /workspace`). Config file paths should use host paths; the scripts handle the translation.

**File permissions:**
- Shared group `sd22` — scripts wrap commands with `sg sd22 -c "..."` to ensure files created by Singularity are group-writable
- `umask 002` is set in dgx scripts so new files are group-writable by default

**vLLM orchestration (`run_full_benchmark.sh`):**
- Ports are derived from `SLURM_JOB_ID` (`BASE_PORT = 8000 + (JOB_ID % 1000)`) to avoid conflicts across concurrent jobs
- With 1 GPU all agents share one model; with 2+ GPUs seeker gets `MODEL1`, oracle/pruner get `MODEL2`
- Script polls `GET /v1/models` before starting benchmarks to confirm vLLM is ready
- Creates `.servers_override_<JOBID>.yaml` with the actual node IP; clean up manually after the job ends

**Data creation scripts:** `data/create_geo_160.py` and `data/create_diseases_160.py` regenerate the dataset CSVs from external sources (GeoNames API, etc.) if the raw data needs to be rebuilt.
