# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Info Gainme is a benchmark that measures **information gain** in LLM conversations using a three-agent architecture:

- **Seeker** — asks yes/no questions to identify a secret target (city, object, or disease)
- **Oracle** — knows the target and answers truthfully
- **Pruner** — eliminates candidates from the pool based on each Q&A pair

Information gain is Shannon entropy reduction: `H = log2(N)` where N is the number of active candidates. The benchmark records win rate, total IG, avg IG/turn, and compliance rate.

## Running benchmarks

**Single config (directly, without SLURM):**
```bash
python3 benchmark_runner.py --config configs/full/8b/geo_160_8b_fo_cot.yaml
```

**Multiple configs via screen (preferred on DGX):**
```bash
screen -dmS benchmarks bash -c 'bash dgx/run_benchmarks_screen.sh configs/full/8b/ 2>&1 | tee logs/screen-8b-all.out; exec bash'
screen -r benchmarks
```

**Via SLURM (may sit in queue due to Priority):**
```bash
bash dgx/run_benchmarks_slurm.sh configs/full/8b/          # whole folder
bash dgx/run_benchmarks_slurm.sh configs/full/8b/geo.yaml  # single config
```

Benchmarks are **resumable** — completed `(target_id, run_index)` pairs are detected from `runs.csv` and skipped automatically.

## Analysis pipeline

Run in order after benchmarks finish:

```bash
# 1. Compute metrics per experiment + unified CSV
bash dgx/run_analyze_results.sh
# or for a specific CSV:
bash dgx/run_analyze_results.sh outputs/models/.../runs.csv

# 2. Synthesize reasoning traces (CoT experiments only)
bash dgx/run_synthesize_traces.sh
# With a local model instead of gpt-4o-mini:
MODEL=Qwen3-8B BASE_URL=http://localhost:8020/v1 bash dgx/run_synthesize_traces.sh

# 3. Analyze reasoning traces
bash dgx/run_analyze_traces.sh
```

`dgx/run_analysis.sh` is a shortcut for step 1 only (runs `analyze_results.py --all` + `generate_unified_csv.py` inside Singularity).

## Configuration

**Experiment configs** live in `configs/full/<model>/`. Each YAML specifies models for all three agents, observability mode (FO/PO), dataset, and experiment name.

**Server endpoints** are centralized in `configs/servers.yaml` — model name → base URL. `config_loader.py` walks up from any config file to find this file automatically. **Never hardcode URLs in individual configs.**

**CoT vs no-CoT** is controlled by `extra_body.chat_template_kwargs.enable_thinking: true` + `use_reasoning: true` on the seeker. No-CoT configs omit both fields entirely.

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
```

**Key flow:** `benchmark_runner.py` → `BenchmarkRunner.run()` → per game: `Orchestrator.from_target()` → loop: Seeker asks → Oracle answers → Pruner prunes → entropy computed → `TurnState` appended → results written to `runs.csv`.

**Parallelism:** `BenchmarkRunner` runs games concurrently via `ThreadPoolExecutor(max_workers=N)`. CSV writes are serialized with a `threading.Lock`. Each thread gets a `copy.deepcopy(pool)` to avoid shared state.

**Geo vs flat domains:** The geo domain uses `KnowledgeGraph` (hierarchical tree: region→subregion→country→state→city). When all cities under a parent are pruned, the parent is also pruned recursively (`apply_pruning` walks up via `has_child`/`contains` edges). Objects and diseases use the flat `CandidatePool` instead.

**Question-choice evaluation** (post-hoc, CoT only): `scripts/evaluate_all_seeker_choices.py` reads a `runs.csv`, finds conversations with `seeker_traces.json`, then for each turn re-runs Oracle+Pruner on every question the Seeker considered to compute counterfactual info gains. Results saved as `question_evaluation.json` per conversation and `question_evaluations_summary.json` per experiment. This pipeline is read-only — it never modifies turns or conversation files.

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

- Models served via **vLLM** with OpenAI-compatible API
- Runs inside **Singularity** container (`vllm_openai_latest.sif`) on DGX H100 nodes
- Shared group `sd22` — scripts use `sg sd22 -c "..."` for file permission inheritance
- `configs/servers.yaml` maps model names to current vLLM endpoint URLs (update here when IPs/ports change)
- `OPENAI_API_KEY` must be set in `.env` (required even for local vLLM endpoints)
