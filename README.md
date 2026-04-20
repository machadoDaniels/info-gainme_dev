# Info Gainme

Benchmark para medir ganho de informação em conversas com modelos de linguagem, usando uma arquitetura de três agentes:

- **Seeker** — faz perguntas de sim/não para identificar um alvo secreto
- **Oracle** — conhece o alvo e responde com veracidade
- **Pruner** — elimina candidatos do pool com base em cada par pergunta/resposta

A métrica principal é o ganho de informação por turno: `IG = H_antes - H_depois`, onde `H = log2(N)` (entropia de Shannon com distribuição uniforme).

---

## Baseline humano (você como Seeker)

Jogue como Seeker via CLI — Oracle e Pruner continuam sendo LLMs (Qwen3-8B).

Dois modos de observabilidade disponíveis:

- **FO (Fully Observable)** — você vê a lista completa de candidatos restantes a cada turno
- **PO (Partially Observable)** — você vê apenas o histórico de perguntas e respostas

```bash
# Fully Observable (você vê os candidatos restantes)
python3 human_benchmark_runner.py --config configs/human/geo_160_human_fo.yaml

# Partially Observable (você vê só o histórico de Q&A)
python3 human_benchmark_runner.py --config configs/human/geo_160_human_po.yaml

# Apenas 5 jogos
python3 human_benchmark_runner.py --config configs/human/geo_160_human_fo.yaml --num-games 5
```

Configs disponíveis em `configs/human/`: `geo`, `objects` e `diseases` × `fo`/`po`. Os resultados são salvos em `outputs/` no mesmo formato dos benchmarks automatizados. Ctrl+C termina após o jogo atual.

---

## Executando benchmarks

### Via SLURM com vLLM - DGX

Sobe os servidores vLLM e roda os benchmarks em um único job SLURM. Precisa escolher partição + modelos via `--export`:

```bash
# H100 partition (default)
sbatch --partition=h100n2 --gres=gpu:2 \
  --export=ALL,MODEL1=Qwen/Qwen3-4B-Thinking-2507,MODEL1_NAME=Qwen3-4B-Thinking-2507,MODEL2=Qwen/Qwen3-8B,MODEL2_NAME=Qwen3-8B,MODE=dual,CONFIGS_TARGET=configs/full/4b/cot/ \
  dgx/run_full_benchmark.sh

# B200 partition (Blackwell — usa CUDA graphs automaticamente, mais rápido)
sbatch --partition=b200n1 --gres=gpu:2 \
  --export=ALL,MODEL1=allenai/Olmo-3-7B-Think,MODEL1_NAME=Olmo-3-7B-Think,MODEL2=Qwen/Qwen3-8B,MODEL2_NAME=Qwen3-8B,MODE=dual,CONFIGS_TARGET=configs/full/olmo3-7b/cot/ \
  dgx/run_full_benchmark.sh

# Seeker externo (endpoint já em configs/servers.yaml) — sobe só oracle/pruner
sbatch dgx/run_external_seeker_benchmark.sh configs/full/235b/no_cot/
sbatch dgx/run_external_seeker_benchmark.sh configs/full/llama-70b/no_cot/
```

**Partições disponíveis:** `h100n2` (H100), `h100n3` (H100), `b200n1` (Blackwell 180GB). Cada node tem seu próprio `/raid` — **não compartilham filesystem** (sincronize configs/outputs via `rsync` quando mover entre nodes).

**Env vars úteis** no `--export`:
- `VLLM_MAX_NUM_SEQS=32` — requests vLLM paralelos (default 32)
- `VLLM_ENGINE_READY_TIMEOUT_S=3600` — timeout de startup/download (aumente para modelos grandes)
- `VLLM_ENFORCE_EAGER=true|false` — override da heurística por partição (auto-desativado em B200)
- `RUNS_PER_TARGET=1` — passa `--runs-per-target` para o benchmark (default 3)

Monitore com `watch squeue -u $USER` e `tail -f logs/info-gainme-full-<JOBID>.out`. Logs do vLLM ficam em `logs/info-gainme-full-<JOBID>-vllm-<MODEL_NAME>.log`.

### Submetendo jobs como outro usuário (alias `asjulia`)

Em fluxos multi-usuário (ex: rodar em `h100n3` como `user_juliadollis`), use o alias `asjulia` definido no `.bashrc`:

```bash
bash -ic 'asjulia "cd /raid/user_danielpedrozo/projects/info-gainme_dev; \
  sbatch --partition=h100n3 --gres=gpu:2 \
    --export=ALL,MODEL1=...,MODEL1_NAME=...,MODEL2=...,MODEL2_NAME=...,MODE=dual,CONFIGS_TARGET=... \
    dgx/run_full_benchmark.sh"'
```

Use `;` entre comandos dentro das aspas (o shell externo interpretaria `&&`). Consulte fila da julia com `bash -ic 'asjulia "squeue -u user_juliadollis"'` (ou alias `sqj`).

Para adicionar um novo modelo:
1. Adicione o endpoint em `configs/servers.yaml` (para seeker externo) ou passe `MODEL1=<hf_repo>` no `--export` (para seeker local)
2. Crie os configs em `configs/full/<modelo>/no_cot/` (e `cot/` se o modelo suportar thinking)
3. Submeta com `sbatch dgx/run_external_seeker_benchmark.sh configs/full/<modelo>/no_cot/` ou `dgx/run_full_benchmark.sh`


### Sem subir vLLM (servidores já rodando)

Use quando os endpoints já estão ativos e registrados em `configs/servers.yaml`:

```bash
# Via screen — abre sessão em background para cada config
bash dgx/run_benchmarks_slurm.sh configs/full/8b/
bash dgx/run_benchmarks_slurm.sh configs/full/8b/geo_160_8b_fo_cot.yaml  # config individual

# Direto no terminal
python3 benchmark_runner.py --config configs/full/8b/geo_160_8b_fo_cot.yaml
```

Benchmarks são **resumíveis** — runs já completos são detectados no `runs.csv` e pulados automaticamente.

---

## Configuração

Os configs de experimento ficam em `configs/full/<modelo>/`. Cada YAML define os modelos dos três agentes, modo de observabilidade (`FULLY_OBSERVABLE` / `PARTIALLY_OBSERVABLE`), dataset e nome do experimento.

Os endpoints dos servidores vLLM ficam centralizados em **`configs/servers.yaml`** (nome do modelo → URL). Nunca edite URLs diretamente nos configs individuais.

CoT é habilitado no seeker com:
```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
use_reasoning: true
```

---

## Pipeline de análise

```
Benchmark
  └── runs.csv + conversations/*/seeker.json
         │
         ├── dgx/run_analyze_results.sh   →  summary.json + variance.json + unified_experiments.csv
         │
         └── (só CoT) dgx/run_synthesize_traces.sh  →  seeker_traces.json por conversa
                              │
                              └── dgx/run_analyze_traces.sh  →  reasoning_traces_analysis.json
```

### `dgx/run_analyze_results.sh`

Calcula métricas agregadas por experimento (win rate, IG médio, turnos, compliance) e gera o CSV unificado.

```bash
./dgx/run_analyze_results.sh                          # todos os experimentos
./dgx/run_analyze_results.sh outputs/models/.../runs.csv  # um CSV específico
```

**Saída:** `summary.json`, `variance.json` por experimento + `outputs/unified_experiments.csv`

---

### `dgx/run_synthesize_traces.sh`

Para cada conversa CoT, usa um LLM para extrair do `seeker.json` as opções consideradas, o raciocínio e a escolha final por turno. Idempotente — pula se `seeker_traces.json` já existir.

```bash
./dgx/run_synthesize_traces.sh                             # todos
./dgx/run_synthesize_traces.sh outputs/models/.../runs.csv # um CSV específico
MODEL=Qwen3-8B BASE_URL=http://10.100.0.112:8020/v1 ./dgx/run_synthesize_traces.sh  # modelo local
```

**Saída:** `conversations/<alvo>/seeker_traces.json`

---

### `dgx/run_all_synthesize_traces.sh`

Atalho para sintetizar todos os traces de uma vez, sem `sg sd22` (ambiente pessoal).

```bash
./dgx/run_all_synthesize_traces.sh
```

---

### `dgx/run_analyze_traces.sh`

Lê todos os `seeker_traces.json` e produz análise agregada do raciocínio do Seeker: perguntas mais consideradas, padrões de decisão, distribuição de turnos. Só processa experimentos CoT.

```bash
./dgx/run_analyze_traces.sh
./dgx/run_analyze_traces.sh /caminho/para/outputs  # diretório customizado
```

**Saída:** `outputs/reasoning_traces_analysis.json`

---

## Infraestrutura vLLM

### `dgx/run_vllm_multimodel.sh`

Sobe dois servidores vLLM na mesma máquina, compartilhando VRAM. As portas e modelos são definidos no script e refletidos em `configs/servers.yaml`.

Para verificar status e métricas dos servidores:
```bash
curl http://<ip>:<porta>/v1/models
curl http://<ip>:<porta>/metrics
```

---

## Estrutura de outputs

```
outputs/
├── unified_experiments.csv              ← tabela consolidada de todos os experimentos
├── reasoning_traces_analysis.json       ← análise agregada de reasoning traces (CoT)
└── models/
    └── s_<seeker>__o_<oracle>__p_<pruner>/
        └── <experimento>/
            ├── runs.csv                 ← resultados brutos (um row por run)
            ├── summary.json            ← métricas globais e por alvo
            ├── variance.json           ← variância do IG por alvo
            └── conversations/
                └── <alvo>_run<N>/
                    ├── seeker.json     ← histórico completo do Seeker
                    ├── oracle.json     ← histórico completo do Oracle
                    ├── pruner.json     ← histórico do Pruner
                    ├── turns.jsonl     ← detalhes turno a turno
                    ├── metadata.json   ← resultado, config e pool stats do jogo
                    └── seeker_traces.json  ← reasoning traces sintetizados (CoT)
```
