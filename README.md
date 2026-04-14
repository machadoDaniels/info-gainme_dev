# Info Gainme

Benchmark para medir ganho de informação em conversas com modelos de linguagem, usando uma arquitetura de três agentes:

- **Seeker** — faz perguntas de sim/não para identificar um alvo secreto
- **Oracle** — conhece o alvo e responde com veracidade
- **Pruner** — elimina candidatos do pool com base em cada par pergunta/resposta

A métrica principal é o ganho de informação por turno: `IG = H_antes - H_depois`, onde `H = log2(N)` (entropia de Shannon com distribuição uniforme).

---

## Baseline humano (você como Seeker)

Jogue como Seeker via CLI — Oracle e Pruner continuam sendo LLMs (Qwen3-8B).

```bash
# Escolhe alvos aleatoriamente (padrão: 1 jogo)
python3 human_benchmark_runner.py --config configs/human/geo_160_human_fo.yaml

# 5 jogos com seed fixo
python3 human_benchmark_runner.py --config configs/human/geo_160_human_fo.yaml --num-games 5 --seed 42

# Todos os alvos do dataset
python3 human_benchmark_runner.py --config configs/human/geo_160_human_fo.yaml --num-games 0
```

Configs disponíveis em `configs/human/`: `geo`, `objects` e `diseases` × `fo`/`po`. Os resultados são salvos em `outputs/` no mesmo formato dos benchmarks automatizados. Ctrl+C termina após o jogo atual.

---

## Executando benchmarks

### Via SLURM com vLLM - DGX

Sobe os servidores vLLM e roda os benchmarks em um único job SLURM:

```bash
# Seeker local — sobe todos os modelos necessários
sbatch dgx/run_full_benchmark.sh configs/full/4b/          # pasta inteira
sbatch dgx/run_full_benchmark.sh configs/full/4b/cot/geo_160_4b_thinking_fo_cot.yaml  # config individual

# Seeker externo (endpoint já em configs/servers.yaml) — sobe só oracle/pruner
sbatch dgx/run_external_seeker_benchmark.sh configs/full/235b/no_cot/
sbatch dgx/run_external_seeker_benchmark.sh configs/full/llama-70b/no_cot/
```

Monitore com `watch squeue -u $USER` e `tail -f logs/info-gainme-full-<JOBID>.out`.

Para adicionar um novo modelo:
1. Adicione o endpoint em `configs/servers.yaml`
2. Crie os configs em `configs/full/<modelo>/no_cot/` (e `cot/` se o modelo suportar thinking)
3. Submeta com `sbatch dgx/run_external_seeker_benchmark.sh configs/full/<modelo>/no_cot/`


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
