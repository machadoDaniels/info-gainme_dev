#!/usr/bin/env python3
"""Gera seeker_traces.json para todas as conversas de um experimento.

Lê os seeker.json de cada conversa, sintetiza os reasoning traces via LLM
e salva seeker_traces.json no mesmo diretório.

Usage:
    # Processa todos os runs.csv sob outputs/
    python multi_synthesize_reasoning_traces.py --all

    # Processa um runs.csv específico
    python multi_synthesize_reasoning_traces.py --runs outputs/models/.../runs.csv

    # Opções de modelo/URL
    python multi_synthesize_reasoning_traces.py --all --model Qwen3-8B --base-url http://10.100.0.121:8020/v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.reasoning_synthesis import create_seeker_traces_file
from src.agents.llm_config import LLMConfig
from src.utils import ClaryLogger

load_dotenv()
logger = ClaryLogger.get_logger(__name__)

OUTPUTS_BASE = project_root / "outputs"  # project_root já é a raiz do projeto


def find_runs_csvs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("runs.csv"))


def conversation_dirs_from_csv(runs_csv: Path) -> list[Path]:
    dirs = []
    with runs_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("conversation_path", "")
            if rel:
                full = OUTPUTS_BASE / rel
                if full.exists() and full not in dirs:
                    dirs.append(full)
    return dirs


def process_conversation(conv_dir: Path, llm_config: LLMConfig) -> tuple[Path, bool, str]:
    seeker_json = conv_dir / "seeker.json"
    traces_json = conv_dir / "seeker_traces.json"

    if not seeker_json.exists():
        return conv_dir, False, "seeker.json não encontrado"

    if traces_json.exists():
        return conv_dir, True, "já existe (skip)"

    try:
        create_seeker_traces_file(seeker_json, traces_json, llm_config)
        return conv_dir, True, "ok"
    except Exception as e:
        return conv_dir, False, str(e)


def process_runs_csv(runs_csv: Path, llm_config: LLMConfig, max_workers: int) -> None:
    conv_dirs = conversation_dirs_from_csv(runs_csv)
    if not conv_dirs:
        logger.warning("Nenhuma conversa encontrada em %s", runs_csv)
        return

    experiment = runs_csv.parent.name
    logger.info("📂 %s — %d conversas", experiment, len(conv_dirs))

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_conversation, d, llm_config): d for d in conv_dirs}
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Sintetizando [{experiment}]",
            unit="conv",
            leave=True,
        )
        for future in bar:
            _, success, msg = future.result()
            if not success:
                fail += 1
                logger.warning("  ❌ %s — %s", futures[future].name, msg)
            elif msg.startswith("já existe"):
                skip += 1
            else:
                ok += 1
            bar.set_postfix(ok=ok, skip=skip, fail=fail, refresh=True)

    logger.info("  ✅ %d gerados | ⏭️  %d skip | ❌ %d erros", ok, skip, fail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera seeker_traces.json para todos os experimentos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Processa todos os runs.csv sob outputs/")
    group.add_argument("--runs", type=Path, help="Caminho para um runs.csv específico")
    parser.add_argument("--model", default="gpt-4o-mini", help="Modelo LLM para síntese")
    parser.add_argument("--base-url", default=None, help="Base URL do servidor LLM (padrão: OpenAI)")
    parser.add_argument("--workers", type=int, default=8, help="Threads paralelas por experimento")
    args = parser.parse_args()

    import os
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    llm_config = LLMConfig(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        timeout=120.0,
    )

    if args.all:
        runs_csvs = find_runs_csvs(OUTPUTS_BASE)
        logger.info("🔎 %d runs.csv encontrados", len(runs_csvs))
        for runs_csv in tqdm(runs_csvs, desc="Experimentos (runs.csv)", unit="exp"):
            process_runs_csv(runs_csv, llm_config, args.workers)
    else:
        process_runs_csv(args.runs, llm_config, args.workers)

    logger.info("✅ Concluído.")


if __name__ == "__main__":
    main()
