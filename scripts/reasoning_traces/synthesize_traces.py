#!/usr/bin/env python3
"""Synthesize reasoning traces for all CoT conversations.

Reads seeker.json from each conversation, synthesises reasoning traces via LLM
and appends one JSON record per conversation to a single JSONL file.

Usage:
    # Full sweep (all CoT runs.csv under outputs/)
    python3 scripts/reasoning_traces/synthesize_traces.py --all

    # Single runs.csv
    python3 scripts/reasoning_traces/synthesize_traces.py --runs outputs/models/.../runs.csv

    # Single conversation (debug)
    python3 scripts/reasoning_traces/synthesize_traces.py --seeker-file outputs/models/.../seeker.json

Output:
    outputs/seeker_traces.jsonl   (one conversation per line, appended on resume)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.reasoning_synthesis import synthesize_conversation
from src.agents.llm_config import LLMConfig
from src.utils import ClaryLogger

load_dotenv()
logger = ClaryLogger.get_logger(__name__)

OUTPUTS_BASE = project_root / "outputs"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_runs_csvs(base_dir: Path) -> list[Path]:
    return sorted(
        p for p in base_dir.rglob("runs.csv")
        if (p.parent.name.endswith("_cot") or "_cot_" in p.parent.name)
        and "no_cot" not in p.parent.name
    )


def seeker_paths_from_csv(
    runs_csv: Path,
    run_index: int | None = None,
    sample_indices: list[int] | None = None,
) -> list[Path]:
    """Same row-filter conventions as ``scripts/judge_eval``.

    Filters applied in this order:
      - ``run_index``      keep only rows with this ``run_index`` column
      - ``sample_indices`` deterministic 0-based positions in the filtered list

    Both ``None`` → full sweep.
    """
    with runs_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if run_index is not None:
        rows = [r for r in rows if str(r.get("run_index", "")).strip() == str(run_index)]
    if sample_indices is not None:
        rows = [rows[i] for i in sample_indices if 0 <= i < len(rows)]
    paths: list[Path] = []
    for row in rows:
        rel = row.get("conversation_path", "")
        if rel:
            seeker = OUTPUTS_BASE / rel / "seeker.json"
            if seeker.exists() and seeker not in paths:
                paths.append(seeker)
    return paths


def _parse_sample_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_one(
    seeker_path: Path,
    llm_config: LLMConfig,
    turn_workers: int,
    out_jsonl: Path,
    lock: threading.Lock,
    done_paths: set[str],
) -> tuple[bool, str]:
    """Synthesize one conversation and append to JSONL. Returns (success, msg)."""
    key = str(seeker_path)
    if key in done_paths:
        return True, "skip"

    try:
        data = synthesize_conversation(seeker_path, llm_config, turn_workers=turn_workers)
    except Exception as e:
        return False, str(e)

    data["analysis_model"] = llm_config.model

    with lock:
        with out_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        done_paths.add(key)

    return True, "ok"


def process_batch(
    seeker_paths: list[Path],
    llm_config: LLMConfig,
    max_workers: int,
    turn_workers: int,
    out_jsonl: Path,
    lock: threading.Lock,
    done_paths: set[str],
    desc: str,
    tqdm_position: int = 0,
    tqdm_leave: bool = True,
) -> None:
    todo = [p for p in seeker_paths if str(p) not in done_paths]
    if not todo:
        logger.info("  ⏭️  %s — tudo já sintetizado", desc)
        return

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, p, llm_config, turn_workers, out_jsonl, lock, done_paths): p
            for p in todo
        }
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"[{desc}]",
            unit="conv",
            position=tqdm_position,
            leave=tqdm_leave,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        for future in bar:
            success, msg = future.result()
            if msg == "skip":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
                logger.warning("  ❌ %s — %s", futures[future].name, msg)
            bar.set_postfix(ok=ok, skip=skip, fail=fail, refresh=True)

    logger.info("  ✅ %d gerados | ⏭️  %d skip | ❌ %d erros", ok, skip, fail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Processa todos os runs.csv CoT sob outputs/")
    group.add_argument("--runs", type=Path, help="Caminho para um runs.csv específico")
    group.add_argument("--seeker-file", type=Path, help="Classifica uma única conversa (debug)")
    parser.add_argument("--model", default="nvidia/Kimi-K2.5-NVFP4")
    parser.add_argument("--base-url", default="http://200.137.197.131:60002/v1")
    parser.add_argument("--api-key", default="NINGUEM-TA-PURO-2K26")
    parser.add_argument("--workers", type=int, default=8, help="Conversas paralelas por experimento")
    parser.add_argument("--turn-workers", type=int, default=4, help="Chamadas LLM paralelas por conversa")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=OUTPUTS_BASE / "seeker_traces.jsonl",
        help="JSONL de saída (uma conversa por linha).",
    )
    parser.add_argument("--force", action="store_true", help="Re-sintetiza mesmo que já esteja no JSONL")
    parser.add_argument("--run-index", type=int, default=None,
                        help="Mantém só linhas com esse run_index na runs.csv (ex: 1 = só run01). "
                             "Mesma convenção do scripts/judge_eval.")
    parser.add_argument("--sample-indices", type=str, default=None,
                        help="Posições 0-based separadas por vírgula dentro da runs.csv após "
                             "--run-index (ex: '0,10,20,...,150'). Determinístico.")
    args = parser.parse_args()
    sample_indices = _parse_sample_indices(args.sample_indices)

    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=120.0,
    )
    logger.info("🤖 Modelo: %s @ %s", args.model, args.base_url)

    # Resumability: load already-done seeker_paths from existing JSONL.
    out_jsonl: Path = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out_jsonl.exists():
        out_jsonl.unlink()
    done_paths: set[str] = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done_paths.add(json.loads(line)["seeker_path"])
            except Exception:
                pass
    logger.info("📋 %d conversas já sintetizadas no JSONL", len(done_paths))

    lock = threading.Lock()

    with logging_redirect_tqdm():
        if args.seeker_file:
            success, msg = process_one(
                args.seeker_file, llm_config, args.turn_workers, out_jsonl, lock, done_paths
            )
            logger.info("%s — %s", args.seeker_file.name, msg if success else f"❌ {msg}")

        elif args.runs:
            paths = seeker_paths_from_csv(args.runs, args.run_index, sample_indices)
            logger.info("📂 %s — %d conversas", args.runs.parent.name, len(paths))
            process_batch(paths, llm_config, args.workers, args.turn_workers,
                          out_jsonl, lock, done_paths, desc=args.runs.parent.name)

        else:  # --all
            runs_csvs = find_runs_csvs(OUTPUTS_BASE)
            logger.info("🔎 %d runs.csv CoT encontrados", len(runs_csvs))
            if args.run_index is not None:
                logger.info("🎯 run_index=%d", args.run_index)
            if sample_indices is not None:
                logger.info("🎯 sample_indices=%s", sample_indices)
            logger.info("⚙️  workers=%d  turn-workers=%d", args.workers, args.turn_workers)
            for i, runs_csv in enumerate(tqdm(
                runs_csvs, desc="Experimentos", unit="exp", position=0,
                leave=True, dynamic_ncols=True, file=sys.stderr,
            )):
                paths = seeker_paths_from_csv(runs_csv, args.run_index, sample_indices)
                process_batch(
                    paths, llm_config, args.workers, args.turn_workers,
                    out_jsonl, lock, done_paths,
                    desc=runs_csv.parent.name, tqdm_position=1, tqdm_leave=False,
                )

    total = sum(1 for _ in out_jsonl.open(encoding="utf-8") if _.strip())
    logger.info("✅ Concluído. %d conversas no JSONL: %s", total, out_jsonl)


if __name__ == "__main__":
    main()
