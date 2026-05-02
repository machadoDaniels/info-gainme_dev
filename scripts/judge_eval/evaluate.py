"""Judge-evaluate Oracle or Pruner answers against a larger model.

Usage:
    # Debug one conversation
    python3 scripts/judge_eval/evaluate.py --target oracle \\
      --conversation outputs/models/.../conversations/disease-.../

    # One experiment
    python3 scripts/judge_eval/evaluate.py --target pruner \\
      --runs outputs/models/.../runs.csv --workers 8 --turn-workers 4

    # All experiments
    python3 scripts/judge_eval/evaluate.py --target oracle --all

Writes ``oracle_judge_eval.json`` / ``pruner_judge_eval.json`` next to the
original agent files. Idempotent — conversations already judged by the same
``--judge-model`` are skipped.

Endpoint resolution priority: --base-url > env VLLM_<MODEL> > servers.yaml.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.judge_evaluation import Kind, build_judge_adapter, run_eval  # noqa: E402

load_dotenv()
logger = logging.getLogger("judge_eval")

OUTPUTS_BASE = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _conversations_from_runs_csv(
    runs_csv: Path,
    run_index: Optional[int] = None,
    sample_indices: Optional[list[int]] = None,
) -> list[Path]:
    """List conversations from a ``runs.csv``.

    Filters (applied in this order):
      - ``run_index`` — keep only rows whose ``run_index`` column matches
        (e.g., ``1`` → only first-run-per-target).
      - ``sample_indices`` — pick rows at those 0-based positions within the
        filtered list (e.g., ``[10,20,...,90]`` → 9 targets per experiment).
    """
    out: list[Path] = []
    seen: set[Path] = set()
    with runs_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if run_index is not None:
        rows = [r for r in rows if str(r.get("run_index", "")).strip() == str(run_index)]
    if sample_indices is not None:
        rows = [rows[i] for i in sample_indices if 0 <= i < len(rows)]
    for row in rows:
        rel = row.get("conversation_path", "")
        if not rel:
            continue
        conv = OUTPUTS_BASE / rel
        if conv.exists() and conv not in seen:
            seen.add(conv)
            out.append(conv)
    return out


def _resolve_conversations(args: argparse.Namespace) -> list[tuple[str, list[Path]]]:
    if args.conversation:
        return [(args.conversation.name, [args.conversation])]
    sample = _parse_sample_indices(args.sample_indices)
    if args.runs:
        return [(args.runs.parent.name,
                 _conversations_from_runs_csv(args.runs, args.run_index, sample))]
    groups: list[tuple[str, list[Path]]] = []
    for runs_csv in sorted(OUTPUTS_BASE.rglob("runs.csv")):
        convs = _conversations_from_runs_csv(runs_csv, args.run_index, sample)
        if convs:
            groups.append((runs_csv.parent.name, convs))
    return groups


def _parse_sample_indices(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def _load_servers(override: Optional[Path]) -> dict[str, str]:
    servers: dict[str, str] = {}
    base = PROJECT_ROOT / "configs" / "servers.yaml"
    for path in (base, override):
        if path and path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            servers.update(data.get("servers", {}) or {})
    return servers


def _resolve_base_url(model: str, explicit: Optional[str], override: Optional[Path]) -> Optional[str]:
    if explicit:
        return explicit
    env_key = "VLLM_" + re.sub(r"[^A-Z0-9]", "_", model.upper())
    return os.environ.get(env_key) or _load_servers(override).get(model)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge-evaluate Oracle or Pruner answers against a larger model.",
    )
    parser.add_argument("--target", choices=["oracle", "pruner"], required=True,
                        help="Which agent to judge.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process every runs.csv under outputs/")
    group.add_argument("--runs", type=Path, help="A specific runs.csv file")
    group.add_argument("--conversation", type=Path, help="A single conversation dir (debug)")

    parser.add_argument("--run-index", type=int, default=None,
                        help="Keep only rows with this run_index (e.g., 1 = first run per target)")
    parser.add_argument("--sample-indices", type=str, default=None,
                        help="Comma-separated 0-based positions in runs.csv after --run-index filter "
                             "(e.g., '10,20,30,40,50,60,70,80,90' → 9 targets per experiment)")

    parser.add_argument("--judge-model", default="gpt-oss-120b")
    parser.add_argument("--base-url", default=None, help="Skip servers.yaml lookup")
    parser.add_argument("--servers-override", type=Path, default=None,
                        help=".servers_override_<JOBID>.yaml emitted by dgx/run_judge_eval.sh")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)

    parser.add_argument("--workers", type=int, default=8, help="Conversations in parallel")
    parser.add_argument("--turn-workers", type=int, default=4, help="Turns per conversation in parallel")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    return parser


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()

    base_url = _resolve_base_url(args.judge_model, args.base_url, args.servers_override)
    if not base_url:
        logger.error("No endpoint for judge model %r. Use --base-url, --servers-override, "
                     "or add it to configs/servers.yaml.", args.judge_model)
        sys.exit(2)

    adapter = build_judge_adapter(
        model=args.judge_model, base_url=base_url, api_key=args.api_key,
        temperature=args.temperature, timeout=args.timeout,
    )
    logger.info("Target: %s  |  Judge: %s @ %s", args.target, args.judge_model, base_url)

    kind: Kind = args.target
    groups = _resolve_conversations(args)
    total = sum(len(convs) for _, convs in groups)
    logger.info("%d conversations across %d experiment(s)", total, len(groups))

    ok = skip = fail = 0
    lock = threading.Lock()

    def _one(conv: Path) -> tuple[str, Optional[str]]:
        try:
            _, was_skipped = run_eval(
                kind, conv, adapter, args.turn_workers,
                overwrite=args.force,
            )
            return ("skip" if was_skipped else "ok"), None
        except Exception as exc:
            return "fail", f"{type(exc).__name__}: {exc}"

    with logging_redirect_tqdm():
        for label, convs in groups:
            if not convs:
                continue
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_one, c): c for c in convs}
                bar = tqdm(as_completed(futures), total=len(futures),
                           desc=f"[{label}]", unit="conv",
                           dynamic_ncols=True, file=sys.stderr)
                for fut in bar:
                    status, err = fut.result()
                    with lock:
                        if status == "ok":
                            ok += 1
                        elif status == "skip":
                            skip += 1
                        else:
                            fail += 1
                            logger.warning("  ✗ %s — %s", futures[fut].name, err)
                    bar.set_postfix(ok=ok, skip=skip, fail=fail, refresh=True)

    logger.info("Done. ok=%d  skip=%d  fail=%d", ok, skip, fail)


if __name__ == "__main__":
    main()
