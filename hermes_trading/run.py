"""Entrypoint.

Resolves the asset from state/goal.yaml (override with --asset) and starts the loop.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from .loop import run_loop

ROOT = Path(__file__).resolve().parent.parent
GOAL = ROOT / "state" / "goal.yaml"


def _asset_from_goal() -> str:
    try:
        data = yaml.safe_load(GOAL.read_text(encoding="utf-8-sig")) or {}
        return str(data.get("asset", "BTC/USDT"))
    except FileNotFoundError:
        return "BTC/USDT"


def main() -> None:
    parser = argparse.ArgumentParser(prog="hermes-trading")
    parser.add_argument("--asset", default=None, help="ccxt ticker, e.g. BTC/USDT")
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="run for N minutes then exit cleanly (for scheduled hosts). Default: forever",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single tick and exit (implies --minutes is ignored)",
    )
    args = parser.parse_args()
    asset = args.asset or _asset_from_goal()
    try:
        asyncio.run(
            run_loop(
                asset,
                max_minutes=None if args.once else args.minutes,
                max_ticks=1 if args.once else None,
            )
        )
    except KeyboardInterrupt:
        print("shutting down", flush=True)


if __name__ == "__main__":
    main()
