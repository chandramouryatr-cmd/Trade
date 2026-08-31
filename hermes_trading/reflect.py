"""Reflection cycle — revise EXACTLY ONE strategy variable from recent outcomes.

Modes:
  --fallback   deterministic rule (used before Hermes is installed)
  --hermes     call the `hermes` binary with a formatted prompt, apply its hypothesis

Either way: bump strategy.yaml `version`, copy the prior version to
state/history/v{NNNN}.yaml, append the hypothesis to state/hypotheses.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .score import _max_drawdown, _realised_return, score

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
STRATEGY = STATE / "strategy.yaml"
GOAL = STATE / "goal.yaml"
TRADES = STATE / "trades.jsonl"
HYPOTHESES = STATE / "hypotheses.jsonl"
HISTORY = STATE / "history"
MARKER = STATE / ".reflect_marker.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trade_count() -> int:
    if not TRADES.exists():
        return 0
    return sum(1 for l in TRADES.read_text(encoding="utf-8-sig").splitlines() if l.strip())


def _marker_count() -> int:
    try:
        return int(
            json.loads(MARKER.read_text(encoding="utf-8-sig")).get("trades_at_last_reflection", 0)
        )
    except (FileNotFoundError, ValueError):
        return 0


def _marker_write(n: int) -> None:
    MARKER.write_text(
        json.dumps({"trades_at_last_reflection": n, "ts": _now()}, indent=2), encoding="utf-8"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    # utf-8-sig transparently strips a BOM if some editor/shell wrote one
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _read_trades(limit: int) -> list[dict[str, Any]]:
    if not TRADES.exists():
        return []
    lines = [l for l in TRADES.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return [json.loads(l) for l in lines][-limit:]


def _bump_version(v: Any) -> str:
    try:
        return f"{int(v) + 1:02d}"
    except (TypeError, ValueError):
        return "02"


def _save_prior(prior: dict[str, Any]) -> Path:
    HISTORY.mkdir(parents=True, exist_ok=True)
    raw = str(prior.get("version", "01"))
    n = int(raw) if raw.isdigit() else 1
    dest = HISTORY / f"v{n:04d}.yaml"
    dest.write_text(yaml.safe_dump(prior, sort_keys=False), encoding="utf-8")
    return dest


def _apply(strat: dict[str, Any], hypo: dict[str, Any]) -> None:
    """Apply a single dotted-path variable change to `strat` in place."""
    path = str(hypo["variable"]).split(".")
    node = strat
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = hypo["new_value"]


def _commit(strat: dict[str, Any], hypo: dict[str, Any], mode: str) -> None:
    prior = _load_yaml(STRATEGY)
    saved = _save_prior(prior)
    strat["version"] = _bump_version(prior.get("version", "01"))
    STRATEGY.write_text(yaml.safe_dump(strat, sort_keys=False), encoding="utf-8")

    record = {
        "ts": _now(),
        "mode": mode,
        "from_version": prior.get("version", "01"),
        "to_version": strat["version"],
        "variable": hypo["variable"],
        "old_value": hypo.get("old_value"),
        "new_value": hypo["new_value"],
        "rationale": hypo.get("rationale", ""),
        "predicted_score_direction": hypo.get("predicted_score_direction", "up"),
        "prior_saved_to": str(saved.relative_to(ROOT)),
    }
    with HYPOTHESES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(
        f"strategy {prior.get('version', '01')} -> {strat['version']}  "
        f"changed {hypo['variable']}: {hypo.get('old_value')} -> {hypo['new_value']}",
        flush=True,
    )


def _fallback(strat: dict, goal: dict, trades: list[dict]) -> dict[str, Any] | None:
    if not trades:
        print("no closed trades yet — nothing to reflect on", flush=True)
        return None

    ret = _realised_return(trades)
    dd = _max_drawdown(trades)
    target = float(goal.get("target_return_30d", 0.05))
    max_dd = float(goal.get("max_drawdown", 0.08))
    entry = strat.get("entry", {})

    if dd > max_dd:
        old = float(strat.get("stop_loss_pct", 2.0))
        return {
            "variable": "stop_loss_pct",
            "old_value": old,
            "new_value": round(old - 0.2, 3),
            "rationale": f"drawdown {dd:.2%} over max {max_dd:.2%} — tighten the stop",
            "predicted_score_direction": "up",
        }
    if ret < target:
        old = float(entry.get("threshold", 30))
        return {
            "variable": "entry.threshold",
            "old_value": old,
            "new_value": round(old + 2, 3),
            "rationale": f"realised {ret:.2%} under target {target:.2%} — loosen entry",
            "predicted_score_direction": "up",
        }

    print(f"on track (ret={ret:.2%}, dd={dd:.2%}) — no change this cycle", flush=True)
    return None


_HERMES_PROMPT = """You are revising a paper-trading strategy. Change EXACTLY ONE variable.

Current strategy.yaml:
{strategy}
Goal (goal.yaml):
{goal}
Last {n} closed trades (JSON lines):
{trades}

Current composite score against the goal: {score:+.3f}  (range -1..+1)

Reply with ONLY a JSON object, no prose:
{{"variable": "<dotted path in strategy>", "old_value": <v>, "new_value": <v>,
  "rationale": "<one sentence>", "predicted_score_direction": "up"}}
"""


def _hermes(strat: dict, goal: dict, trades: list[dict]) -> dict[str, Any] | None:
    binary = shutil.which("hermes")
    if not binary:
        print("hermes binary not on PATH — use --fallback instead", flush=True)
        return None

    prompt = _HERMES_PROMPT.format(
        strategy=yaml.safe_dump(strat, sort_keys=False),
        goal=yaml.safe_dump(goal, sort_keys=False),
        n=len(trades),
        trades="\n".join(json.dumps(t, separators=(",", ":")) for t in trades) or "(none)",
        score=score(trades, goal),
    )
    proc = subprocess.run(
        [binary, "run", "--quiet"],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0:
        print(f"hermes exited {proc.returncode}: {proc.stderr.strip()}", flush=True)
        return None

    out = proc.stdout.strip()
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end == -1:
        print(f"no JSON object in hermes output:\n{out}", flush=True)
        return None
    try:
        hypo = json.loads(out[start : end + 1])
    except json.JSONDecodeError as exc:
        print(f"bad JSON from hermes: {exc}", flush=True)
        return None
    if "variable" not in hypo or "new_value" not in hypo:
        print(f"hermes hypothesis missing keys: {hypo}", flush=True)
        return None
    return hypo


def main() -> None:
    parser = argparse.ArgumentParser(prog="hermes_trading.reflect")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--fallback", action="store_true", help="deterministic single-variable rule")
    grp.add_argument("--hermes", action="store_true", help="ask the hermes binary")
    parser.add_argument("--trades", type=int, default=25, help="recent trades to consider")
    parser.add_argument(
        "--if-due",
        action="store_true",
        help="only reflect when >= goal.reflection_every trades have closed since last time",
    )
    args = parser.parse_args()

    strat = _load_yaml(STRATEGY)
    goal = _load_yaml(GOAL)

    if args.if_due:
        every = int(goal.get("reflection_every", 5))
        total, last = _trade_count(), _marker_count()
        if total - last < every:
            print(f"not due: {total - last}/{every} new closed trades since last reflection", flush=True)
            sys.exit(0)
        print(f"due: {total - last} new closed trades (>= {every})", flush=True)

    trades = _read_trades(args.trades)
    hypo = _fallback(strat, goal, trades) if args.fallback else _hermes(strat, goal, trades)
    if hypo is None:
        if args.if_due:
            _marker_write(_trade_count())  # avoid re-triggering every run on a no-op
        sys.exit(0)

    hypo.setdefault("old_value", None)
    _apply(strat, hypo)
    _commit(strat, hypo, "fallback" if args.fallback else "hermes")
    if args.if_due:
        _marker_write(_trade_count())


if __name__ == "__main__":
    main()
