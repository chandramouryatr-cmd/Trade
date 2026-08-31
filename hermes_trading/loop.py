"""24/7 reliability loop.

Every minute:
  * pull price (required) + onchain/news/macro context (best-effort) via adapters
  * evaluate the strategy in state/strategy.yaml (RSI entry, stop-loss / RSI exit)
  * open or close a single paper position
  * append every closed trade to state/trades.jsonl
  * write state/heartbeat.json

Reliability:
  * per-adapter retries (3 attempts, exponential backoff)
  * circuit-break: 5 consecutive failed ticks halts the loop
  * a price SchemaError halts the loop immediately
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml

from .adapters import macro, news, onchain, price
from .adapters.price import SchemaError as PriceSchemaError

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
TRADES = STATE / "trades.jsonl"
HEARTBEAT = STATE / "heartbeat.json"
STRATEGY = STATE / "strategy.yaml"
GOAL = STATE / "goal.yaml"

POLL_SECONDS = 60
MAX_RETRIES = 3
CIRCUIT_BREAK_AFTER = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for a, b in zip(closes[-period - 1 : -1], closes[-period:]):
        diff = b - a
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


async def _with_retries(factory: Callable[[], Coroutine[Any, Any, Any]], name: str) -> Any:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 - adapters raise varied errors
            last_exc = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
    raise RuntimeError(f"adapter {name} failed after {MAX_RETRIES} tries: {last_exc}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_heartbeat(row: dict[str, Any]) -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps(row, indent=2), encoding="utf-8")


class Position:
    __slots__ = ("side", "entry", "size_r", "opened_at", "peak", "trough")

    def __init__(self, side: str, entry: float, size_r: float) -> None:
        self.side = side
        self.entry = entry
        self.size_r = size_r
        self.opened_at = _now()
        self.peak = entry
        self.trough = entry


async def run_loop(
    asset: str,
    max_minutes: float | None = None,
    max_ticks: int | None = None,
) -> None:
    """Run the trading loop.

    max_minutes / max_ticks bound the run for scheduled hosts (e.g. GitHub
    Actions) that can't hold a process open forever. Both None -> run forever.
    """
    consecutive_failures = 0
    position: Position | None = None
    closed = 0
    ticks = 0
    started_at = time.time()
    bound = []
    if max_minutes is not None:
        bound.append(f"{max_minutes:g} min")
    if max_ticks is not None:
        bound.append(f"{max_ticks} ticks")
    print(
        f"Booting hermes-trading worker  asset={asset}  mode=paper"
        + (f"  bound={' / '.join(bound)}" if bound else ""),
        flush=True,
    )

    while True:
        if max_minutes is not None and (time.time() - started_at) >= max_minutes * 60:
            print(f"{_now()}  reached {max_minutes:g}-minute bound — exiting cleanly", flush=True)
            return
        if max_ticks is not None and ticks >= max_ticks:
            print(f"{_now()}  reached {max_ticks}-tick bound — exiting cleanly", flush=True)
            return
        ticks += 1
        tick_started = time.time()
        try:
            strat = _load_yaml(STRATEGY)
            goal = _load_yaml(GOAL)

            px = await _with_retries(lambda: price.fetch(asset, "1m", 200), "price")
            if px.get("schema_version") != price.SCHEMA_VERSION:
                raise PriceSchemaError(
                    f"price schema {px.get('schema_version')} != {price.SCHEMA_VERSION}"
                )

            context: dict[str, Any] = {}
            for mod, key in ((onchain, "onchain"), (news, "news"), (macro, "macro")):
                try:
                    context[key] = await _with_retries(lambda m=mod: m.fetch(asset), key)
                except Exception as exc:  # noqa: BLE001 - context is best-effort
                    context[key] = {"error": str(exc)}

            closes = px["closes"]
            last = px["last"]

            entry = strat.get("entry", {})
            rsi = _rsi(closes, int(entry.get("rsi_period", 14)))
            threshold = float(entry.get("threshold", 30))
            direction = entry.get("direction", "long")
            exit_rsi = float(entry.get("exit_rsi", 70))
            stop_pct = float(strat.get("stop_loss_pct", 2.0))
            size_r = float(strat.get("position_size_r", 0.5))

            if position is None:
                fire = (direction == "long" and rsi <= threshold) or (
                    direction == "short" and rsi >= 100 - threshold
                )
                if fire:
                    position = Position(direction, last, size_r)
                    print(f"{_now()}  OPEN  {direction} @ {last:.2f}  rsi={rsi:.1f}", flush=True)
            else:
                position.peak = max(position.peak, last)
                position.trough = min(position.trough, last)
                if position.side == "long":
                    move = (last - position.entry) / position.entry
                    stop_hit = last <= position.entry * (1 - stop_pct / 100)
                    tp_hit = rsi >= exit_rsi
                else:
                    move = (position.entry - last) / position.entry
                    stop_hit = last >= position.entry * (1 + stop_pct / 100)
                    tp_hit = rsi <= 100 - exit_rsi

                if stop_hit or tp_hit:
                    ret_pct = move * 100 * position.size_r
                    row = {
                        "ts": _now(),
                        "asset": asset,
                        "side": position.side,
                        "entry": position.entry,
                        "exit": last,
                        "return_pct": round(ret_pct, 4),
                        "reason": "stop_loss" if stop_hit else "take_profit",
                        "rsi_exit": round(rsi, 2),
                        "strategy_version": strat.get("version", "??"),
                        "opened_at": position.opened_at,
                    }
                    _append_jsonl(TRADES, row)
                    closed += 1
                    print(
                        f"{_now()}  CLOSE {position.side} @ {last:.2f}  "
                        f"ret={ret_pct:+.2f}%  ({row['reason']})  total_closed={closed}",
                        flush=True,
                    )
                    position = None

            _write_heartbeat(
                {
                    "ts": _now(),
                    "asset": asset,
                    "last_price": last,
                    "rsi": round(rsi, 2),
                    "in_position": position is not None,
                    "closed_trades": closed,
                    "strategy_version": strat.get("version", "??"),
                    "consecutive_failures": consecutive_failures,
                }
            )
            consecutive_failures = 0

        except PriceSchemaError as exc:
            print(f"{_now()}  SCHEMA ERROR (price): {exc} — halting loop", flush=True)
            raise
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            print(
                f"{_now()}  tick failed ({consecutive_failures}/{CIRCUIT_BREAK_AFTER}): {exc}",
                flush=True,
            )
            if consecutive_failures >= CIRCUIT_BREAK_AFTER:
                print(f"{_now()}  circuit breaker tripped — halting loop", flush=True)
                raise

        if max_ticks is not None and ticks >= max_ticks:
            continue  # let the top-of-loop check exit without a trailing sleep
        elapsed = time.time() - tick_started
        nap = max(1.0, POLL_SECONDS - elapsed)
        if max_minutes is not None:
            remaining = max_minutes * 60 - (time.time() - started_at)
            if remaining <= 0:
                continue
            nap = min(nap, remaining)
        await asyncio.sleep(nap)
