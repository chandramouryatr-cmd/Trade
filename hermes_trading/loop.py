"""24/7 reliability loop.

Every minute:
  * pull price (required) + onchain/news/macro context (best-effort) via adapters
  * evaluate the strategy in state/strategy.yaml using the full OHLC candles
    (close for RSI, high/low for ATR if the stop is set to "atr") via
    strategy_engine — the same signal code the backtester uses
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

from . import costs
from . import strategy_engine as se
from .adapters import macro, news, onchain, price
from .adapters.price import SchemaError as PriceSchemaError

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
TRADES = STATE / "trades.jsonl"
HEARTBEAT = STATE / "heartbeat.json"
STRATEGY = STATE / "strategy.yaml"
GOAL = STATE / "goal.yaml"
POSITION = STATE / "position.json"

POLL_SECONDS = 60
MAX_RETRIES = 3
CIRCUIT_BREAK_AFTER = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


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

    def __init__(
        self,
        side: str,
        entry: float,
        size_r: float,
        opened_at: str | None = None,
        peak: float | None = None,
        trough: float | None = None,
    ) -> None:
        self.side = side
        self.entry = entry
        self.size_r = size_r
        self.opened_at = opened_at or _now()
        self.peak = entry if peak is None else peak
        self.trough = entry if trough is None else trough

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "entry": self.entry,
            "size_r": self.size_r,
            "opened_at": self.opened_at,
            "peak": self.peak,
            "trough": self.trough,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Position":
        return cls(
            side=d["side"],
            entry=float(d["entry"]),
            size_r=float(d["size_r"]),
            opened_at=d.get("opened_at"),
            peak=float(d["peak"]) if d.get("peak") is not None else None,
            trough=float(d["trough"]) if d.get("trough") is not None else None,
        )


def _existing_trade_count() -> int:
    if not TRADES.exists():
        return 0
    return sum(1 for l in TRADES.read_text(encoding="utf-8-sig").splitlines() if l.strip())


def _load_position() -> Position | None:
    try:
        raw = json.loads(POSITION.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, ValueError):
        return None
    if not raw:
        return None
    try:
        return Position.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def _save_position(position: Position | None) -> None:
    if position is None:
        POSITION.write_text("null", encoding="utf-8")
    else:
        POSITION.write_text(json.dumps(position.to_dict(), indent=2), encoding="utf-8")


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
    position: Position | None = _load_position()
    closed = _existing_trade_count()  # cumulative across runs, from trades.jsonl
    ticks = 0
    started_at = time.time()
    if position is not None:
        print(
            f"{_now()}  resumed open {position.side} from {position.opened_at} @ {position.entry:.2f}",
            flush=True,
        )
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
            strat = se.migrate_strategy(_load_yaml(STRATEGY))
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

            # full OHLC, not just the close — the same candles feed both the
            # live loop and the backtester now, via strategy_engine
            closes = px["closes"]
            highs = [c["high"] for c in px["candles"]]
            lows = [c["low"] for c in px["candles"]]
            last = px["last"]
            src = px.get("exchange", "?")

            entry_cfg = strat.get("entry", {})
            exit_cfg = strat.get("exit", {})
            rsi_period = int(entry_cfg.get("rsi_period", 14))
            size_r = float(strat.get("position_size_r", 0.5))

            rsi = se.rsi_last(closes, rsi_period)
            sma_val = (
                se.sma_last(closes, int(entry_cfg.get("trend_filter", {}).get("sma_period", 200)))
                if entry_cfg.get("trend_filter", {}).get("enabled", False)
                else None
            )
            atr_val = (
                se.atr_last(highs, lows, closes, int(exit_cfg.get("stop", {}).get("atr_period", 14)))
                if exit_cfg.get("stop", {}).get("method") == "atr"
                else None
            )

            if position is None:
                if se.entry_ok(strat, last, rsi, sma_val):
                    position = Position(entry_cfg.get("direction", "long"), last, size_r)
                    print(f"{_now()}  OPEN  {position.side} @ {last:.2f}  rsi={rsi:.1f}", flush=True)
            else:
                position.peak = max(position.peak, last)
                position.trough = min(position.trough, last)
                hit, reason = se.exit_check(strat, last, position.entry, rsi, atr_val)
                if hit:
                    move = (last - position.entry) / position.entry
                    ret_pct = move * 100 * position.size_r
                    trade_cost = costs.cost_pct(position.size_r)
                    row = {
                        "ts": _now(),
                        "asset": asset,
                        "side": position.side,
                        "entry": position.entry,
                        "exit": last,
                        "size_r": position.size_r,
                        "return_pct": round(ret_pct, 4),
                        "cost_pct": round(trade_cost, 4),
                        "net_return_pct": round(ret_pct - trade_cost, 4),
                        "reason": reason,
                        "rsi_exit": round(rsi, 2),
                        "strategy_version": strat.get("version", "??"),
                        "opened_at": position.opened_at,
                    }
                    _append_jsonl(TRADES, row)
                    closed += 1
                    print(
                        f"{_now()}  CLOSE {position.side} @ {last:.2f}  "
                        f"gross={ret_pct:+.2f}%  net={row['net_return_pct']:+.2f}%  "
                        f"({row['reason']})  total_closed={closed}",
                        flush=True,
                    )
                    position = None

            _save_position(position)
            _write_heartbeat(
                {
                    "ts": _now(),
                    "asset": asset,
                    "price_source": src,
                    "last_price": last,
                    "rsi": round(rsi, 2),
                    "in_position": position is not None,
                    "closed_trades": closed,
                    "strategy_version": strat.get("version", "??"),
                    "consecutive_failures": consecutive_failures,
                }
            )
            print(
                f"{_now()}  tick {ticks}  {src}:{px.get('asset', asset)}  "
                f"last={last:.2f}  rsi={rsi:.1f}  pos={'yes' if position else 'no'}  "
                f"closed={closed}",
                flush=True,
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
