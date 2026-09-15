"""Shared strategy math and signals.

One source of truth for both the live loop (`loop.py`) and the backtester
(`backtest.py`), so a rule can never mean two different things in the two
places. Indicator functions come in two forms:

  * `*_series(...)`  -> full numpy array, one value per candle (fast for backtests)
  * `*_last(...)`    -> just the final value (convenient for the live loop)

Signals (`entry_ok`, `exit_check`) take already-computed indicator values so
the caller controls how they were produced.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# indicators
# --------------------------------------------------------------------------- #
def rsi_series(closes, period: int = 14) -> np.ndarray:
    """Wilder's RSI. Returns an array the same length as `closes` (first
    `period` entries are warm-up and set to 50)."""
    c = np.asarray(closes, dtype=float)
    out = np.full(c.size, 50.0)
    if c.size < period + 1:
        return out
    delta = np.diff(c)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, delta.size):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0.0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def sma_series(closes, period: int) -> np.ndarray:
    """Simple moving average; entries before a full window are NaN."""
    c = np.asarray(closes, dtype=float)
    out = np.full(c.size, np.nan)
    if period <= 0 or c.size < period:
        return out
    csum = np.cumsum(np.insert(c, 0, 0.0))
    out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def atr_series(highs, lows, closes, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range; warm-up entries are NaN."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    atr = tr[:period].mean()
    out[period] = atr
    for i in range(period, tr.size):
        atr = (atr * (period - 1) + tr[i]) / period
        out[i + 1] = atr
    return out


def rsi_last(closes, period: int = 14) -> float:
    return float(rsi_series(closes, period)[-1])


def sma_last(closes, period: int) -> float | None:
    v = sma_series(closes, period)[-1]
    return None if np.isnan(v) else float(v)


def atr_last(highs, lows, closes, period: int = 14) -> float | None:
    v = atr_series(highs, lows, closes, period)[-1]
    return None if np.isnan(v) else float(v)


# --------------------------------------------------------------------------- #
# strategy shape
# --------------------------------------------------------------------------- #
def migrate_strategy(strat: dict) -> dict:
    """Upgrade an old flat strategy.yaml (threshold/exit_rsi/stop_loss_pct at
    the top level, no `exit` block) to the richer shape this module expects
    (a `exit.stop` block, an `entry.trend_filter` block). Values carry over
    unchanged; this only reshapes the dict. Safe to call on an
    already-migrated strategy — it is left untouched."""
    if "exit" in strat:
        return strat

    entry = dict(strat.get("entry", {}))
    exit_rsi = entry.pop("exit_rsi", 70)
    entry.setdefault("trend_filter", {"enabled": False, "sma_period": 200})

    return {
        "version": strat.get("version", "01"),
        "entry": entry,
        "exit": {
            "rsi_exit": exit_rsi,
            "take_profit_pct": None,
            "stop": {
                "method": "fixed",
                "fixed_pct": float(strat.get("stop_loss_pct", 2.0)),
                "atr_period": 14,
                "atr_mult": 2.5,
            },
        },
        "position_size_r": strat.get("position_size_r", 0.5),
    }


# --------------------------------------------------------------------------- #
# signals  (long-only for now)
# --------------------------------------------------------------------------- #
def entry_ok(cfg: dict, price: float, rsi_val: float, sma_val: float | None) -> bool:
    e = cfg.get("entry", {})
    if e.get("direction", "long") != "long":
        return False
    if rsi_val > float(e.get("threshold", 30)):
        return False
    tf = e.get("trend_filter", {})
    if tf.get("enabled", False):
        if sma_val is None or price <= sma_val:
            return False
    return True


def exit_check(
    cfg: dict,
    price: float,
    entry_price: float,
    rsi_val: float,
    atr_val: float | None,
) -> tuple[bool, str]:
    x = cfg.get("exit", {})
    move_pct = (price - entry_price) / entry_price * 100.0

    tp = x.get("take_profit_pct")
    if tp is not None and move_pct >= float(tp):
        return True, "take_profit"

    st = x.get("stop", {})
    method = st.get("method", "fixed")
    if method == "atr" and atr_val:
        if price <= entry_price - float(st.get("atr_mult", 2.5)) * atr_val:
            return True, "stop_loss"
    else:
        if price <= entry_price * (1.0 - float(st.get("fixed_pct", 2.0)) / 100.0):
            return True, "stop_loss"

    rx = x.get("rsi_exit")
    if rx is not None and rsi_val >= float(rx):
        return True, "rsi_exit"

    return False, ""
