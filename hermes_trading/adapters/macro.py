"""Macro adapter — broad risk backdrop from public market data via yfinance.

Pulls the dollar index, S&P 500 and VIX. No API key required.
"""
from __future__ import annotations

import asyncio

SCHEMA_VERSION = 1


class SchemaError(RuntimeError):
    """Raised when the upstream payload does not match the expected shape."""


def _sync_fetch() -> dict[str, float]:
    import yfinance as yf

    out: dict[str, float] = {}
    for sym in ("DX-Y.NYB", "^GSPC", "^VIX"):
        try:
            hist = yf.Ticker(sym).history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                out[sym] = float(hist["Close"].iloc[-1])
        except Exception:  # noqa: BLE001 - a single missing series is tolerable
            continue
    return out


async def fetch(asset: str = "BTC/USDT") -> dict:
    data = await asyncio.to_thread(_sync_fetch)
    if not data:
        raise SchemaError("macro: no series returned")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "yfinance",
        "dxy": data.get("DX-Y.NYB"),
        "sp500": data.get("^GSPC"),
        "vix": data.get("^VIX"),
    }
