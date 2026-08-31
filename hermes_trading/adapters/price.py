"""Price adapter — OHLCV candles from a public exchange via ccxt.

Free by default (no API key). Set EXCHANGE_API_KEY / EXCHANGE_API_SECRET in .env
to use an authenticated endpoint, and EXCHANGE_ID to pick the exchange.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

SCHEMA_VERSION = 1


class SchemaError(RuntimeError):
    """Raised when the upstream payload does not match the expected shape."""


def _sync_fetch(asset: str, timeframe: str, limit: int) -> list[list[float]]:
    import ccxt

    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    klass = getattr(ccxt, exchange_id)
    params: dict[str, Any] = {"enableRateLimit": True}
    key = os.getenv("EXCHANGE_API_KEY") or ""
    secret = os.getenv("EXCHANGE_API_SECRET") or ""
    if key and secret:
        params["apiKey"] = key
        params["secret"] = secret
    ex = klass(params)
    return ex.fetch_ohlcv(asset, timeframe=timeframe, limit=limit)


async def fetch(asset: str = "BTC/USDT", timeframe: str = "1m", limit: int = 200) -> dict:
    rows = await asyncio.to_thread(_sync_fetch, asset, timeframe, limit)
    if not rows or len(rows[0]) < 6:
        raise SchemaError("price: unexpected OHLCV shape")
    closes = [float(r[4]) for r in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "asset": asset,
        "timeframe": timeframe,
        "candles": [
            {
                "ts": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ],
        "closes": closes,
        "last": closes[-1],
    }
