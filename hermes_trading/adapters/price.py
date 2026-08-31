"""Price adapter — OHLCV candles from public exchanges via ccxt.

No API key required. Tries several exchanges in order and uses whichever
answers first — some exchanges (e.g. Binance) geo-block cloud/CI IPs with
HTTP 451, so a single hard-coded exchange is fragile.

Overrides via .env:
  EXCHANGE_ID       force one exchange (skip the fallback list)
  EXCHANGE_API_KEY / EXCHANGE_API_SECRET   use an authenticated client
"""
from __future__ import annotations

import asyncio
import os

SCHEMA_VERSION = 1

# ordered by "works from US cloud IPs without a key, decent 1m history"
DEFAULT_EXCHANGES = ["kraken", "coinbase", "bitstamp", "binance"]


class SchemaError(RuntimeError):
    """Raised when the upstream payload does not match the expected shape."""


def _symbol_variants(asset: str) -> list[str]:
    variants = [asset]
    if asset.endswith("/USDT"):
        variants.append(asset[:-1])  # USDT -> USD (Kraken/Coinbase/Bitstamp)
    elif asset.endswith("/USD"):
        variants.append(asset + "T")
    return variants


def _sync_fetch(asset: str, timeframe: str, limit: int) -> tuple[str, str, list[list[float]]]:
    import ccxt

    forced = os.getenv("EXCHANGE_ID")
    exchanges = [forced] if forced else DEFAULT_EXCHANGES
    key = os.getenv("EXCHANGE_API_KEY") or ""
    secret = os.getenv("EXCHANGE_API_SECRET") or ""

    errors: list[str] = []
    for exid in exchanges:
        if not hasattr(ccxt, exid):
            errors.append(f"{exid}: unknown to ccxt")
            continue
        params: dict = {"enableRateLimit": True, "timeout": 15000}
        if key and secret:
            params["apiKey"] = key
            params["secret"] = secret
        ex = getattr(ccxt, exid)(params)
        for symbol in _symbol_variants(asset):
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if rows and len(rows) >= 2 and len(rows[0]) >= 6:
                    return exid, symbol, rows
                errors.append(f"{exid} {symbol}: thin/none")
            except Exception as exc:  # noqa: BLE001 - ccxt raises many error types
                errors.append(f"{exid} {symbol}: {type(exc).__name__}: {str(exc)[:100]}")
    raise RuntimeError("all price sources failed -> " + " | ".join(errors))


async def fetch(asset: str = "BTC/USDT", timeframe: str = "1m", limit: int = 200) -> dict:
    exid, symbol, rows = await asyncio.to_thread(_sync_fetch, asset, timeframe, limit)
    closes = [float(r[4]) for r in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "asset": symbol,
        "requested_asset": asset,
        "exchange": exid,
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
