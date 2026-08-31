"""On-chain adapter — basic network stats.

Free fallback uses blockchain.info (BTC network). Set GLASSNODE_API_KEY in .env
to pull active-address counts from Glassnode instead.
"""
from __future__ import annotations

import os

import httpx

SCHEMA_VERSION = 1


class SchemaError(RuntimeError):
    """Raised when the upstream payload does not match the expected shape."""


async def fetch(asset: str = "BTC/USDT") -> dict:
    key = os.getenv("GLASSNODE_API_KEY") or ""
    async with httpx.AsyncClient(timeout=15) as client:
        if key:
            base = asset.split("/")[0].upper()
            r = await client.get(
                "https://api.glassnode.com/v1/metrics/addresses/active_count",
                params={"a": base, "api_key": key, "i": "24h"},
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data or "v" not in data[-1]:
                raise SchemaError("onchain: unexpected glassnode shape")
            return {
                "schema_version": SCHEMA_VERSION,
                "source": "glassnode",
                "active_addresses": float(data[-1]["v"]),
            }

        r = await client.get("https://api.blockchain.info/stats", params={"format": "json"})
        r.raise_for_status()
        data = r.json()
        if "n_tx" not in data or "hash_rate" not in data:
            raise SchemaError("onchain: unexpected blockchain.info shape")
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "blockchain.info",
            "n_tx_24h": float(data.get("n_tx", 0.0)),
            "hash_rate": float(data.get("hash_rate", 0.0)),
            "difficulty": float(data.get("difficulty", 0.0)),
        }
