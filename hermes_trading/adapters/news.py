"""News adapter — crude headline sentiment.

Free mode (no NEWS_API_KEY) returns a neutral reading so the loop stays fed.
With a key it queries newsapi.org and scores headlines with a small word list.
"""
from __future__ import annotations

import os

import httpx

SCHEMA_VERSION = 1

_POS = {"surge", "rally", "bull", "gain", "soar", "record", "approve", "adopt", "upgrade"}
_NEG = {"crash", "plunge", "bear", "hack", "ban", "lawsuit", "selloff", "fear", "exploit"}


class SchemaError(RuntimeError):
    """Raised when the upstream payload does not match the expected shape."""


def _score_headlines(titles: list[str]) -> float:
    if not titles:
        return 0.0
    s = 0
    for t in titles:
        low = t.lower()
        s += sum(w in low for w in _POS)
        s -= sum(w in low for w in _NEG)
    return max(-1.0, min(1.0, s / len(titles)))


async def fetch(asset: str = "BTC/USDT") -> dict:
    key = os.getenv("NEWS_API_KEY") or ""
    if not key:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "none",
            "sentiment": 0.0,
            "headline_count": 0,
        }

    base = asset.split("/")[0]
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": base,
                "language": "en",
                "pageSize": 50,
                "sortBy": "publishedAt",
                "apiKey": key,
            },
        )
        r.raise_for_status()
        data = r.json()
        if "articles" not in data:
            raise SchemaError("news: unexpected newsapi shape")
        titles = [a.get("title", "") for a in data["articles"]]
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "newsapi",
            "sentiment": _score_headlines(titles),
            "headline_count": len(titles),
        }
