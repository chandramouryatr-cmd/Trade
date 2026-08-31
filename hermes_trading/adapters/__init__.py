"""Data adapters. Each module exposes `async def fetch(...) -> dict` with a
`schema_version` field, and raises `SchemaError` on an unexpected payload shape.
"""
from . import macro, news, onchain, price

__all__ = ["price", "onchain", "news", "macro"]
