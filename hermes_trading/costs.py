"""Trading costs for the paper model.

A paper trade that fills at exactly the price it saw, for free, flatters the
result. Real trading pays an exchange fee on each side and usually gets a
slightly worse price than the one on screen (slippage). Both are charged here
as a percent of the traded amount, on entry AND exit.

Defaults can be overridden in state/goal.yaml:

    costs:
      fee_pct_each_way: 0.10
      slippage_pct_each_way: 0.03

`return_pct` in trades.jsonl stays the gross (free) result so history is never
rewritten; scoring uses `net_return_pct`, which is gross minus these costs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

GOAL = Path(__file__).resolve().parent.parent / "state" / "goal.yaml"

DEFAULTS = {"fee_pct_each_way": 0.10, "slippage_pct_each_way": 0.03}


@lru_cache(maxsize=1)
def load() -> dict[str, float]:
    cfg = dict(DEFAULTS)
    try:
        raw = yaml.safe_load(GOAL.read_text(encoding="utf-8-sig")) or {}
        cfg.update({k: float(v) for k, v in (raw.get("costs") or {}).items() if k in DEFAULTS})
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return cfg


def cost_pct(size_r: float = 0.5) -> float:
    """Round-trip cost as a percent of the ACCOUNT. Fees hit the traded
    amount, and only `size_r` of the account is traded, so it scales."""
    c = load()
    return 2.0 * (c["fee_pct_each_way"] + c["slippage_pct_each_way"]) * size_r


def net_return_pct(trade: Mapping[str, Any]) -> float:
    """Trade result after costs. Uses the stored net figure when the loop
    recorded one, otherwise subtracts the configured cost from the gross."""
    if "net_return_pct" in trade:
        return float(trade["net_return_pct"])
    return float(trade.get("return_pct", 0.0)) - cost_pct(float(trade.get("size_r", 0.5)))
