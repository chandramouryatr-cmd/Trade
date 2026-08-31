"""Score realised trade outcomes against the operator's goal.yaml.

`score(trades, goal)` returns a single float in [-1, +1]:
  +1  -> meeting or beating every part of the goal
   0  -> roughly break-even
  -1  -> at or past the failure floor
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _realised_return(trades: Sequence[Mapping[str, Any]]) -> float:
    """Compound the per-trade returns. `return_pct` is a percent, e.g. 1.5 == +1.5%."""
    equity = 1.0
    for t in trades:
        equity *= 1.0 + float(t.get("return_pct", 0.0)) / 100.0
    return equity - 1.0


def _max_drawdown(trades: Sequence[Mapping[str, Any]]) -> float:
    """Largest peak-to-trough drop of the equity curve, as a positive fraction."""
    equity = peak = 1.0
    worst = 0.0
    for t in trades:
        equity *= 1.0 + float(t.get("return_pct", 0.0)) / 100.0
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def _sharpe(trades: Sequence[Mapping[str, Any]]) -> float:
    """Per-batch Sharpe: mean/stdev of trade returns, scaled by sqrt(n)."""
    rs = [float(t.get("return_pct", 0.0)) / 100.0 for t in trades]
    if len(rs) < 2:
        return 0.0
    mean = sum(rs) / len(rs)
    var = sum((x - mean) ** 2 for x in rs) / (len(rs) - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return 0.0
    return (mean / sd) * math.sqrt(len(rs))


def score(trades: Sequence[Mapping[str, Any]], goal: Mapping[str, Any]) -> float:
    if not trades:
        return 0.0

    target = float(goal.get("target_return_30d", 0.05))
    max_dd = float(goal.get("max_drawdown", 0.08))
    min_sharpe = float(goal.get("min_sharpe", 1.2))
    floor = float(goal.get("failure_below", -0.04))

    ret = _realised_return(trades)
    dd = _max_drawdown(trades)
    sh = _sharpe(trades)

    # return component: +1 at/above target, 0 at break-even, -1 at the floor
    if ret >= 0:
        ret_c = min(1.0, ret / target) if target > 0 else 0.0
    else:
        ret_c = max(-1.0, ret / abs(floor)) if floor < 0 else -1.0

    # drawdown component: +1 with no drawdown, 0 at the limit, -1 at 2x the limit
    dd_c = max(-1.0, 1.0 - dd / max_dd) if max_dd > 0 else 0.0

    # sharpe component: 0 at zero, +1 at the bar
    sh_c = max(-1.0, min(1.0, sh / min_sharpe)) if min_sharpe > 0 else 0.0

    composite = 0.5 * ret_c + 0.3 * dd_c + 0.2 * sh_c
    return max(-1.0, min(1.0, composite))
