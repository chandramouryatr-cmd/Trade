"""Replay a strategy over historical candles, with an honest train/check split.

  uv run python -m hermes_trading.backtest --compare

Pulls ~2 years of hourly BTC candles (yfinance, no key), splits them into an
older TRAIN slice and a newer CHECK slice, runs a small parameter search on
TRAIN only, then reports how both the current naive rule and the searched
"smart" rule did on the CHECK slice they never saw.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from . import strategy_engine as se
from .score import _max_drawdown, _realised_return, _sharpe, score

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
OUT = STATE / "backtest"
DATA = STATE / "market_data"

WARMUP = 220  # candles reserved so SMA(200)/RSI/ATR are all valid before trade 1

# current live design, expressed in the richer strategy shape
NAIVE = {
    "entry": {
        "direction": "long",
        "indicator": "rsi",
        "threshold": 30,
        "rsi_period": 14,
        "trend_filter": {"enabled": False},
    },
    "exit": {
        "rsi_exit": 70,
        "take_profit_pct": None,
        "stop": {"method": "fixed", "fixed_pct": 2.0},
    },
    "position_size_r": 0.5,
}

# grid searched on TRAIN only
GRID = {
    "threshold": [28, 32, 36, 40],
    "sma_period": [120, 200, 320],
    "rsi_exit": [55, 60, 65],
    "take_profit_pct": [3.0, 5.0, 8.0],
    "atr_mult": [2.0, 2.75, 3.5],
}


def fetch_history(symbol: str = "BTC-USD", interval: str = "1h", period: str = "730d") -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / f"{symbol}_{interval}_{period}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import yfinance as yf

    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"no history returned for {symbol} {interval} {period}")
    candles = [
        {
            "ts": ix.isoformat(),
            "open": float(row.Open),
            "high": float(row.High),
            "low": float(row.Low),
            "close": float(row.Close),
            "volume": float(row.Volume),
        }
        for ix, row in zip(df.index, df.itertuples())
    ]
    cache.write_text(json.dumps(candles), encoding="utf-8")
    return candles


def _smart_cfg(threshold, sma_period, rsi_exit, take_profit_pct, atr_mult) -> dict:
    return {
        "entry": {
            "direction": "long",
            "indicator": "rsi",
            "threshold": threshold,
            "rsi_period": 14,
            "trend_filter": {"enabled": True, "sma_period": sma_period},
        },
        "exit": {
            "rsi_exit": rsi_exit,
            "take_profit_pct": take_profit_pct,
            "stop": {"method": "atr", "atr_period": 14, "atr_mult": atr_mult},
        },
        "position_size_r": 0.5,
    }


def simulate(candles: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)

    rsi_p = int(cfg["entry"].get("rsi_period", 14))
    sma_p = int(cfg["entry"].get("trend_filter", {}).get("sma_period", 200))
    atr_p = int(cfg["exit"].get("stop", {}).get("atr_period", 14))
    size_r = float(cfg.get("position_size_r", 0.5))

    rsi = se.rsi_series(closes, rsi_p)
    sma = se.sma_series(closes, sma_p)
    atr = se.atr_series(highs, lows, closes, atr_p)

    trades: list[dict] = []
    curve: list[dict] = []
    equity = 1.0
    entry_price = 0.0
    entry_i = -1

    for i in range(WARMUP, len(candles)):
        px = closes[i]
        sma_v = None if np.isnan(sma[i]) else float(sma[i])
        atr_v = None if np.isnan(atr[i]) else float(atr[i])

        if entry_i < 0:
            if se.entry_ok(cfg, px, rsi[i], sma_v):
                entry_price, entry_i = px, i
        else:
            hit, reason = se.exit_check(cfg, px, entry_price, rsi[i], atr_v)
            if hit:
                move = float((px - entry_price) / entry_price)
                equity *= 1.0 + move * size_r
                trades.append(
                    {
                        "entry_ts": candles[entry_i]["ts"],
                        "exit_ts": candles[i]["ts"],
                        "entry": round(float(entry_price), 2),
                        "exit": round(float(px), 2),
                        "return_pct": round(move * 100.0 * size_r, 4),
                        "reason": reason,
                        "bars_held": i - entry_i,
                    }
                )
                entry_i = -1
        curve.append({"ts": candles[i]["ts"], "equity": round(equity, 6)})

    return trades, curve


def metrics(trades: list[dict], goal: dict) -> dict:
    if not trades:
        return {"trades": 0, "total_return_pct": 0.0, "score_vs_goal": 0.0}
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    return {
        "trades": len(trades),
        "total_return_pct": round(_realised_return(trades) * 100.0, 3),
        "max_drawdown_pct": round(_max_drawdown(trades) * 100.0, 3),
        "sharpe": round(_sharpe(trades), 3),
        "win_rate_pct": round(100.0 * wins / len(trades), 1),
        "avg_trade_pct": round(float(sum(t["return_pct"] for t in trades)) / len(trades), 4),
        "score_vs_goal": round(score(trades, goal), 3),
    }


def _downsample(curve: list[dict], n: int = 160) -> list[dict]:
    if len(curve) <= n:
        return curve
    step = len(curve) / n
    return [curve[min(int(k * step), len(curve) - 1)] for k in range(n)] + [curve[-1]]


def search(train: list[dict], goal: dict) -> tuple[dict, dict]:
    best_cfg, best_m, best_key = None, None, -1e9
    combos = list(itertools.product(*GRID.values()))
    for threshold, sma_period, rsi_exit, tp, atr_mult in combos:
        cfg = _smart_cfg(threshold, sma_period, rsi_exit, tp, atr_mult)
        tr, _ = simulate(train, cfg)
        if len(tr) < 8:  # too few trades to trust
            continue
        m = metrics(tr, goal)
        key = m["score_vs_goal"] - 0.15 * max(0.0, m["max_drawdown_pct"] - 100 * goal["max_drawdown"])
        if key > best_key:
            best_cfg, best_m, best_key = cfg, m, key
    return best_cfg, best_m


def main() -> None:
    ap = argparse.ArgumentParser(prog="hermes_trading.backtest")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--split", type=float, default=0.7, help="fraction of history used for TRAIN")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    goal = yaml.safe_load((STATE / "goal.yaml").read_text(encoding="utf-8-sig")) or {}
    goal.setdefault("target_return_30d", 0.05)
    goal.setdefault("max_drawdown", 0.08)
    goal.setdefault("min_sharpe", 1.2)
    goal.setdefault("failure_below", -0.04)

    candles = fetch_history(args.symbol, args.interval, args.period)
    cut = int(len(candles) * args.split)
    train, check = candles[:cut], candles[cut - WARMUP :]
    span = f"{candles[0]['ts'][:10]} .. {candles[-1]['ts'][:10]}"
    print(f"{len(candles)} {args.interval} candles  ({span})", flush=True)
    print(f"train: {len(train)}   check: {len(check)}   split @ {candles[cut]['ts'][:10]}", flush=True)

    naive_train_tr, _ = simulate(train, NAIVE)
    naive_check_tr, naive_check_curve = simulate(check, NAIVE)
    print("\nNAIVE (current live design)")
    print("  train:", metrics(naive_train_tr, goal))
    print("  check:", metrics(naive_check_tr, goal))

    smart_cfg, smart_train_m = search(train, goal)
    if smart_cfg is None:
        print("\nsmart search found nothing with enough trades — stopping")
        return
    smart_check_tr, smart_check_curve = simulate(check, smart_cfg)
    print("\nSMART (searched on train only)")
    print("  params:", json.dumps(smart_cfg["entry"]) )
    print("        ", json.dumps(smart_cfg["exit"]))
    print("  train:", smart_train_m)
    print("  check:", metrics(smart_check_tr, goal))

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "span": span,
        "interval": args.interval,
        "split_at": candles[cut]["ts"][:10],
        "goal": goal,
        "naive": {
            "cfg": NAIVE,
            "train": metrics(naive_train_tr, goal),
            "check": metrics(naive_check_tr, goal),
        },
        "smart": {
            "cfg": smart_cfg,
            "train": smart_train_m,
            "check": metrics(smart_check_tr, goal),
        },
        "check_equity": {
            "naive": _downsample(naive_check_curve),
            "smart": _downsample(smart_check_curve),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "smart_strategy.yaml").write_text(
        yaml.safe_dump({"version": "smart-candidate", **smart_cfg}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'summary.json'} and {OUT / 'smart_strategy.yaml'}")


if __name__ == "__main__":
    main()
