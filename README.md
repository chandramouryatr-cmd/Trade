# hermes-trading

A self-improving **paper-trading** worker. It pulls live market data, takes
simulated trades against an RSI strategy, logs every outcome, and periodically
revises **exactly one** strategy variable based on results.

Nothing here touches a real exchange. The live-execution path is not imported
unless both flags in `.env` are flipped (`HERMES_TRADING_MODE=live` and
`HERMES_TRADING_I_ACCEPT_RISK=true`).

## Layout

| Path | Purpose |
|------|---------|
| `hermes_trading/run.py` | entrypoint — resolves asset from `state/goal.yaml` |
| `hermes_trading/loop.py` | async loop: RSI entry, stop-loss / RSI exit, retries, circuit breaker |
| `hermes_trading/reflect.py` | reflection cycle — `--fallback` (deterministic) or `--hermes` |
| `hermes_trading/score.py` | `score(trades, goal) -> [-1, +1]` |
| `hermes_trading/adapters/` | price (ccxt) · onchain · news · macro (yfinance) |
| `state/goal.yaml` | success / failure definition — **config, edit freely** |
| `state/strategy.yaml` | current strategy — bumped each reflection |
| `state/history/` | every prior strategy version |
| `state/trades.jsonl` | every closed paper trade |
| `state/hypotheses.jsonl` | every reflection decision + rationale |

## Run locally

```bash
uv sync
uv run python -m hermes_trading.run --minutes 12   # bounded
uv run python -m hermes_trading.run                # forever
uv run python -m hermes_trading.reflect --fallback # force one reflection
```

## Backtesting

`hermes_trading/strategy_engine.py` holds the indicator math and entry/exit
rules shared by the live loop and the backtester, so the two can never
silently disagree.

```bash
uv run python -m hermes_trading.backtest --compare
```

Pulls ~2 years of hourly BTC candles (yfinance, cached under
`state/market_data/`, gitignored), splits them into an older TRAIN slice and
a newer CHECK slice, searches a small grid on TRAIN only, and reports both
the current live rule and the searched rule on the CHECK slice they never
saw. Writes `state/backtest/summary.json` (also gitignored — regenerate any
time). Finding: neither beat the goal on unseen data — see the conversation
history for the honest read.

### The reflection rule has bounds, not an open-ended ratchet

`reflect.py --fallback` clips every variable it can touch to a `BOUNDS`
range (e.g. `entry.threshold` stays within 20-45) and can move a value
*back* when the strategy is actually meeting the return target, not just
loosen forever. See `BOUNDS` and `_fallback` in `hermes_trading/reflect.py`.

## Hosting (free, via GitHub Actions)

`.github/workflows/trade.yml` runs the worker on a schedule (~every 15-30 min),
for a 12-minute window each time, and carries `state/` forward between runs via
the Actions cache. A downloadable `hermes-state` artifact is published each run.

Requirements:
- **Public repo** for unlimited Actions minutes (there are no secrets in here).
- Actions enabled, and the `trade.yml` workflow enabled on the Actions tab.
- At least one commit every 60 days, or GitHub disables the schedule.
