# Crypto Alpha Research Engine

A local-first, point-in-time research system for discovering and rejecting crypto-market
hypotheses. The MVP ingests BTC market data, creates causal features, classifies regimes,
finds historical analogues, evaluates hypotheses, trains time-series models, and runs
cost-aware baseline backtests.

> **Research software, not financial advice.** This release is paper-trading only. It has
> no withdrawal route, exchange credential model, leverage, shorting, or live-order client.
> Synthetic demo results are plumbing tests and are not evidence of an edge.

## Production terminal

The Vercel adapter is BTC-only and never substitutes synthetic observations:

- Kraken BTC/USD is the primary live spot and OHLCV source.
- Coinbase Exchange BTC-USD is a quote-level cross-check.
- Kraken Futures supplies BTC perpetual funding, open interest, basis, liquidations, and
  long/short positioning.
- Deribit supplies a public BTC options snapshot (near-ATM mark IV, option volume, and
  put/call open-interest ratio).
- DefiLlama supplies aggregate stablecoin supply, mempool.space supplies Bitcoin network
  fee pressure, Alternative.me supplies its attributed BTC Fear & Greed index, and public
  daily market charts supply the US 10Y, dollar index, Nasdaq, gold, and VIX.
- A cross-venue deviation above 75 basis points stops the response.
- Kraken's uncommitted final OHLC candle is excluded.
- Polymarket's public Gamma API supplies active BTC prediction-market context.
- Neon Postgres stores issued forecasts and realized outcomes across deployments.
- A protected Vercel Cron route issues and settles forecasts every 15 minutes.
- If a required market feed is unavailable, the UI says `DATA UNAVAILABLE`.
- ETF flows, exchange reserves/miner flows, wallet clusters, and information velocity are
  labeled `PROVIDER_REQUIRED` until licensed point-in-time feeds are configured.

Eight fixed research horizons are produced from completed candles:

| Display horizon | Training bars | Forecast steps |
|---|---:|---:|
| 15 minutes | 15m | 1 |
| 1 hour | 1h | 1 |
| 4 hours | 4h | 1 |
| 8 hours | 4h | 2 |
| 12 hours | 4h | 3 |
| Daily | 1d | 1 |
| Weekly | 1w | 1 |
| Monthly | 15d | 2 |

The baseline is a fixed regularized linear model using causal OHLCV momentum, volatility,
trend, range, and volume features. Each horizon is trained separately. The UI reports a
base price, empirical 80% residual range, chronological holdout sample count, direction
accuracy, expanding-window walk-forward accuracy, cost-aware baseline comparison, median
absolute price error, mean absolute return error, and range coverage. Each forecast also
publishes ranked feature contributions, a source-to-output decision trace, embargoed
historical analog states, and fixed-threshold hypothesis results.
These are historical validation statistics—not a live trading track record or guarantee.

Prediction-market probabilities are visible as public context. Their model weight remains
zero until timestamped history has enough settled samples for point-in-time validation.
This prevents a current, untested probability from being presented as a proven forecast
input.

Read-only endpoints:

```text
GET  /api/market/live
GET  /api/forecast/live
GET  /api/prediction-markets
GET  /api/intelligence/live
GET  /api/tracking
GET  /api/research/terminal
GET  /api/cron/forecasts   # Vercel Cron only; requires CRON_SECRET
POST /api/research/run
```

## Quick start

Requirements: Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
alpha demo --days 1200
uvicorn alpha_engine.api:app --reload
```

Open [http://localhost:8000](http://localhost:8000). The same flow runs in Docker:

```bash
docker compose up --build
```

The default demo is deterministic and offline. To try public Binance daily candles:

```bash
alpha demo --days 1200 --source binance
```

Availability of that endpoint varies by jurisdiction. No authentication is used.

## What is implemented

| Area | MVP implementation |
|---|---|
| Spot and derivatives | Paginated public Binance spot/USD-M candle adapters |
| Funding | Public historical funding adapter |
| ETF/options/OI/basis | Point-in-time CSV adapters for licensed or manual exports |
| Stablecoin/on-chain/macro/sentiment | Same validated adapter contract |
| Storage | DuckDB tables plus Zstandard-compressed Parquet snapshots |
| Features | Momentum, volatility, trend, drawdown, volume, funding, OI, basis, stablecoin and sentiment |
| Regimes | Explainable bull expansion/exhaustion, bear, capitulation and neutral baseline |
| Market DNA | Causal standardized nearest-neighbor search with embargo |
| Hypotheses | Train-derived thresholds, held-out validation, stability and uplift |
| Models | Random forest with purged time-series cross-validation |
| Backtesting | Delayed fills, turnover, fees, slippage, drawdown and exposure |
| Tracking | Immutable JSONL runs with data fingerprints; MLflow-compatible field model |
| Execution | In-memory long-only paper broker behind hard risk limits |
| Product surface | FastAPI, OpenAPI, Palantir/Bloomberg-style BTC terminal, CLI |
| Forecasting | Eight horizon-specific ridge baselines with chronological holdout scores |
| Prediction markets | Public Polymarket BTC market probabilities and directional context |
| Forecast ledger | Idempotent DuckDB issuance, settlement, and horizon/model scorecards |

## Architecture

```text
Public/CSV/Synthetic sources
          │
          ▼
 Point-in-time validation ─── rejects impossible release timestamps
          │
          ▼
 DuckDB catalog + Parquet snapshots
          │
          ▼
 Feature store ─── one-bar lag + max_available_at lineage
          │
          ├── Regime detection
          ├── Historical state similarity
          ├── Hypothesis discovery/validation
          └── Purged time-series model training
                          │
                          ▼
          Cost-aware walk-forward backtests
                          │
                          ▼
             API/dashboard + paper broker
```

Source code lives under `src/alpha_engine`:

- `data/`: connector contracts, public/CSV adapters, DuckDB/Parquet store
- `research/`: leakage controls, feature store, regimes, similarity, hypotheses
- `backtesting/`: strategies, realistic costs, metrics, walk-forward folds
- `modeling.py`: purged time-series training
- `risk.py` and `paper.py`: exposure/drawdown controls and simulated fills
- `forecast_tracking.py`: immutable forecast issuance, settlement and promotion gates
- `pipeline.py`, `cli.py`, and `api.py`: end-to-end orchestration

## Accuracy and model-evolution policy

“Accurate data” and “accurate forecasts” are separate claims. The system can enforce
source provenance, timestamps, schema checks, completed bars, cross-venue checks, and
fail-closed behavior. Future prices remain uncertain.

Every issued forecast is stored in Neon with its model version, issue time, target time,
anchor price, predicted price, range, and prediction-market snapshot. The decision time
is the close time of the latest completed 15-minute bar, making issuance idempotent even
when the dashboard is refreshed repeatedly. Once a target time passes, the first completed
15-minute close at or after the target settles it exactly once. Accuracy is calculated
separately by horizon and model version.

Retraining follows a champion/challenger policy:

1. Freeze a point-in-time training snapshot.
2. Train the challenger with purged time-series folds.
3. Score it on later, untouched data and accumulated live forecasts.
4. Require at least 100 settled predictions for the evaluated horizon.
5. Reject it if price error, direction accuracy, or interval calibration regresses.
6. Require manual approval even when every automated gate passes.

The engine never autonomously increases risk, enables leverage, or promotes a model.

## Point-in-time and leakage policy

Every input must contain:

- `timestamp`: when the economic event occurred;
- `available_at`: when a strategy could actually have known it.

Raw values are delayed before features are calculated. Feature rows retain
`max_available_at`, and validation fails if it exceeds the decision timestamp. Similarity
search fits its scaler only on eligible history and applies an embargo. Model folds purge
the forecast horizon between train and test. Backtest signals execute one bar later.

These controls reduce common leakage paths; they do not prove a dataset is truly
point-in-time. Vendor revision policies still need independent review.

## Bringing real data

See `config/connectors.example.yaml`. CSV-backed sources must include `timestamp` and
`available_at`, plus one or more numeric values. Recommended initial columns:

| File | Suggested fields |
|---|---|
| `etf_flows.csv` | `net_flow_usd`, issuer-level flows |
| `open_interest.csv` | `open_interest`, venue |
| `basis.csv` | annualized 1m/3m futures basis |
| `options.csv` | ATM IV, 25-delta skew, put/call OI |
| `stablecoins.csv` | supply by asset, exchange inflows |
| `onchain.csv` | exchange balances, realized cap, miner flows |
| `macro.csv` | DXY, yields, Nasdaq, gold, VIX |
| `sentiment.csv` | source volume, polarity, information velocity |

Provider credentials are optional and must be injected as environment variables. Never put
secrets in YAML, notebooks, source files, images, or experiment parameters.

## Research discipline

The engine should be used to falsify ideas:

1. Write the economic reason a signal might persist.
2. Freeze source definitions and release-time assumptions.
3. Discover only on the training window.
4. Evaluate once on untouched regimes and later dates.
5. Stress fees, slippage, missing data and execution delay.
6. Compare with BTC buy-and-hold and a simple regime baseline.
7. Paper trade before considering any separately reviewed live-execution system.

Do not interpret high synthetic-data metrics as performance. Do not promote hypotheses with
small samples, unstable parameters, weak held-out results, or dependence on one market era.

## Development

```bash
make install
make lint
make test
make demo
```

The intended CI workflow runs Ruff, strict MyPy, tests with a coverage floor, and a basic
committed-secret scan on Python 3.12 and 3.13.

The ready-to-install workflow is stored at `config/github-actions-ci.yml.example`. Move it
to `.github/workflows/ci.yml` using a GitHub credential with `workflow` permission; GitHub
rejects workflow-path writes from credentials without that separate scope.

## Roadmap

### Phase 1 — research MVP (this repository)

- Reproducible local pipeline, connector contracts, causal features and storage
- Regimes, similarities, hypotheses, cross-validation and baseline backtests
- Experiment records, API/dashboard, paper broker, tests, Docker and CI

### Phase 2 — data quality and evidence

- Venue-normalized Kraken/Coinbase/Deribit adapters and resumable ingestion
- Dataset manifests, checksums, revision snapshots, freshness/quality monitoring
- Licensed ETF, options and on-chain integrations
- Deflated Sharpe, probability of backtest overfitting, bootstrap/Monte Carlo tests
- Richer walk-forward orchestration and model registry

### Phase 3 — supervised paper operation

- Durable paper ledger, exchange WebSocket market data and reconciliation
- Alerts, stale-feed shutdown, duplicate-order prevention and operator kill switch
- Shadow execution, latency/slippage attribution and multi-month acceptance gates

### Explicitly out of scope

- Withdrawal permissions
- Live or leveraged trading
- Autonomous strategy promotion or position-size increases
- Custody of funds or exchange secrets

Any future live execution should be a separate service and security review, not a flag added
to this research process.
