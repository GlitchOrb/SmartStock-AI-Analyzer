# SmartStock AI Analyzer

Realtime NASDAQ momentum scanner with:
- Full NASDAQ universe + prefilter pipeline
- Two-stage scan (FAST -> DEEP) for speed
- Provider abstraction (`alpaca` / `yfinance` fallback)
- FastAPI realtime backend (`/scan`, `/ticker/{ticker}/snapshot`, `WS /stream`)
- Streamlit interactive UI with chart overlays and user plan levels
- Telegram alerts for score surge / threshold crossing / breakout surge

## Architecture

### Core modules
- `marketdata/provider_base.py` - provider interface
- `marketdata/alpaca_provider.py` - realtime + extended-hours capable provider
- `marketdata/yfinance_provider.py` - delayed dev fallback provider
- `data/universe.py` - NASDAQ universe + prefilter
- `api/scanner_engine.py` - fast scan pipeline + scoring/surge logic + alert text
- `api/surge.py` - surge condition engine and evidence payload
- `api/cache_layer.py` - TTL cache layer (in-memory + optional Redis URL)
- `api/server.py` - FastAPI service
- `realtime/aggregator.py` - tick -> 1s/1m aggregation and tape buffers
- `storage/plan_store.py` - SQLite user entry/stop/target persistence
- `storage/watchlist_store.py` - SQLite watchlist persistence
- `alerts/telegram.py` - Telegram Bot API sender
- `ui/streamlit_app.py` - scanner + detail + chart + level editor UI

## Environment Variables

Use `.env.example` as template.

- `MARKET_DATA_PROVIDER=alpaca|yfinance`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_DATA_FEED=iex|sip`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Compatibility aliases (also accepted):
- `API_KEY` (same as `ALPACA_API_KEY`)
- `API_SECRET_KEY` (same as `ALPACA_SECRET_KEY`)

## Windows Quick Start

1. Create venv and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

2. Start backend API:
```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

3. Start Streamlit UI:
```powershell
streamlit run ui/streamlit_app.py
```

4. Open browser:
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

5. Verify provider:
```powershell
python -c "import requests; print(requests.get('http://localhost:8000/health',timeout=10).json())"
```
Expected when Alpaca is configured:
- `"provider": "alpaca"`
- `"provider_configured": true`

## Notes

- Alpaca is recommended for realtime + extended hours streaming.
- yfinance mode is fallback and delayed, with an internal universe cap for stability.
- UI includes one-line warning only and focuses on engineering workflow.
- Scanner API supports two-stage controls:
  - `GET /scan?fast_top_k=120&deep_top_n=40`
  - `GET /scan?full_universe=true` (Alpaca mode)
- Watchlist API:
  - `GET /watchlist`
  - `PUT /watchlist`
  - `POST /watchlist/{ticker}`
  - `DELETE /watchlist/{ticker}`
- Metrics API:
  - `GET /metrics`

## Troubleshooting

- If UI says `No scan results`:
  - check `GET /health` first
  - verify `provider=alpaca` and `provider_configured=true` when using Alpaca
  - if `provider=yfinance`, expect delayed data and capped scan size
- If scan is slow:
  - reduce `FAST top K` / `DEEP top N` in UI
  - keep auto refresh interval at 3-5 minutes
