# SmartStock AI Analyzer

Realtime NASDAQ momentum scanner with:
- Full NASDAQ universe + prefilter pipeline
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
- `api/server.py` - FastAPI service
- `realtime/aggregator.py` - tick -> 1s/1m aggregation and tape buffers
- `storage/plan_store.py` - SQLite user entry/stop/target persistence
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

## Docker / Cloud Run

### Build images
```bash
docker build -f Dockerfile.api -t gcr.io/<PROJECT_ID>/smartstock-api:latest .
docker build -f Dockerfile.ui -t gcr.io/<PROJECT_ID>/smartstock-ui:latest .
```

### Push images
```bash
docker push gcr.io/<PROJECT_ID>/smartstock-api:latest
docker push gcr.io/<PROJECT_ID>/smartstock-ui:latest
```

### Deploy API (Cloud Run)
```bash
gcloud run deploy smartstock-api \
  --image gcr.io/<PROJECT_ID>/smartstock-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MARKET_DATA_PROVIDER=alpaca,ALPACA_API_KEY=...,ALPACA_SECRET_KEY=...,ALPACA_DATA_FEED=iex,TELEGRAM_BOT_TOKEN=...,TELEGRAM_CHAT_ID=...
```

### Deploy UI (optional Cloud Run)
```bash
gcloud run deploy smartstock-ui \
  --image gcr.io/<PROJECT_ID>/smartstock-ui:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Cloud Scheduler (periodic scan alerts)
```bash
gcloud scheduler jobs create http smartstock-scan-job \
  --schedule="*/5 * * * 1-5" \
  --uri="https://<API_URL>/scan?send_alerts=true" \
  --http-method=GET
```

## Notes

- Alpaca is recommended for realtime + extended hours streaming.
- yfinance mode is fallback and delayed.
- UI includes one-line warning only and focuses on engineering workflow.

