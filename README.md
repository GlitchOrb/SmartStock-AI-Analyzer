# SmartStock AI Analyzer

> Stock analysis system with a 6-agent pipeline, RAG-based news retrieval, and PDF report generation.

Fetches market data via yfinance, runs sentiment analysis on Google News using FAISS + Gemini, and outputs a buy/hold/sell recommendation as a Korean PDF report. Everything runs on free-tier tools.

## Install

```
pip install -r requirements.txt
```

Create `.env` with your [Gemini API key](https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

Optional env vars: `MODEL_NAME` (default: `gemini-2.0-flash`), `CACHE_TTL` (default: `3600`), `REPORT_DEPTH` (default: `standard`).

For Korean PDF output, place [NanumGothic.ttf](https://fonts.google.com/specimen/Nanum+Gothic) in `assets/fonts/`.

## Usage

```
streamlit run app.py
```

## How it works

6 agents run sequentially:

```
DataAgent        - fetches OHLCV + fundamentals, computes 12 technical indicators
ResearchAgent    - fetches Google News RSS, builds FAISS index, retrieves top-5 chunks
SentimentAgent   - sentiment score [-1, +1], pros/cons, 2-5 source citations
AnalysisAgent    - merges quant + qual data into bull/base/bear scenarios
RecommendAgent   - buy/hold/sell + confidence % + invalidation triggers
ReportAgent      - PDF with charts and citations (falls back to .md)
```

3 depth modes: quick (2 LLM calls), standard (4), deep (6).

### RAG pipeline

```
Google News RSS -> feedparser -> difflib dedup (80%) -> LangChain Documents
  -> HuggingFace embeddings (all-MiniLM-L6-v2)
  -> FAISS vector store (disk-cached)
  -> top-5 similarity search
  -> Gemini structured JSON output
```

### Technical indicators

- Momentum: RSI(14)
- Trend: MACD(12,26,9), MA-20, MA-60
- Volatility: Bollinger Bands(20,2), annualized volatility
- Risk: MDD, relative volume
- Fundamentals: PER, PBR, ROE, EPS, debt ratio, market cap, beta, dividend yield

## Project structure

```
app.py                          - streamlit entry point
agents/
  research_agent.py             - RAG news research
  sentiment_agent.py            - sentiment scoring + citations
  analysis_agent.py             - bull/base/bear scenarios
  recommendation_agent.py       - buy/hold/sell rating
rag/
  loader.py                     - Google News RSS fetcher + dedup
  vectorstore.py                - FAISS index + embeddings
data/
  fetcher.py                    - yfinance data + technicals
reporting/
  pdf_generator.py              - ReportLab PDF + matplotlib charts
schemas/
  agents.py                     - pydantic v2 I/O models
  config.py                     - settings loader
  enums.py                      - Signal, Sector, Depth enums
utils/
  gemini.py                     - gemini client (rate-limited)
  logger.py                     - structured logger
  cache.py                      - TTL file cache
assets/fonts/
  NanumGothic.ttf               - Korean font for PDF
```

## Tech stack

- **LLM** - Google Gemini 2.0 Flash
- **RAG** - FAISS + all-MiniLM-L6-v2
- **Data** - yfinance
- **News** - Google News RSS + feedparser
- **Frontend** - Streamlit
- **PDF** - ReportLab + Matplotlib
- **Schemas** - Pydantic v2
- **Orchestration** - LangChain

## Disclaimer

For educational and research purposes only. Not financial advice.

## License

[MIT](LICENSE)
