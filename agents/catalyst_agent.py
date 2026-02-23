from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import re
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import feedparser
import yfinance as yf

from schemas.agents import CatalystEvent
from schemas.config import settings


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

MONTH_PATTERN = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
MONTH_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

EVENT_BASE_IMPACT = {
    "Earnings": 0.90,
    "Regulatory / Approval": 0.85,
    "Product Launch": 0.75,
    "Conference / Investor Day": 0.65,
    "Macro": 0.60,
}

HIGH_QUALITY_SOURCES = {
    "bloomberg.com": 0.95,
    "reuters.com": 0.95,
    "wsj.com": 0.92,
    "finance.yahoo.com": 0.88,
    "marketwatch.com": 0.84,
    "investing.com": 0.80,
    "seekingalpha.com": 0.78,
    "benzinga.com": 0.74,
}


@dataclass
class CatalystConfig:
    mode: str = "deep"  # fast | deep
    lookahead_days: int = 14
    max_feed_entries: int = 10
    max_events: int = 6
    cache_ttl_seconds: int = 3600


def _safe_date_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    try:
        return str(value).split(" ")[0]
    except Exception:
        return None


def _parse_earnings_date(raw: object) -> date | None:
    """Best-effort normalization for yfinance calendar formats."""
    if raw is None:
        return None

    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw

    if isinstance(raw, (list, tuple)):
        return _parse_earnings_date(raw[0]) if raw else None

    if hasattr(raw, "index") and hasattr(raw, "loc"):
        try:
            if "Earnings Date" in raw.index:
                val = raw.loc["Earnings Date"]
                if hasattr(val, "iloc"):
                    return _parse_earnings_date(val.iloc[0])
                return _parse_earnings_date(val)
        except Exception:
            return None

    if hasattr(raw, "get"):
        try:
            return _parse_earnings_date(raw.get("Earnings Date"))
        except Exception:
            return None

    s = _safe_date_str(raw)
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _domain_from_url(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or ""
        return hostname.lower().removeprefix("www.")
    except Exception:
        return ""


def _source_credibility(source_url: str, source_name: str) -> float:
    domain = _domain_from_url(source_url)
    if domain in HIGH_QUALITY_SOURCES:
        return HIGH_QUALITY_SOURCES[domain]
    if any(k in source_name.lower() for k in ("reuters", "bloomberg", "wsj", "yahoo")):
        return 0.88
    return 0.60


def _classify_event(text: str) -> str | None:
    t = text.lower()
    if any(k in t for k in ("earnings", "eps", "quarterly results", "q1", "q2", "q3", "q4", "guidance")):
        return "Earnings"
    if any(k in t for k in ("fda", "approval", "approved", "clearance", "regulator", "sec", "antitrust")):
        return "Regulatory / Approval"
    if any(k in t for k in ("launch", "unveil", "introduce", "release", "new product", "debut")):
        return "Product Launch"
    if any(k in t for k in ("investor day", "conference", "summit", "fireside", "presentation", "webcast")):
        return "Conference / Investor Day"
    if any(k in t for k in ("fomc", "cpi", "ppi", "fed", "rate decision", "jobs report", "payrolls")):
        return "Macro"
    return None


def _next_weekday(base: date, weekday: int) -> date:
    delta = (weekday - base.weekday()) % 7
    if delta == 0:
        delta = 7
    return base + timedelta(days=delta)


def _extract_event_dates(text: str, reference_date: date) -> list[date]:
    lower = text.lower()
    out: set[date] = set()

    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lower):
        month = int(m.group(1))
        day = int(m.group(2))
        year_raw = m.group(3)
        year = reference_date.year
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000
        try:
            candidate = date(year, month, day)
            if candidate < reference_date and not year_raw:
                candidate = date(year + 1, month, day)
            out.add(candidate)
        except ValueError:
            continue

    month_regex = re.compile(
        rf"\b{MONTH_PATTERN}\s+(\d{{1,2}})(?:,?\s+(\d{{2,4}}))?\b",
        re.IGNORECASE,
    )
    for m in month_regex.finditer(text):
        month_name = m.group(1).lower()
        month = MONTH_TO_NUM.get(month_name, 0)
        day = int(m.group(2))
        year_raw = m.group(3)
        year = reference_date.year
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000
        if month:
            try:
                candidate = date(year, month, day)
                if candidate < reference_date and not year_raw:
                    candidate = date(year + 1, month, day)
                out.add(candidate)
            except ValueError:
                continue

    if "tomorrow" in lower:
        out.add(reference_date + timedelta(days=1))
    if "next week" in lower:
        out.add(reference_date + timedelta(days=7))
    if "this week" in lower:
        out.add(reference_date + timedelta(days=max(0, 4 - reference_date.weekday())))

    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, weekday in weekday_map.items():
        if f"next {day_name}" in lower:
            out.add(_next_weekday(reference_date + timedelta(days=1), weekday))
        elif re.search(rf"\bon\s+{day_name}\b", lower):
            out.add(_next_weekday(reference_date - timedelta(days=1), weekday))

    return sorted(out)


def _date_in_window(d: date, now_date: date, lookahead_days: int) -> bool:
    return now_date <= d <= (now_date + timedelta(days=lookahead_days))


class CatalystAgent:
    def __init__(self, config: CatalystConfig):
        self.config = config

    def _cache_path(self, ticker: str) -> Path:
        key = (
            f"{ticker}_{self.config.mode}_{self.config.lookahead_days}_"
            f"{self.config.max_feed_entries}_{self.config.max_events}"
        )
        return settings.cache_dir / f"catalyst_{key}.json"

    def _load_cache(self, ticker: str) -> list[CatalystEvent] | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            saved_at = float(payload.get("saved_at", 0))
            if (datetime.now().timestamp() - saved_at) > self.config.cache_ttl_seconds:
                return None
            return [CatalystEvent(**item) for item in payload.get("events", [])]
        except Exception:
            return None

    def _save_cache(self, ticker: str, events: list[CatalystEvent]) -> None:
        path = self._cache_path(ticker)
        payload = {
            "saved_at": datetime.now().timestamp(),
            "events": [event.model_dump() for event in events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _earnings_event(self, ticker: str, now_date: date) -> list[CatalystEvent]:
        out: list[CatalystEvent] = []
        try:
            ticker_obj = yf.Ticker(ticker)
            cal = getattr(ticker_obj, "calendar", None)
            d = _parse_earnings_date(cal)
            if d is None:
                return out
            if not _date_in_window(d, now_date, self.config.lookahead_days):
                return out

            out.append(
                CatalystEvent(
                    event_type="Earnings",
                    date=d.isoformat(),
                    description=f"{ticker} scheduled earnings date",
                    impact_score=0.92,
                    credibility_score=0.92,
                    source_name="yfinance",
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    duplicate_mentions=1,
                )
            )
        except Exception:
            return out
        return out

    def _from_news_rss(self, ticker: str, now_date: date) -> list[CatalystEvent]:
        if self.config.mode == "fast":
            return []

        query = quote_plus(f"{ticker} stock earnings OR launch OR approval OR conference")
        url = GOOGLE_NEWS_RSS.format(query=query)
        try:
            feed = feedparser.parse(url)
        except Exception:
            return []

        entries = getattr(feed, "entries", [])[: self.config.max_feed_entries]
        if not entries:
            return []

        raw_events: list[dict] = []
        for entry in entries:
            title = str(entry.get("title", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            text = f"{title}. {summary}"
            event_type = _classify_event(text)
            if not event_type:
                continue

            published_parsed = entry.get("published_parsed")
            ref_date = now_date
            if published_parsed:
                try:
                    ref_date = datetime(*published_parsed[:6]).date()
                except Exception:
                    ref_date = now_date

            event_dates = _extract_event_dates(text, ref_date)
            event_dates = [d for d in event_dates if _date_in_window(d, now_date, self.config.lookahead_days)]
            if not event_dates:
                continue

            source_name = ""
            source = entry.get("source")
            if isinstance(source, dict):
                source_name = str(source.get("title", "")).strip()
            source_url = str(entry.get("link", "")).strip()
            credibility = _source_credibility(source_url, source_name)
            base_impact = EVENT_BASE_IMPACT.get(event_type, 0.60)

            for event_date in event_dates:
                desc = title if title else f"{ticker} {event_type.lower()} signal from news"
                key = f"{event_type}|{event_date.isoformat()}|{re.sub(r'\\W+', ' ', desc.lower())[:80]}"
                raw_events.append(
                    {
                        "key": key,
                        "event_type": event_type,
                        "date": event_date.isoformat(),
                        "description": desc[:220],
                        "source_name": source_name or "Google News",
                        "source_url": source_url,
                        "credibility": credibility,
                        "base_impact": base_impact,
                    }
                )

        if not raw_events:
            return []

        grouped: dict[str, list[dict]] = {}
        for item in raw_events:
            grouped.setdefault(item["key"], []).append(item)

        out: list[CatalystEvent] = []
        for _, group in grouped.items():
            first = group[0]
            dup = len(group)
            credibility = max(x["credibility"] for x in group)
            duplicate_boost = min(0.20, 0.04 * max(0, dup - 1))
            impact = min(1.0, (0.7 * first["base_impact"]) + (0.3 * credibility) + duplicate_boost)
            out.append(
                CatalystEvent(
                    event_type=first["event_type"],
                    date=first["date"],
                    description=first["description"],
                    impact_score=round(impact, 3),
                    credibility_score=round(credibility, 3),
                    source_name=first["source_name"],
                    source_url=first["source_url"],
                    duplicate_mentions=dup,
                )
            )

        out.sort(key=lambda e: (e.date, -e.impact_score))
        return out[: self.config.max_events]

    def run(self, ticker: str) -> list[CatalystEvent]:
        cached = self._load_cache(ticker)
        if cached is not None:
            return cached

        now_date = datetime.now().date()
        events = self._earnings_event(ticker, now_date)
        events.extend(self._from_news_rss(ticker, now_date))

        dedup: dict[tuple[str, str, str], CatalystEvent] = {}
        for event in events:
            key = (event.event_type, event.date, event.description.lower())
            if key not in dedup:
                dedup[key] = event
            else:
                existing = dedup[key]
                existing.duplicate_mentions += event.duplicate_mentions
                existing.impact_score = round(min(1.0, max(existing.impact_score, event.impact_score)), 3)
                existing.credibility_score = round(max(existing.credibility_score, event.credibility_score), 3)

        final_events = list(dedup.values())
        final_events.sort(key=lambda e: (-e.impact_score, e.date))
        final_events = final_events[: self.config.max_events]

        self._save_cache(ticker, final_events)
        return final_events


def run_catalyst_agent_fast(ticker: str, lookahead_days: int = 14) -> list[CatalystEvent]:
    cfg = CatalystConfig(
        mode="fast",
        lookahead_days=lookahead_days,
        max_feed_entries=0,
        max_events=2,
        cache_ttl_seconds=3600,
    )
    return CatalystAgent(cfg).run(ticker)


def run_catalyst_agent_deep(ticker: str, lookahead_days: int = 14) -> list[CatalystEvent]:
    cfg = CatalystConfig(
        mode="deep",
        lookahead_days=lookahead_days,
        max_feed_entries=25,
        max_events=10,
        cache_ttl_seconds=7200,
    )
    return CatalystAgent(cfg).run(ticker)


def run_catalyst_agent(ticker: str, lookahead_days: int = 14) -> list[CatalystEvent]:
    return run_catalyst_agent_deep(ticker, lookahead_days=lookahead_days)
