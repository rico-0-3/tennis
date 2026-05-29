"""
prediccion/sentiment_engine.py
================================
Analisi del sentiment pre-partita usando notizie da fonti pubbliche gratuite:
  - Google News RSS      (copertura globale, no API key)
  - Reddit r/tennis      (community sentiment, API JSON pubblica)
  - BBC Sport Tennis     (RSS ufficiale, no API key)
  - Tennis Abstract RSS  (news specializzate, no API key)

I testi vengono poi interpretati da Groq (llama-3.1-8b-instant, free tier)
che restituisce una probabilità di vittoria basata sul sentiment e i fattori chiave.

Nota: Twitter/X richiede API a pagamento dal 2023 — escluso.
"""

import re
import json
import time
import feedparser
import requests
from datetime import datetime, timedelta
from urllib.parse import quote_plus


# ── Costanti ──────────────────────────────────────────────────────────────────

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.1-8b-instant"
MAX_HEADLINES  = 8    # per fonte per giocatore
REQUEST_TIMEOUT = 10  # secondi

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": _UA}


# ── Fetch: Google News RSS ─────────────────────────────────────────────────────

def _fetch_google_news(player: str, days_back: int = 7) -> list:
    q   = quote_plus(f"{player} tennis")
    url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"
    try:
        feed   = feedparser.parse(url)
        cutoff = datetime.now() - timedelta(days=days_back)
        out    = []
        for e in feed.entries[:20]:
            try:
                pub = datetime(*e.published_parsed[:6])
                if pub < cutoff:
                    continue
            except Exception:
                pass
            out.append(e.title)
            if len(out) >= MAX_HEADLINES:
                break
        return out
    except Exception:
        return []


# ── Fetch: Reddit r/tennis ─────────────────────────────────────────────────────

def _fetch_reddit(player: str, days_back: int = 7) -> list:
    q   = quote_plus(player)
    url = (
        f"https://www.reddit.com/r/tennis/search.json"
        f"?q={q}&sort=new&t=week&limit=20&restrict_sr=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        posts    = r.json().get("data", {}).get("children", [])
        cutoff_t = time.time() - days_back * 86400
        out      = []
        for p in posts:
            d = p.get("data", {})
            if d.get("created_utc", 0) < cutoff_t:
                continue
            title = d.get("title", "").strip()
            if title:
                out.append(title)
            if len(out) >= MAX_HEADLINES:
                break
        return out
    except Exception:
        return []


# ── Fetch: BBC Sport Tennis RSS ────────────────────────────────────────────────

def _fetch_bbc(player: str) -> list:
    """Titoli BBC Sport Tennis filtrati per nome giocatore."""
    url = "https://feeds.bbci.co.uk/sport/tennis/rss.xml"
    try:
        feed = feedparser.parse(url)
        name_lower = player.lower().split()[-1]  # usa cognome
        out = []
        for e in feed.entries[:30]:
            if name_lower in e.title.lower():
                out.append(e.title)
                if len(out) >= 4:
                    break
        return out
    except Exception:
        return []


# ── Fetch: Tennis Abstract (feed notizie) ─────────────────────────────────────

def _fetch_tennisabstract(player: str) -> list:
    url = "https://www.tennisabstract.com/blog/feed/"
    try:
        feed = feedparser.parse(url)
        name_lower = player.lower().split()[-1]
        out = []
        for e in feed.entries[:20]:
            if name_lower in e.title.lower() or (
                hasattr(e, "summary") and name_lower in e.summary.lower()
            ):
                out.append(e.title)
                if len(out) >= 4:
                    break
        return out
    except Exception:
        return []


# ── Aggregatore ───────────────────────────────────────────────────────────────

def fetch_all_news(player: str, days_back: int = 7) -> dict:
    """
    Aggrega notizie da tutte le fonti disponibili.
    Ritorna: {'google': [...], 'reddit': [...], 'bbc': [...], 'tennisabstract': [...]}
    """
    return {
        "google":         _fetch_google_news(player, days_back),
        "reddit":         _fetch_reddit(player, days_back),
        "bbc":            _fetch_bbc(player),
        "tennisabstract": _fetch_tennisabstract(player),
    }


def _format_news(news_dict: dict) -> str:
    lines = []
    for src, items in news_dict.items():
        for item in items:
            lines.append(f"[{src.upper()}] {item}")
    return "\n".join(lines) if lines else "No recent news found."


def count_headlines(news_dict: dict) -> int:
    return sum(len(v) for v in news_dict.values())


# ── Groq Analysis ─────────────────────────────────────────────────────────────

def _build_prompt(player1: str, player2: str,
                  news1: dict, news2: dict,
                  tournament: str, surface: str) -> str:
    return f"""You are an expert tennis analyst. Analyze recent news to assess pre-match sentiment.

Match: {player1} vs {player2} | {tournament} | {surface} court

=== {player1} — Recent News ===
{_format_news(news1)}

=== {player2} — Recent News ===
{_format_news(news2)}

Analyze for: injuries/physical issues, recent form, motivation, mental state, surface suitability, home advantage, travel fatigue, reported withdrawals or issues.

If no news is available for a player, treat as neutral.

Return ONLY this JSON (no markdown, no extra text):
{{
  "prob_player1": <float 0.0-1.0>,
  "sentiment_p1": "positive" | "negative" | "neutral",
  "sentiment_p2": "positive" | "negative" | "neutral",
  "summary": "<one concise sentence with the key sentiment insight>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "confidence": "low" | "medium" | "high"
}}"""


def analyze_with_groq(player1: str, player2: str,
                      news1: dict, news2: dict,
                      tournament: str, surface: str,
                      groq_key: str) -> dict | None:
    """Chiama Groq API per analisi sentiment strutturata. Ritorna dict o None."""
    prompt  = _build_prompt(player1, player2, news1, news2, tournament, surface)
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens":  350,
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=headers,
                             json=payload, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{[\s\S]*\}", content)
        return json.loads(m.group() if m else content)
    except Exception:
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def get_match_sentiment(player1: str, player2: str,
                        tournament: str, surface: str,
                        groq_key: str) -> dict | None:
    """
    Pipeline completa: fetch notizie → analisi Groq → risultato strutturato.

    Returns dict con chiavi:
      prob_player1, sentiment_p1, sentiment_p2, summary, key_factors,
      confidence, headlines_p1, headlines_p2, n_headlines_p1, n_headlines_p2
    oppure None se tutto fallisce.
    """
    news1 = fetch_all_news(player1)
    news2 = fetch_all_news(player2)

    result = analyze_with_groq(
        player1, player2, news1, news2, tournament, surface, groq_key
    )
    if result is None:
        return None

    # Flatten headlines per display nella UI
    result["headlines_p1"]    = [h for items in news1.values() for h in items]
    result["headlines_p2"]    = [h for items in news2.values() for h in items]
    result["n_headlines_p1"]  = count_headlines(news1)
    result["n_headlines_p2"]  = count_headlines(news2)
    result["sources_p1"]      = {k: v for k, v in news1.items() if v}
    result["sources_p2"]      = {k: v for k, v in news2.items() if v}
    return result
