"""
test_minuti.py — Verifica che estrai_minuti() funzioni correttamente.

Due sezioni:
  1. Unit test su HTML mock con pattern reali ATP
  2. Fetch reale (requests) per vedere se ATP espone duration nell'HTML
     (Nota: ATP blocca requests senza Selenium → risultato atteso = 403)

Uso:
    cd scraping/
    python test_minuti.py
"""

import re
import sys
import requests
from bs4 import BeautifulSoup


# ── Copia locale di estrai_minuti ─────────────────────────────────────────────
def estrai_minuti(match_el) -> int | None:
    for klass in ['duration', 'match-duration', 'time', 'match-time']:
        el = match_el.find(class_=lambda c: c and klass in c.lower() if c else False)
        if el:
            txt = el.get_text(strip=True)
            m = re.search(r'(\d+)\s*h[^\d]*(\d+)', txt)
            if m: return int(m.group(1)) * 60 + int(m.group(2))
            m = re.search(r'(\d+)\s*min', txt, re.IGNORECASE)
            if m: return int(m.group(1))
            m = re.search(r'(\d+):(\d{2})', txt)
            if m: return int(m.group(1)) * 60 + int(m.group(2))
    full_text = match_el.get_text(' ', strip=True)
    m = re.search(r'(\d+)\s*h[^\d]*(\d+)\s*m', full_text, re.IGNORECASE)
    if m: return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r'(\d{2,3})\s*min', full_text, re.IGNORECASE)
    if m: return int(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIT TEST — HTML MOCK
# ═══════════════════════════════════════════════════════════════════════════════

cases = [
    # (descrizione, html_del_match, minuti_attesi_o_None)
    (
        "class='match-duration', testo '1h 23m'",
        '<div class="match"><div class="match-duration">1h 23m</div></div>',
        83,
    ),
    (
        "class='duration', testo '2h 05m'",
        '<div class="match"><span class="duration">2h 05m</span></div>',
        125,
    ),
    (
        "class='match-time', testo '95 min'",
        '<div class="match"><span class="match-time">95 min</span></div>',
        95,
    ),
    (
        "class='time', testo '1:47'",
        '<div class="match"><div class="time">1:47</div></div>',
        107,
    ),
    (
        "testo libero '2h 11m' senza classe specifica",
        '<div class="match"><p>Duration: 2h 11m</p></div>',
        131,
    ),
    (
        "testo libero '110 min'",
        '<div class="match"><p>Match time: 110 min</p></div>',
        110,
    ),
    (
        "nessuna durata presente",
        '<div class="match"><p>Sinner def. Zverev 6-3 6-4</p></div>',
        None,
    ),
]

print("=" * 60)
print("UNIT TEST — estrai_minuti() su HTML mock")
print("=" * 60)

passed = 0
for desc, html, atteso in cases:
    soup = BeautifulSoup(html, "html.parser")
    el = soup.find("div", class_="match")
    risultato = estrai_minuti(el)
    ok = risultato == atteso
    stato = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  [{stato}] {desc}")
    if not ok:
        print(f"         atteso={atteso}, ottenuto={risultato}")

print(f"\nRisultato: {passed}/{len(cases)} test passati")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FETCH REALE — verifica presenza classe 'duration' nell'HTML ATP
#    (ATP blocca requests → usa Selenium in produzione)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FETCH REALE — ATP Results page (Miami 2026)")
print("=" * 60)

URL = "https://www.atptour.com/en/scores/archive/miami/403/2026/results"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.atptour.com/",
}

try:
    r = requests.get(URL, headers=HEADERS, timeout=20)
    print(f"  HTTP {r.status_code} — {len(r.text):,} char ricevuti")

    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        matches = soup.find_all("div", class_="match")
        print(f"  div.match trovati: {len(matches)}")

        trovati = 0
        for m in matches:
            min_val = estrai_minuti(m)
            if min_val:
                trovati += 1
                if trovati <= 3:
                    print(f"  Durata: {min_val} min — {m.get_text(' ', strip=True)[:80]}")

        print(f"  Match con durata: {trovati}/{len(matches)}")

        # Cerca classi che contengono 'duration' ovunque nella pagina
        els = soup.find_all(class_=re.compile(r'duration', re.I))
        print(f"  Elementi class*='duration' nella pagina: {len(els)}")
        for e in els[:5]:
            print(f"    <{e.name} class='{e.get('class')}'> {e.get_text(strip=True)[:60]}")
    else:
        print(f"  ATP ha bloccato la richiesta (atteso: serve Selenium/undetected_chromedriver)")
        print(f"  La logica di estrai_minuti() e' verificata dai test mock sopra.")

except Exception as e:
    print(f"  Errore: {e}")
    print(f"  La logica di estrai_minuti() e' verificata dai test mock sopra.")
