# Design: Predizioni Automatiche Prossime Partite

**Data:** 2026-04-08  
**Stato:** Approvato

---

## Obiettivo

Aggiungere alla pipeline di aggiornamento dati il calcolo automatico delle predizioni per le partite ATP della settimana corrente. I risultati vengono esposti in una nuova pagina Streamlit (`Torneos.py`) filtrabili per livello torneo, con prediction bar, %, quota ideale e H2H.

---

## Componenti

### 1. `scraping/scraper_proximas_partidas.py` (nuovo)

- Fonte dati: SofaScore API interna (`https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{YYYY-MM-DD}`) — JSON puro, nessun Selenium.
- Scarica i prossimi 7 giorni di partite ATP (filtra per categoria ATP: tour + challenger).
- Per ogni partita estrae: `p1`, `p2`, `torneo`, `livello` (Grand Slam / Masters 1000 / ATP 500 / ATP 250 / Challenger), `superficie`, `turno`, `data`.
- Normalizza i nomi giocatori (rimozione accenti, formato "Cognome Nome" → "Nome Cognome") per matchare `perfiles_jugadores.pkl`.
- Salva in `scraping/proximas_partidas.json`.
- Gestione errori: se API non risponde, salva lista vuota con timestamp e continua senza bloccare la pipeline.

### 2. `prediccion/predecir_proximas.py` (nuovo)

- Standalone script (no Streamlit), eseguito dopo `generar_perfiles.py` e la copia dei pkl.
- Carica tutti i pkl necessari identici alla pagina predictor: `modelo_finale.pkl`, `perfiles_jugadores.pkl`, `elo_surface.pkl`, `elo_overall.pkl`, `streak_players.pkl`, `momentum_surface.pkl`, `h2h_surface.pkl`, `last_match_date.pkl`, `opp_quality.pkl`, `match_load.pkl`, `upset_hist.pkl`, `late_round_hist.pkl`, `stats_superficie_v2.pkl`, `ranking_2026.csv`.
- Legge `scraping/proximas_partidas.json`.
- Per ogni partita:
  - Se uno dei due giocatori non è in `perfiles`: salva entry con `skipped: true, motivo: "giocatore non nel database"`.
  - Altrimenti: costruisce le stesse 30 feature del predictor (`ANN_FEATURES`), esegue predizione con `modelo_finale`, calcola H2H da `historialTenis.csv`, calcola quote ideali (overround 8.1%).
- Salva `prediccion/predicciones_proximas.json`:

```json
{
  "updated_at": "2026-04-08T14:30:00",
  "partite": [
    {
      "p1": "Carlos Alcaraz",
      "p2": "Novak Djokovic",
      "torneo": "Monte Carlo Masters",
      "livello": "Masters 1000",
      "superficie": "Clay",
      "turno": "Ottavi",
      "data": "2026-04-10",
      "prob_p1": 0.63,
      "favorito": "Carlos Alcaraz",
      "confidenza": 0.63,
      "h2h_p1": 5,
      "h2h_p2": 3,
      "quota_fav": 1.42,
      "quota_sfav": 2.71,
      "skipped": false
    }
  ]
}
```

### 3. `aggiorna_tutto.py` (modificato)

Aggiunge FASE 10 in fondo al `main()`, sempre eseguita:

```python
sezione("🔟  FASE 10 — Predizioni Prossime Partite")
esegui("scraper_proximas_partidas.py", SCRAPING, "Scraping partite upcoming")
esegui("predecir_proximas.py", PREDICCION, "Predizioni batch prossime partite")
```

### 4. `run_pipeline.py` (modificato)

Aggiunge le stesse 2 chiamate nella sezione "SINCRONIZZAZIONE FINALE", sempre eseguite (non dietro flag):

```python
sezione("🔟  FASE 10 — Predizioni Prossime Partite")
esegui("scraper_proximas_partidas.py", SCRAPING, "Scraping partite upcoming")
esegui("predecir_proximas.py", PREDICCION, "Predizioni batch prossime partite")
```

### 5. `pages/Torneos.py` (riscritto)

- Legge `prediccion/predicciones_proximas.json` (path relativo alla root del progetto).
- Mostra timestamp "Aggiornato il: ..." in alto.
- Selectbox per filtrare per livello: "Tutti", "Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger".
- Per ogni partita non skippata: card con
  - Nomi giocatori + barra probabilità orizzontale (Plotly)
  - % confidenza favorito
  - Quota ideale favorito / sfavorito
  - Superficie + Turno + Data
  - H2H (es. "Alcaraz 5 – 3 Djokovic")
- Partite skippate: mostrate in fondo in un expander "Partite senza predizione (X)" con motivo.
- Se il file non esiste: messaggio "Nessuna predizione disponibile — avvia la pipeline di aggiornamento."

---

## Flusso dati

```
GitHub Actions (run_pipeline.py, sempre in fondo)
  └─ scraper_proximas_partidas.py
       → SofaScore API (requests JSON)
       → scraping/proximas_partidas.json
  └─ predecir_proximas.py
       → legge proximas_partidas.json + tutti i pkl
       → prediccion/predicciones_proximas.json
       → committato in git da "Commit & Push risultati"

Streamlit (pages/Torneos.py)
  └─ legge predicciones_proximas.json (file statico)
  └─ filtra per livello
  └─ mostra cards
```

---

## Dipendenze

Nessuna nuova libreria: `requests` già in requirements, tutto il resto già usato dal predictor.

---

## Gestione errori

- SofaScore API down → `proximas_partidas.json` vuoto → `predicciones_proximas.json` con lista vuota → pagina mostra "Nessuna partita trovata per questa settimana."
- Giocatore non nel database pkl → entry `skipped: true` → mostrato in expander separato.
- `predicciones_proximas.json` non esiste (primo run prima della pipeline) → pagina mostra messaggio appropriato.
- Errore pkl mancante → `predecir_proximas.py` logga warning e skippa la partita.
