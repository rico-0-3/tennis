"""
4_🔄_Aggiorna_Dati.py  —  Pagina Streamlit per aggiornare i dati
================================================================
Lancia la pipeline di aggiornamento tramite GitHub Actions,
direttamente dal sito web con un bottone.
"""

import streamlit as st
import requests
import time
from datetime import datetime, timezone

st.set_page_config(page_title="Aggiorna Dati", page_icon="🔄", layout="wide")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🔄 Aggiorna Dati")
st.markdown("""
Questa pagina permette di **aggiornare i dati** del Tennis Predictor direttamente dal sito,
senza dover accendere il PC. L'aggiornamento viene eseguito su **GitHub Actions**.
""")

st.divider()

# ─── Protezione con password ─────────────────────────────────────────────────
ADMIN_PASSWORD = "Tennis2026"

if not ADMIN_PASSWORD:
    st.error("⚠️ **ADMIN_PASSWORD** non configurata nei Secrets di Streamlit Cloud.")
    st.stop()

if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

if not st.session_state.admin_auth:
    st.markdown("### 🔐 Accesso riservato")
    pwd = st.text_input("Inserisci la password amministratore:", type="password")
    if st.button("🔓 Accedi", use_container_width=True):
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.error("❌ Password errata.")
    st.stop()

# ─── Configurazione GitHub ────────────────────────────────────────────────────
# Il token va messo nei Secrets di Streamlit Cloud come GITHUB_TOKEN
# Il repo va configurato qui sotto

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO  = st.secrets.get("GITHUB_REPO", "")   # formato: "utente/nome-repo"
WORKFLOW_FILE = "update_pipeline.yml"

if not GITHUB_TOKEN or not GITHUB_REPO:
    st.error("""
    ⚠️ **Configurazione mancante!**

    Per usare questa pagina, devi aggiungere questi **Secrets** nelle impostazioni di Streamlit Cloud:

    1. Vai su [Streamlit Cloud](https://share.streamlit.io) → la tua app → **Settings** → **Secrets**
    2. Aggiungi:
    ```toml
    GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    GITHUB_REPO = "tuo-username/nome-repo"
    ```

    Il **GITHUB_TOKEN** si crea su GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens.
    Permessi necessari: **Actions (Read & Write)** e **Contents (Read & Write)**.
    """)
    st.stop()

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# ─── Funzioni GitHub API ─────────────────────────────────────────────────────

def avvia_workflow(inputs: dict) -> tuple[bool, str]:
    """Avvia il workflow su GitHub Actions tramite API."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    payload = {
        "ref": "main",
        "inputs": {k: str(v).lower() for k, v in inputs.items()}
    }
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=15)
        if resp.status_code == 204:
            return True, ""
        else:
            detail = resp.json() if resp.headers.get('content-type','').startswith('application/json') else resp.text
            return False, f"HTTP {resp.status_code}: {detail}"
    except Exception as e:
        return False, f"Errore di connessione: {e}"


def get_ultimo_run() -> dict | None:
    """Ottiene l'ultimo run del workflow."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
    try:
        resp = requests.get(url, headers=HEADERS, params={"per_page": 1}, timeout=10)
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            return runs[0] if runs else None
    except:
        pass
    return None


def get_run_jobs(run_id: int) -> list:
    """Ottiene i job di un run specifico."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/jobs"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("jobs", [])
    except:
        pass
    return []


def get_job_logs(job_id: int) -> str:
    """Ottiene i log di un job specifico."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/jobs/{job_id}/logs"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except:
        pass
    return ""


# ─── Sezione: Seleziona le fasi ──────────────────────────────────────────────

st.subheader("⚙️ Seleziona le fasi da eseguire")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📡 Dati")
    scraping    = st.checkbox("🌐 Scraping ATP (Ranking + Partite + Elaborazione)", value=True)
    fusione     = st.checkbox("📎 Fusione Storico", value=True)
    profili     = st.checkbox("👤 Profili Giocatori", value=True)
    bio         = st.checkbox("🧬 Bio Giocatori (DOB + altezza da tennisstats.com)", value=False)
    court_speed = st.checkbox("🏟️ Court Speed (scraping + arricchimento)", value=True)

    if bio:
        st.info("ℹ️ Lo scraping bio fa ~400 richieste HTTP. Solo se i profili sono sbagliati.")

with col2:
    st.markdown("##### 🧠 Modelli")
    modelli_base = st.checkbox("📈 Training Modelli Base (XGBoost, Ensemble, LR)", value=False)
    ann          = st.checkbox("🧠 Training ANN Testa a Testa (lento ~1-2h su CPU)", value=False)
    special_bets = st.checkbox("🎲 Training Special Bets (Ace, DF, Break)", value=False)

    if ann or special_bets:
        st.warning("⚠️ Il training esteso può richiedere molto tempo. Il runner di GitHub Actions ha un timeout di 3 ore.")

st.divider()

# ─── Bottone Avvia ────────────────────────────────────────────────────────────

if st.button("🚀 Avvia Aggiornamento", type="primary", use_container_width=True):
    inputs = {
        "esegui_scraping":    scraping,
        "esegui_fusione":     fusione,
        "esegui_profili":     profili,
        "esegui_bio":         bio,
        "esegui_court_speed": court_speed,
        "esegui_modelli":     modelli_base,
        "esegui_ann":         ann,
        "esegui_special_bets": special_bets, # Nuova variabile!
    }

    with st.spinner("📡 Invio richiesta a GitHub Actions..."):
        ok, errore = avvia_workflow(inputs)

    if ok:
        st.success("✅ **Workflow avviato con successo!** Puoi seguire il progresso qui sotto.")
        st.balloons()

        # Aspetta qualche secondo che GitHub crei il run
        time.sleep(5)

        # ─── Monitoring in tempo reale ────────────────────────────────────
        status_container = st.empty()
        progress_bar = st.progress(0, text="In attesa...")
        log_container = st.empty()

        max_poll = 360  # poll per max ~60 minuti (ogni 10s)
        for i in range(max_poll):
            run = get_ultimo_run()
            if not run:
                time.sleep(10)
                continue

            status = run.get("status", "unknown")
            conclusion = run.get("conclusion")
            run_url = run.get("html_url", "#")

            # Aggiorna lo stato
            if status == "queued":
                status_container.info(f"⏳ **In coda...** ([vedi su GitHub]({run_url}))")
                progress_bar.progress(10, text="In coda su GitHub Actions...")
            elif status == "in_progress":
                # Ottieni dettagli sui job
                jobs = get_run_jobs(run["id"])
                steps_info = ""
                if jobs:
                    job = jobs[0]
                    steps = job.get("steps", [])
                    completed = sum(1 for s in steps if s.get("status") == "completed")
                    total = len(steps)
                    if total > 0:
                        pct = min(90, int(20 + (completed / total) * 70))
                        progress_bar.progress(pct, text=f"Esecuzione... ({completed}/{total} step completati)")

                        # Mostra gli step
                        steps_md = ""
                        for s in steps:
                            name = s.get("name", "?")
                            s_status = s.get("status", "?")
                            s_conclusion = s.get("conclusion")
                            if s_conclusion == "success":
                                steps_md += f"- ✅ {name}\n"
                            elif s_conclusion == "failure":
                                steps_md += f"- ❌ {name}\n"
                            elif s_status == "in_progress":
                                steps_md += f"- ⏳ **{name}** (in corso...)\n"
                            else:
                                steps_md += f"- ⬜ {name}\n"

                        steps_info = steps_md

                status_container.warning(f"⚙️ **In esecuzione...** ([vedi su GitHub]({run_url}))")
                if steps_info:
                    log_container.markdown(f"### 📋 Progresso Step\n{steps_info}")

            elif status == "completed":
                progress_bar.progress(100, text="Completato!")

                if conclusion == "success":
                    status_container.success(f"🎉 **Aggiornamento completato con successo!** ([vedi dettagli]({run_url}))")
                    log_container.markdown("""
                    ### ✅ Cosa succede ora?
                    I dati aggiornati sono stati **committati nel repo**.
                    Streamlit Cloud rileverà automaticamente le modifiche
                    e **riavvierà l'app** con i dati nuovi entro pochi minuti.

                    Per applicare subito: **ricarica la pagina** (F5).
                    """)
                elif conclusion == "failure":
                    status_container.error(f"❌ **Aggiornamento fallito!** ([vedi log su GitHub]({run_url}))")
                else:
                    status_container.warning(f"⚠️ **Completato con esito: {conclusion}** ([dettagli]({run_url}))")
                break

            time.sleep(10)
        else:
            status_container.warning("⏰ **Timeout di monitoraggio** — il workflow potrebbe essere ancora in corso. Controlla su GitHub.")

    else:
        st.error(f"""
        ❌ **Impossibile avviare il workflow.**

        **Errore:** `{errore}`

        **Debug:**
        - Repo: `{GITHUB_REPO}`
        - Token: `{GITHUB_TOKEN[:8]}...` ({len(GITHUB_TOKEN)} caratteri)
        - Workflow: `{WORKFLOW_FILE}`

        Possibili cause:
        - Token GitHub non valido o scaduto
        - Il token non ha il permesso `repo` + `workflow` (serve Classic Token)
        - Nome repo errato nei Secrets (formato: `utente/nome-repo`)
        - Il file workflow non esiste nel branch `main`
        """)

# ─── Sezione: Ultimo aggiornamento ───────────────────────────────────────────

st.divider()
st.subheader("📜 Ultimo Aggiornamento")

run = get_ultimo_run()
if run:
    status = run.get("status", "unknown")
    conclusion = run.get("conclusion")
    created = run.get("created_at", "?")
    run_url = run.get("html_url", "#")

    # Formatta la data
    try:
        dt = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        data_str = dt.strftime("%d/%m/%Y %H:%M UTC")
    except:
        data_str = created

    # Icona stato
    if status == "completed" and conclusion == "success":
        icona = "✅"
    elif status == "completed" and conclusion == "failure":
        icona = "❌"
    elif status in ("queued", "in_progress"):
        icona = "⏳"
    else:
        icona = "❔"

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Stato", f"{icona} {conclusion or status}")
    col_info2.metric("Data", data_str)
    col_info3.markdown(f"[🔗 Vedi su GitHub]({run_url})")
else:
    st.info("Nessun aggiornamento precedente trovato.")