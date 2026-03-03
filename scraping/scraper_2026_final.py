import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = "atp_torneos_2026_final.csv"
ARCHIVO_SALIDA = "atp_matches_2026_indetectable.csv"
ANIO = 2026

print(f"🥷 INICIANDO MODO INDETECTABLE ({ANIO})...")

# 1. PREPARAR DATOS
try:
    df = pd.read_csv(ARCHIVO_ENTRADA)
    urls = []
    names = []
    print("🔗 Procesando enlaces...")
    for link in df['Link_Resultados']:
        parts = link.strip().split('/')
        try:
            if 'tournaments' in parts:
                idx = parts.index('tournaments')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            elif 'archive' in parts:
                idx = parts.index('archive')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            else:
                continue # Saltamos links raros
            
            # Usamos EN (Inglés) para mayor estabilidad
            new_link = f"https://www.atptour.com/en/scores/archive/{nombre}/{id_t}/{ANIO}/results"
            urls.append(new_link)
            names.append(nombre)
        except:
            pass
except Exception as e:
    print(f"❌ Error leyendo CSV: {e}")
    exit()

# 2. INICIAR NAVEGADOR INDETECTABLE
# OJO: Esto abre una ventana de Chrome que NO dice "Chrome está siendo controlado..."
options = uc.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument("--headless") # NO uses headless con Cloudflare

print("🚀 Lanzando Chrome parcheado (puede tardar unos segundos)...")
driver = uc.Chrome(options=options, version_main=145)

all_matches = []

# 3. BUCLE DE TORNEOS
for i, url in enumerate(urls):
    torneo = names[i]
    print(f"\n🌍 [{i+1}/{len(urls)}] {torneo}")
    print(f"   Link: {url}")
    
    try:
        driver.get(url)
        
        # TIEMPO DE SEGURIDAD PARA CLOUDFLARE
        # Si sale el challenge, undetected-chromedriver suele pasarlo solo,
        # o te deja hacer clic sin bloquearte.
        time.sleep(5) 
        
        # CHEQUEO DE BLOQUEO MANUAL
        if "just a moment" in driver.title.lower():
            print("🛑 Cloudflare detectado. Tienes 10 segundos para hacer clic si es necesario...")
            time.sleep(10)

        # SCROLL PARA CARGAR PARTIDOS
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # PARSEO
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        matches = soup.find_all('div', class_='match')
        
        if not matches:
            print("   ⚠️ 0 partidos. Se asume que llegamos a torneos futuros.")
            print("   🛑 ¡Deteniendo el scraper para ahorrar tiempo!")
            break  # Esto rompe el bucle y pasa directo a guardar
            
        print(f"   ✅ ¡{len(matches)} PARTIDOS!")
        
        count = 0
        for m in matches:
            try:
                # Extracción rápida
                round_txt = m.find('div', class_='match-header').get_text(strip=True).split("-")[0]
                
                players = m.find_all('div', class_='stats-item')
                if len(players) < 2: continue
                
                p1 = players[0].find('div', class_='name').get_text(strip=True).split("(")[0].strip()
                p2 = players[1].find('div', class_='name').get_text(strip=True).split("(")[0].strip()
                
                if players[0].find('div', class_='winner'):
                    winner, loser = p1, p2
                    w_node, l_node = players[0], players[1]
                else:
                    winner, loser = p2, p1
                    w_node, l_node = players[1], players[0]
                
                # Score simple
                score_parts = []
                sw = w_node.select('.score-item span')
                sl = l_node.select('.score-item span')
                for k in range(min(len(sw), len(sl))):
                    v1, v2 = sw[k].get_text(strip=True), sl[k].get_text(strip=True)
                    if v1 and v2: score_parts.append(f"{v1}-{v2}")
                
                all_matches.append({
                    'tourney_id': f"{ANIO}-{torneo}-{i}",
                    'tourney_name': torneo,
                    'surface': 'Hard',
                    'winner_name': winner,
                    'loser_name': loser,
                    'score': " ".join(score_parts),
                    'round': round_txt,
                    'minutes': 100
                })
                count += 1
            except: continue
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Si Chrome crashó, recrear el navegador
        if "no such window" in str(e) or "target window" in str(e):
            print("   🔄 Chrome se cerró. Reiniciando navegador...")
            try:
                driver.quit()
            except:
                pass
            time.sleep(3)
            driver = uc.Chrome(options=options, version_main=145)
            time.sleep(2)

driver.quit()

# 4. GUARDAR
if all_matches:
    df_new = pd.DataFrame(all_matches)
    # Rellenar columnas extra para compatibilidad
    cols = ['draw_size','tourney_level','tourney_date','match_num','winner_id','winner_seed','winner_entry','winner_hand','winner_ht','winner_ioc','winner_age','loser_id','loser_seed','loser_entry','loser_hand','loser_ht','loser_ioc','loser_age','best_of','winner_rank','winner_rank_points','loser_rank','loser_rank_points']
    for c in cols: df_new[c] = 0
    
    df_new.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n🎉 ¡ÉXITO! {len(df_new)} partidos guardados en {ARCHIVO_SALIDA}")
else:
    print("\n❌ No se bajaron datos.")