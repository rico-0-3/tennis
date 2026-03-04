from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import joblib
import re
import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import joblib
import re
import time
import random
from datetime import date
import os

print("🎾 Scraping multi-anno (Selenium) dei dati di velocità campo ATP...")

# Range di anni da scrapare (dal 1991 al 2026 - tutto lo storico disponibile)
START_YEAR = 1991
END_YEAR = 2026
ANNO_CORRENTE = date.today().year
CSV_PATH = 'court_speed_data.csv'
PKL_PATH = 'court_speed_dict.pkl'

# ── Logica incrementale: carica dati esistenti e salta gli anni già presenti ──
all_data = []
years_to_scrape = list(range(START_YEAR, END_YEAR + 1))

if os.path.exists(CSV_PATH):
    try:
        df_esistente = pd.read_csv(CSV_PATH)
        # Rimuovi le righe DEFAULT_ dell'anno corrente e tutti i dati dell'anno corrente
        # (l'anno corrente va sempre ri-scrapato, gli altri anni sono stabili)
        df_old = df_esistente[
            (df_esistente['year'] != ANNO_CORRENTE) &
            (~df_esistente['tournament'].str.startswith('DEFAULT_', na=False))
        ].copy()
        anni_presenti = set(df_old['year'].unique())
        # Anni da scrapare = mancanti + anno corrente
        years_to_scrape = [y for y in range(START_YEAR, END_YEAR + 1)
                           if y not in anni_presenti or y == ANNO_CORRENTE]
        all_data = df_old.to_dict('records')
        print(f"   📂 CSV esistente: {len(df_old)} record, {len(anni_presenti)} anni già presenti")
        print(f"   ➡️  Da scrapare: {years_to_scrape}")
    except Exception as e:
        print(f"   ⚠️  Impossibile leggere il CSV ({e}) — riscarico tutto")
else:
    print(f"   📂 Nessun CSV precedente — scarico tutto ({START_YEAR}–{END_YEAR})")
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=chrome_options)
print("   ✅ Browser Chrome avviato in modalità headless")

all_data = []
failed_years = []

for year in years_to_scrape:
    url = f"https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year={year}"
    print(f"   -> Scaricando {year}...", end=" ", flush=True)
    
    try:
        driver.get(url)
        time.sleep(2)
        
        # Aspetta che la tabella dati sia caricata
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "surface-speed"))
            )
            data_table = driver.find_element(By.ID, "surface-speed")
        except:
            print(f"⚠️ Tabella non trovata")
            failed_years.append(year)
            continue
        
        rows = data_table.find_elements(By.TAG_NAME, "tr")
        
        year_count = 0
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 5:  # Salta header (th) e righe vuote
                continue
            
            try:
                tournament = cols[1].text.strip()   # Nome torneo
                surface = cols[2].text.strip()       # Superficie
                ace_text = cols[3].text.strip()      # Ace %
                speed_text = cols[4].text.strip()    # Speed
                
                # Parse ace%
                ace_pct = float(ace_text.replace('%', '')) if ace_text else 0
                
                # Parse speed
                try:
                    speed = float(speed_text) if speed_text else 0
                except ValueError:
                    speed = 0
                
                if tournament and ace_pct > 0:
                    all_data.append({
                        'year': year,
                        'tournament': tournament,
                        'surface': surface,
                        'ace_pct': ace_pct,
                        'speed': speed
                    })
                    year_count += 1
            except Exception:
                continue
        
        print(f"✅ {year_count} tornei")
        
        # Pausa casuale tra 1.5 e 3.5 secondi per sembrare umano
        time.sleep(random.uniform(1.5, 3.5))
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        failed_years.append(year)
        time.sleep(2)
        continue

# Chiudi il browser
driver.quit()
print(f"\n🌐 Browser chiuso.")

# Riprova gli anni falliti (una volta)
if failed_years:
    print(f"\n🔄 Riprovo {len(failed_years)} anni falliti: {failed_years}")
    driver2 = webdriver.Chrome(options=chrome_options)
    
    for year in failed_years:
        url = f"https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year={year}"
        print(f"   -> Retry {year}...", end=" ", flush=True)
        
        try:
            time.sleep(random.uniform(3, 5))
            driver2.get(url)
            time.sleep(3)
            
            try:
                WebDriverWait(driver2, 20).until(
                    EC.presence_of_element_located((By.ID, "surface-speed"))
                )
                data_table = driver2.find_element(By.ID, "surface-speed")
            except:
                print(f"⚠️ Tabella non trovata")
                continue
            
            rows = data_table.find_elements(By.TAG_NAME, "tr")
            year_count = 0
            
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 5:
                    continue
                try:
                    tournament = cols[1].text.strip()
                    surface = cols[2].text.strip()
                    ace_text = cols[3].text.strip()
                    speed_text = cols[4].text.strip()
                    ace_pct = float(ace_text.replace('%', '')) if ace_text else 0
                    try:
                        speed = float(speed_text) if speed_text else 0
                    except ValueError:
                        speed = 0
                    if tournament and ace_pct > 0:
                        all_data.append({
                            'year': year,
                            'tournament': tournament,
                            'surface': surface,
                            'ace_pct': ace_pct,
                            'speed': speed
                        })
                        year_count += 1
                except Exception:
                    continue
            
            print(f"✅ {year_count} tornei")
        except Exception as e:
            print(f"❌ Ancora errore: {e}")
    
    driver2.quit()

print(f"\n📥 Totale dati scaricati da internet: {len(all_data)} record")

# Calcola valori di default dinamici per superficie (basati sulla media dei dati reali)
if all_data:
    df_temp = pd.DataFrame(all_data)
    default_values = {}
    
    for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
        surface_data = df_temp[df_temp['surface'] == surface]
        if len(surface_data) > 0:
            default_values[surface] = {
                'ace_pct': round(surface_data['ace_pct'].mean(), 1),
                'speed': round(surface_data['speed'].mean(), 2)
            }
        else:
            # Fallback se non ci sono dati per quella superficie
            fallback = {
                'Hard': {'ace_pct': 11.5, 'speed': 1.10},
                'Clay': {'ace_pct': 6.5, 'speed': 0.65},
                'Grass': {'ace_pct': 12.5, 'speed': 1.15},
                'Carpet': {'ace_pct': 13.0, 'speed': 1.25}
            }
            default_values[surface] = fallback[surface]
    
    print("\n📊 Valori di default calcolati (media per superficie):")
    for surface, vals in default_values.items():
        print(f"   {surface:8s}: Ace={vals['ace_pct']:5.1f}%  Speed={vals['speed']:5.2f}")
    
    # Aggiungi i default per tutti gli anni (sovrascrive quelli già esistenti)
    anni_nel_csv = set(r['year'] for r in all_data)
    for year in range(START_YEAR, END_YEAR + 1):
        if year in anni_nel_csv:
            for surface, vals in default_values.items():
                all_data.append({
                    'year': year,
                    'tournament': f'DEFAULT_{surface}',
                    'surface': surface,
                    'ace_pct': vals['ace_pct'],
                    'speed': vals['speed']
                })
else:
    # Se non ci sono dati scaricati, usa valori fissi ragionevoli
    print("\n⚠️ Nessun dato scaricato, uso valori di default fissi")
    for year in range(START_YEAR, END_YEAR + 1):
        all_data.extend([
            {'year': year, 'tournament': 'DEFAULT_Hard', 'surface': 'Hard', 'ace_pct': 11.5, 'speed': 1.10},
            {'year': year, 'tournament': 'DEFAULT_Clay', 'surface': 'Clay', 'ace_pct': 6.5, 'speed': 0.65},
            {'year': year, 'tournament': 'DEFAULT_Grass', 'surface': 'Grass', 'ace_pct': 12.5, 'speed': 1.15},
            {'year': year, 'tournament': 'DEFAULT_Carpet', 'surface': 'Carpet', 'ace_pct': 13.0, 'speed': 1.25},
        ])

df = pd.DataFrame(all_data)

# Crea un dizionario con chiave (tournament, year)
court_dict = {}
for _, row in df.iterrows():
    # Normalizza il nome del torneo
    key_name = row['tournament'].lower().strip()
    key_name = re.sub(r'[^a-z0-9\s]', '', key_name)
    key_name = re.sub(r'\s+', ' ', key_name)
    
    # Chiave composta: (nome_normalizzato, anno)
    key = (key_name, row['year'])
    
    court_dict[key] = {
        'ace_pct': row['ace_pct'],
        'speed': row['speed'],
        'surface': row['surface']
    }

# Salva CSV e dizionario
df.to_csv('court_speed_data.csv', index=False)
joblib.dump(court_dict, 'court_speed_dict.pkl')

print(f"\n✅ Dati estratti: {len(all_data)} record (tornei × anni)")
print(f"📊 Statistiche globali:")
print(f"   - Ace% medio: {df['ace_pct'].mean():.2f}%")
print(f"   - Speed medio: {df['speed'].mean():.2f}")
print(f"   - Range Ace%: {df['ace_pct'].min():.1f}% - {df['ace_pct'].max():.1f}%")
print(f"   - Range Speed: {df['speed'].min():.2f} - {df['speed'].max():.2f}")
print(f"   - Anni coperti: {df['year'].min()} - {df['year'].max()}")
print(f"   - Tornei unici: {df['tournament'].nunique()}")
print("\n📁 File salvati:")
print("   - court_speed_data.csv")
print("   - court_speed_dict.pkl")

# Mostra alcuni esempi
print("\n🔍 Esempi di dati (ultimi anni):")
print(df[df['year'] >= 2024].head(15).to_string(index=False))
