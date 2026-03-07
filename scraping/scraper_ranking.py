import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import sys
import os

URL_RANKING = "https://www.atptour.com/en/rankings/singles?rankRange=1-400"
ARCHIVO_SALIDA = "ranking_2026.csv"

MAX_INTENTOS = 3

IS_CI = os.environ.get("CI", "").lower() == "true"

def crear_driver():
    """Crea un driver UC con retry sul patcher."""
    for intento in range(3):
        try:
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            if IS_CI:
                options.add_argument("--headless=new")
            driver = uc.Chrome(options=options)
            return driver
        except Exception as e:
            print(f"   ⚠️  Intento {intento+1} di creare Chrome fallito: {e}")
            time.sleep(2)
    return None

print("🏆 RE-ESCANEO DE RANKING (Modo Rastreador por Nombre)...")

data_ranking = []

for intento in range(1, MAX_INTENTOS + 1):
    print(f"\n--- Tentativo {intento}/{MAX_INTENTOS} ---")
    
    driver = crear_driver()
    if driver is None:
        print("   ❌ Impossibile avviare Chrome")
        continue
    
    try:
        driver.get(URL_RANKING)
        print("⏳ Esperando carga de la página...")
        time.sleep(5)
    
        # --- 🛡️ EL ASESINO DE COOKIES ---
        try:
            btn_cookies = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            btn_cookies.click()
            time.sleep(1)
        except:
            driver.execute_script("""
                var cookieBanner = document.getElementById('onetrust-consent-sdk');
                if(cookieBanner) cookieBanner.remove();
                document.body.style.overflow = 'scroll';
            """)

        # --- 📜 AUTO-SCROLL ---
        print("📜 Deslizando para despertar a la tabla...")
        for _ in range(6):
            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(1.5)
            
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tabla = soup.find('table', class_='mega-table')
        
        if tabla:
            filas = tabla.find('tbody').find_all('tr')
            print(f"✅ Encontrados {len(filas)} jugadores. Extrayendo datos...")
            
            for index, row in enumerate(filas):
                try:
                    cols = row.find_all('td')
                    
                    if index == 0:
                        textos_celdas = [td.get_text(strip=True) for td in cols]
                        print(f"\n🔍 [DEBUG FILA 1] Esto es lo que lee el robot en cada celda:\n{textos_celdas}\n")

                    rank_limpio = re.sub(r'\D', '', cols[0].get_text(strip=True))
                    rank = int(rank_limpio) if rank_limpio else 999

                    index_nombre = -1
                    full_link = ""
                    nombre = ""
                    
                    for i, td in enumerate(cols):
                        link_tag = td.find('a', href=True)
                        if link_tag:
                            full_link = f"https://www.atptour.com{link_tag['href']}"
                            nombre = link_tag.get_text(strip=True)
                            index_nombre = i
                            break
                    
                    if index_nombre == -1: continue

                    puntos = 0
                    for td in cols[index_nombre+1 : index_nombre+5]:
                        texto_celda = td.get_text(strip=True).replace(',', '')
                        if texto_celda.isdigit():
                            valor = int(texto_celda)
                            if valor > puntos:
                                puntos = valor

                    data_ranking.append({
                        'player': nombre,
                        'rank': rank,
                        'points': puntos,
                        'url_perfil': full_link
                    })
                    
                    if index < 3:
                        print(f"   -> [OK] {nombre} | Rank: {rank} | Puntos: {puntos}")

                except Exception as e:
                    continue
                
        # --- GUARDADO ---
        if len(data_ranking) > 0:
            pd.DataFrame(data_ranking).to_csv(ARCHIVO_SALIDA, index=False)
            print(f"\n🎉 ¡Ganamos! Archivo '{ARCHIVO_SALIDA}' generado correctamente.")
            break  # Successo, esci dal loop di retry
        else:
            print("\n⚠️ Nessun dato estratto in questo tentativo.")

    except Exception as e:
        print(f"❌ Error General: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass

if len(data_ranking) == 0:
    print("\n❌ CRÍTICO: Dopo tutti i tentativi, non ho dati. Uscita con errore.")
    sys.exit(1)