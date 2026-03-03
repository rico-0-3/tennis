import subprocess
import time
import os
import sys  

def ejecutar_pipeline():
    # Lista ordenada de tus scripts
    scripts = [
        "scraper_ranking.py",
        "scraper_2026_final.py",
        "enriquecer_2026.py",
        "corregir_superficie_ranking.py",
        "juntar_scrapings.py",
        "fusionar_historico_final.py",
        "generar_perfiles.py", # ¡No olvides generar el .pkl al final!
        "scraper_court_speed.py",  # Scraping velocità campo (Ace%, Speed)
        "enriquecer_court_speed.py",  # Arricchire CSV con dati campo
    ]
    
    directorio_scraping = os.path.dirname(os.path.abspath(__file__))

    print("🚀 INICIANDO ACTUALIZACIÓN GLOBAL...")
    inicio = time.time()
    
    for script in scripts:
        print(f"\n" + "="*40)
        print(f"▶️ Ejecutando: {script}")
        print("="*40)
        
        try:
            # Ejecuta el script y espera a que termine
            resultado = subprocess.run([sys.executable, "-X", "utf8", script], check=True, capture_output=True, text=True,encoding='utf-8', cwd=directorio_scraping)
            print(resultado.stdout) # Imprime lo que dice el script
        except subprocess.CalledProcessError as e:
            print(f"❌ Error crítico en {script}!")
            print(e.stderr)
            return False # Detiene todo si algo falla
            
    fin = time.time()
    minutos = (fin - inicio) / 60
    print(f"\n✅ ¡PIPELINE COMPLETADO EN {minutos:.1f} MINUTOS!")
    return True

# Esto permite probarlo desde la terminal
if __name__ == "__main__":
    ejecutar_pipeline()