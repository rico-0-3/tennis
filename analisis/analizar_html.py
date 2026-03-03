from bs4 import BeautifulSoup

print("🕵️‍♂️ Buscando la tabla real en 'atp_source.html'...")

try:
    with open("atp_source.html", "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Buscamos TODAS las apariciones, no solo la primera
    # Usamos 'Brisbane' o 'United Cup'
    matches = soup.find_all(string=lambda text: text and ("Brisbane" in text or "United Cup" in text))

    print(f"\nEncontré el texto {len(matches)} veces. Analicemos los candidatos:\n")

    for i, match in enumerate(matches):
        print(f"--- CANDIDATO {i+1} ---")
        
        # Subimos por los padres hasta encontrar un contenedor interesante (tr, li, div con clase)
        padre = match.parent
        estructura = []
        
        # Recorremos 6 niveles hacia arriba
        elemento_actual = padre
        for _ in range(6):
            if elemento_actual:
                nombre = elemento_actual.name
                clases = elemento_actual.get('class', [])
                clases_str = ".".join(clases) if clases else ""
                
                etiqueta_fmt = f"<{nombre} class='{clases_str}'>"
                estructura.append(etiqueta_fmt)
                
                # Si encontramos un TR (fila de tabla), ¡BINGO!
                if nombre == 'tr':
                    print(f"✅ ¡EUREKA! Parece ser una fila de tabla.")
                    print(f"📌 LA CLASE QUE BUSCAMOS ES PROBABLEMENTE: {clases}")
                
                elemento_actual = elemento_actual.parent
            else:
                break
        
        # Imprimimos la jerarquía visual (de adentro hacia afuera)
        print(" -> ".join(estructura))
        print("\n")

except FileNotFoundError:
    print("❌ No encontré el archivo 'atp_source.html'.")