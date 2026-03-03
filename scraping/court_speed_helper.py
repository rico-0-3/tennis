import joblib
import re
import os

# Carica il dizionario una volta
try:
    COURT_SPEED_DICT = joblib.load('court_speed_dict.pkl')
except:
    # Se non esiste, usa valori di default
    COURT_SPEED_DICT = {}
    for year in range(1991, 2027):
        COURT_SPEED_DICT[('default hard', year)] = {'ace_pct': 11.5, 'speed': 1.10, 'surface': 'Hard'}
        COURT_SPEED_DICT[('default clay', year)] = {'ace_pct': 6.5, 'speed': 0.65, 'surface': 'Clay'}
        COURT_SPEED_DICT[('default grass', year)] = {'ace_pct': 12.5, 'speed': 1.15, 'surface': 'Grass'}

def normalizza_nome_torneo(nome):
    """Normalizza il nome del torneo per matching"""
    if not nome or str(nome) == 'nan':
        return ''
    
    nome = str(nome).lower().strip()
    # Rimuovi caratteri speciali
    nome = re.sub(r'[^a-z0-9\s]', '', nome)
    # Rimuovi spazi multipli
    nome = re.sub(r'\s+', ' ', nome)
    return nome

def get_court_stats(tournament_name, surface='Hard', year=2025):
    """
    Dato il nome di un torneo, superficie e anno, restituisce ace% e speed
    
    Args:
        tournament_name (str): Nome del torneo (es. "Australian Open", "Roland Garros")
        surface (str): Superficie ('Hard', 'Clay', 'Grass')
        year (int): Anno del match (default 2025)
    
    Returns:
        tuple: (ace_pct, speed)
    """
    # Normalizza il nome del torneo
    norm_name = normalizza_nome_torneo(tournament_name)
    
    # 1. Cerca match esatto con anno
    key_exact = (norm_name, year)
    if key_exact in COURT_SPEED_DICT:
        stats = COURT_SPEED_DICT[key_exact]
        return stats['ace_pct'], stats['speed']
    
    # 2. Cerca match parziali con anno
    for (key_name, key_year), stats in COURT_SPEED_DICT.items():
        if key_year == year and (norm_name in key_name or key_name in norm_name):
            if key_name.startswith('default'):
                continue  # Skip default per ora
            return stats['ace_pct'], stats['speed']
    
    # 3. Cerca anno più vicino per questo torneo
    closest_year = None
    min_diff = float('inf')
    
    for (key_name, key_year), stats in COURT_SPEED_DICT.items():
        if norm_name in key_name or key_name in norm_name:
            if key_name.startswith('default'):
                continue
            diff = abs(key_year - year)
            if diff < min_diff:
                min_diff = diff
                closest_year = key_year
    
    if closest_year is not None:
        # Trova i dati con l'anno più vicino
        for (key_name, key_year), stats in COURT_SPEED_DICT.items():
            if key_year == closest_year and (norm_name in key_name or key_name in norm_name):
                return stats['ace_pct'], stats['speed']
    
    # 4. Usa default basato sulla superficie e anno
    default_key = (f'default {surface.lower()}', year)
    if default_key in COURT_SPEED_DICT:
        stats = COURT_SPEED_DICT[default_key]
        return stats['ace_pct'], stats['speed']
    
    # 5. Fallback finale (anno più vicino del default)
    for year_offset in range(0, 20):  # Cerca fino a 20 anni di distanza
        for delta in [0, year_offset, -year_offset]:
            test_year = year + delta
            default_key = (f'default {surface.lower()}', test_year)
            if default_key in COURT_SPEED_DICT:
                stats = COURT_SPEED_DICT[default_key]
                return stats['ace_pct'], stats['speed']
    
    # Fallback assoluto
    if surface.lower() == 'clay':
        return 6.5, 0.65
    elif surface.lower() == 'grass':
        return 12.5, 1.15
    else:  # Hard o altro
        return 11.5, 1.10

def get_court_stats_latest(tournament_name, surface='Hard'):
    """
    Per la PREDIZIONE: restituisce i dati più recenti disponibili per un torneo
    
    Args:
        tournament_name (str): Nome del torneo
        surface (str): Superficie
    
    Returns:
        tuple: (ace_pct, speed)
    """
    norm_name = normalizza_nome_torneo(tournament_name)
    
    # Cerca il torneo e prendi l'anno più recente
    latest_year = None
    latest_stats = None
    
    for (key_name, key_year), stats in COURT_SPEED_DICT.items():
        if key_name.startswith('default'):
            continue
        
        if norm_name in key_name or key_name in norm_name:
            if latest_year is None or key_year > latest_year:
                latest_year = key_year
                latest_stats = stats
    
    if latest_stats:
        return latest_stats['ace_pct'], latest_stats['speed']
    
    # Non trovato, usa default della superficie (anno più recente)
    max_year = max([y for (_, y) in COURT_SPEED_DICT.keys() if isinstance(y, int)], default=2026)
    default_key = (f'default {surface.lower()}', max_year)
    
    if default_key in COURT_SPEED_DICT:
        stats = COURT_SPEED_DICT[default_key]
        return stats['ace_pct'], stats['speed']
    
    # Fallback
    if surface.lower() == 'clay':
        return 6.5, 0.65
    elif surface.lower() == 'grass':
        return 12.5, 1.15
    else:
        return 11.5, 1.10

# Funzione di test
if __name__ == "__main__":
    print("🧪 Test della funzione get_court_stats (per training):")
    
    test_cases = [
        ("Australian Open", "Hard", 2025),
        ("Roland Garros", "Clay", 2024),
        ("Wimbledon", "Grass", 2023),
        ("US Open", "Hard", 2022),
        ("Madrid", "Clay", 2021),
        ("Indian Wells", "Hard", 2025),
        ("Torneo Sconosciuto", "Hard", 2020),
        ("", "Clay", 2025),
    ]
    
    for tourney, surf, yr in test_cases:
        ace, speed = get_court_stats(tourney, surf, yr)
        print(f"  {tourney:20s} ({surf:5s}, {yr}): Ace={ace:5.1f}%  Speed={speed:5.2f}")
    
    print("\n🔮 Test della funzione get_court_stats_latest (per predizione):")
    
    test_cases_latest = [
        ("Australian Open", "Hard"),
        ("Roland Garros", "Clay"),
        ("Wimbledon", "Grass"),
        ("Madrid Masters", "Clay"),
        ("Indian Wells Masters", "Hard"),
        ("Torneo Futuro 2030", "Hard"),
    ]
    
    for tourney, surf in test_cases_latest:
        ace, speed = get_court_stats_latest(tourney, surf)
        print(f"  {tourney:25s} ({surf:5s}): Ace={ace:5.1f}%  Speed={speed:5.2f}")
