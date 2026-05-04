"""Unit test per prediccion/glicko2.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "prediccion"))
from glicko2 import update, Glicko2Store, GLICKO_MU_0, GLICKO_PHI_0, GLICKO_SIGMA_0


def test_winner_rating_increases():
    """Il vincitore deve aumentare il proprio rating."""
    r0, rd0, sig0 = 1500.0, 200.0, 0.06
    r1, rd1, sig1 = 1500.0, 200.0, 0.06
    (rw, rdw, _), (rl, rdl, _) = update(r0, rd0, sig0, r1, rd1, sig1)
    assert rw > r0, "rating vincitore deve aumentare"
    assert rl < r1, "rating perdente deve diminuire"


def test_symmetric_update():
    """Due giocatori con stesso rating: cambio uguale e opposto."""
    r0, rd0, sig0 = 1500.0, 200.0, 0.06
    (rw, rdw, _), (rl, rdl, _) = update(r0, rd0, sig0, r0, rd0, sig0)
    delta_w = rw - r0
    delta_l = rl - r0
    assert abs(delta_w + delta_l) < 1e-6, "delta simmetrico"
    assert delta_w > 0


def test_strong_vs_weak_small_update():
    """Battere un giocatore molto più debole → piccolo guadagno di rating."""
    strong = (2000.0, 50.0, 0.06)  # forte, RD bassa
    weak   = (1200.0, 50.0, 0.06)
    (rw, _, _), _ = update(*strong, *weak)
    delta_strong = rw - strong[0]
    assert delta_strong < 10, f"guadagno atteso < 10, got {delta_strong:.2f}"


def test_rd_decreases_after_match():
    """La RD deve diminuire dopo un match (più certezza)."""
    r0, rd0, sig0 = 1500.0, 350.0, 0.06
    (_, rdw, _), (_, rdl, _) = update(r0, rd0, sig0, r0, rd0, sig0)
    assert rdw < rd0, "RD vincitore deve diminuire"
    assert rdl < rd0, "RD perdente deve diminuire"


def test_rd_capped_at_350():
    """La RD non può superare 350 (valore iniziale massimo)."""
    (rw, rdw, _), (rl, rdl, _) = update(1500, 350, 0.06, 1500, 350, 0.06)
    assert rdw <= 350.0
    assert rdl <= 350.0


def test_store_initial_values():
    store = Glicko2Store()
    r, rd, sig = store.get("Novak Djokovic")
    assert r == GLICKO_MU_0
    assert rd == GLICKO_PHI_0
    assert sig == GLICKO_SIGMA_0


def test_store_update_match():
    store = Glicko2Store()
    store.update_match("Sinner", "Djokovic")
    r_s, _, _ = store.get("Sinner")
    r_d, _, _ = store.get("Djokovic")
    assert r_s > GLICKO_MU_0
    assert r_d < GLICKO_MU_0


def test_store_serialization():
    store = Glicko2Store()
    store.update_match("Alcaraz", "Medvedev")
    d = store.to_dict()
    store2 = Glicko2Store.from_dict(d)
    assert store2.get("Alcaraz") == store.get("Alcaraz")
    assert store2.get("Medvedev") == store.get("Medvedev")


def test_store_independent_surfaces():
    """Store separati per superficie sono indipendenti."""
    hard_store = Glicko2Store()
    clay_store = Glicko2Store()
    hard_store.update_match("Federer", "Nadal")  # Federer vince su Hard
    clay_store.update_match("Nadal", "Federer")  # Nadal vince su Clay
    r_fed_hard, _, _ = hard_store.get("Federer")
    r_fed_clay, _, _ = clay_store.get("Federer")
    assert r_fed_hard > GLICKO_MU_0
    assert r_fed_clay < GLICKO_MU_0
