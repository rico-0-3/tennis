"""
prediccion/glicko2.py
=====================
Implementazione Glicko-2 per tennis (single-match update).

Riferimento: Glickman (2012) "Example of the Glicko-2 system"
http://www.glicko.net/glicko/glicko2.pdf
"""
import math

GLICKO_MU_0    = 1500.0   # rating iniziale
GLICKO_PHI_0   = 350.0    # rating deviation iniziale (massima incertezza)
GLICKO_SIGMA_0 = 0.06     # volatility iniziale
GLICKO_SCALE   = 173.7178 # fattore di scala Glicko-2


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _to_scale(r: float, rd: float):
    """Converte da scala Glicko-1 (r, RD) a scala Glicko-2 (mu, phi)."""
    return (r - 1500.0) / GLICKO_SCALE, rd / GLICKO_SCALE


def _from_scale(mu: float, phi: float):
    """Riconverte da scala Glicko-2 a scala Glicko-1."""
    return GLICKO_SCALE * mu + 1500.0, min(350.0, GLICKO_SCALE * phi)


def update(
    winner_r: float, winner_rd: float, winner_sigma: float,
    loser_r: float,  loser_rd: float,  loser_sigma: float,
) -> tuple:
    """
    Aggiorna i rating Glicko-2 per un singolo match.

    Returns:
        ((r_w, rd_w, sigma_w), (r_l, rd_l, sigma_l))
    """
    mu_w, phi_w = _to_scale(winner_r, winner_rd)
    mu_l, phi_l = _to_scale(loser_r,  loser_rd)

    # Aggiornamento vincitore (outcome = 1.0)
    g_l  = _g(phi_l)
    e_w  = _E(mu_w, mu_l, phi_l)
    v_w  = 1.0 / (g_l**2 * e_w * (1.0 - e_w))
    phi_star_w = math.sqrt(phi_w**2 + winner_sigma**2)
    phi_w_new  = 1.0 / math.sqrt(1.0 / phi_star_w**2 + 1.0 / v_w)
    mu_w_new   = mu_w + phi_w_new**2 * g_l * (1.0 - e_w)
    r_w_new, rd_w_new = _from_scale(mu_w_new, phi_w_new)

    # Aggiornamento perdente (outcome = 0.0)
    g_w  = _g(phi_w)
    e_l  = _E(mu_l, mu_w, phi_w)
    v_l  = 1.0 / (g_w**2 * e_l * (1.0 - e_l))
    phi_star_l = math.sqrt(phi_l**2 + loser_sigma**2)
    phi_l_new  = 1.0 / math.sqrt(1.0 / phi_star_l**2 + 1.0 / v_l)
    mu_l_new   = mu_l + phi_l_new**2 * g_w * (0.0 - e_l)
    r_l_new, rd_l_new = _from_scale(mu_l_new, phi_l_new)

    return (r_w_new, rd_w_new, winner_sigma), (r_l_new, rd_l_new, loser_sigma)


class Glicko2Store:
    """
    Gestisce i rating Glicko-2 per un insieme di giocatori.
    Usato separatamente per ogni superficie (Hard/Clay/Grass) e per overall.
    """

    def __init__(self):
        self._ratings: dict = {}

    def get(self, player: str) -> tuple:
        """Restituisce (r, rd, sigma). Default se giocatore sconosciuto."""
        return self._ratings.get(player, (GLICKO_MU_0, GLICKO_PHI_0, GLICKO_SIGMA_0))

    def update_match(self, winner: str, loser: str) -> None:
        """Aggiorna i rating di vincitore e perdente dopo un match."""
        wr, wrd, wsig = self.get(winner)
        lr, lrd, lsig = self.get(loser)
        (wr_new, wrd_new, wsig_new), (lr_new, lrd_new, lsig_new) = update(
            wr, wrd, wsig, lr, lrd, lsig
        )
        self._ratings[winner] = (wr_new, wrd_new, wsig_new)
        self._ratings[loser]  = (lr_new, lrd_new, lsig_new)

    def to_dict(self) -> dict:
        return dict(self._ratings)

    @classmethod
    def from_dict(cls, d: dict) -> "Glicko2Store":
        store = cls()
        store._ratings = dict(d)
        return store
