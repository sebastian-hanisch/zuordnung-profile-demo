"""Profile (gleichförmige Alternativen) und die wahre Klasse des Entscheiders. Nur numpy.

Ein Profil ist eine Alternative mit gleicher Güte g_h in jedem Kriterium; seine Rohwerte kommen aus dem Generator (↑ linear, ↓ logarithmisch). Die wahre Klasse einer Alternative ist
die Zahl der Profile, deren Nutzen ihr Nutzen für den Entscheider erreicht: 0 ist die unterste Klasse, k - 1 die oberste."""

from __future__ import annotations

import numpy as np

import zpr_constants as C
import zpr_score as sc
from entscheidungsmatrix_generator import to_raw, utility

FORMS = {"guete_linear": "linear", "guete_konkav": "konkav", "guete_cobb": "cobb", "untergrenze": "linear"}


def profile_levels(k: int, delta: float = 0.0) -> tuple[float, ...]:
    """Güte der k - 1 Grenzprofile, um delta verschoben (delta = 0: die Grenzen des Entscheiders)."""
    return tuple(round(g + delta, 10) for g in C.LEVELS[k])


def profile_values(levels, m: int) -> tuple[np.ndarray, np.ndarray]:
    """(signierte Rohwerte, Güten) der Profile, je Zeile ein Profil, aufsteigend."""
    G = np.array([[g] * m for g in levels], dtype=float)
    return to_raw(np.clip(G, 1e-12, 1.0), m) * sc.directions(m), G


def decider_utility(kind: str, G: np.ndarray, S: np.ndarray, w: np.ndarray, theta: float = C.THETA) -> np.ndarray:
    """Nutzen je Alternative: Güte-Formen über den Generator, Rohwerte linear auf der Nenn-Spannweite, Untergrenze: ein Kriterium unter theta gibt den niedrigsten Nutzen."""
    w = np.asarray(w, dtype=float)
    if kind == "roh_linear":
        return sc.normalize(S, S.shape[1]) @ w
    u = utility(np.asarray(G, dtype=float), w, FORMS[kind])
    if kind == "untergrenze":
        u = np.where((np.asarray(G) >= theta).all(axis=1), u, -1e9)
    return u


def true_class(kind: str, G: np.ndarray, S: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Wahre Klasse 0 bis k - 1: Zahl der Profile (an den Grenzen des Entscheiders), deren Nutzen die Alternative erreicht."""
    m = np.asarray(S).shape[1]
    Sp, Gp = profile_values(profile_levels(k), m)
    u = decider_utility(kind, G, S, w)
    up = decider_utility(kind, Gp, Sp, w)
    return (u[:, None] >= up[None, :]).sum(axis=1)
