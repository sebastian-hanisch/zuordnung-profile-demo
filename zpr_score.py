"""Nenn-Spannweite und Normierung, Kendalls tau-b. Nur numpy; eigene Kopie der kleinen Helfer (kein Import über Repos).
Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import CRITERIA


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def directions(m: int) -> np.ndarray:
    return np.array([CRITERIA[k][1] for k in range(m)], dtype=float)


def normalize(S: np.ndarray, m: int | None = None) -> np.ndarray:
    """Auf die feste Nenn-Spannweite: (Wert - Untergrenze) / (Obergrenze - Untergrenze); unabhängig von der Menge."""
    S = np.asarray(S, dtype=float)
    lo, hi = nominal_bounds(S.shape[1] if m is None else m)
    return (S - lo) / (hi - lo)


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)
