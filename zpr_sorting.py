"""Zuordnungsregeln: Summe gegen Profile, ELECTRE TRI (pessimistisch, optimistisch), FlowSort, Quantilklassen, Profile aus den Quantilen. Nur numpy.

Alle Regeln arbeiten auf normierten Werten (Nenn-Spannweite, `zpr_score.normalize`); `Sp` sind die signierten Rohwerte der Profile, aufsteigend (k - 1 Zeilen). Rückgabe: Klasse 0 bis k - 1."""

from __future__ import annotations

import numpy as np

import zpr_constants as C
import zpr_score as sc


def class_by_sum(S: np.ndarray, w: np.ndarray, Sp: np.ndarray) -> np.ndarray:
    """Zahl der Profile, deren gewichtete Summe die der Alternative nicht übersteigt."""
    w = np.asarray(w, dtype=float)
    s = sc.normalize(S) @ w
    sp = sc.normalize(Sp) @ w
    return (s[:, None] >= sp[None, :]).sum(axis=1)


def outranks(Na: np.ndarray, Nb: np.ndarray, w: np.ndarray, lam: float, veto: float | None = None) -> bool:
    """a übertrifft b: Gewichte der Kriterien, in denen a mindestens so gut ist wie b, zusammen mindestens lam, und (mit Veto) kein Kriterium, in dem b um mehr als `veto` besser ist."""
    conc = float(((Na >= Nb) * w).sum())
    if veto is not None and bool(((Nb - Na) > veto).any()):
        return False
    return conc >= lam - 1e-12


def electre_tri(S: np.ndarray, w: np.ndarray, Sp: np.ndarray, lam: float = C.DEFAULT_LAM, veto: float | None = None, mode: str = "pess") -> np.ndarray:
    """ELECTRE TRI mit Grenzprofilen. Pessimistisch: von oben die erste Klasse, deren unteres Profil die Alternative übertrifft. Optimistisch: von unten das erste Profil,
    das die Alternative strikt übertrifft (b S a und nicht a S b); die Alternative liegt darunter."""
    if mode not in ("pess", "opt"):
        raise ValueError(f"unbekannter Modus: {mode}")
    w = np.asarray(w, dtype=float)
    Na, Nb = sc.normalize(S), sc.normalize(Sp)
    k1 = len(Sp)
    out = np.zeros(len(Na), dtype=int)
    for i in range(len(Na)):
        a_s_b = [outranks(Na[i], Nb[h], w, lam, veto) for h in range(k1)]
        if mode == "pess":
            c = 0
            for h in range(k1 - 1, -1, -1):
                if a_s_b[h]:
                    c = h + 1
                    break
        else:
            c = k1
            for h in range(k1):
                if outranks(Nb[h], Na[i], w, lam, veto) and not a_s_b[h]:
                    c = h
                    break
        out[i] = c
    return out


def net_flow_against(x: np.ndarray, R: np.ndarray, w: np.ndarray, p: float = C.FLOW_P) -> float:
    """Nettofluss von x gegen die Referenzmenge R (V-Form mit Schwelle p, Anteil der Nenn-Spannweite): Mittel über R von Summe_j w_j (P(x - r) - P(r - x))."""
    D = np.asarray(x, dtype=float)[None, :] - np.asarray(R, dtype=float)
    P = np.clip(D / p, 0.0, 1.0)
    Pr = np.clip(-D / p, 0.0, 1.0)
    return float(((P - Pr) * np.asarray(w, dtype=float)).sum(axis=-1).mean())


def flowsort(S: np.ndarray, w: np.ndarray, Sp: np.ndarray, p: float = C.FLOW_P) -> np.ndarray:
    """FlowSort in einer Lesart: Der Nettofluss der Alternative gegen die Profile wird mit den Flüssen der Profile verglichen, jeweils gegen die übrigen Profile und die Alternative;
    die Klasse ist die Zahl der Profile, deren Fluss der Fluss der Alternative erreicht."""
    Na, Nb = sc.normalize(S), sc.normalize(Sp)
    k1 = len(Nb)
    out = np.zeros(len(Na), dtype=int)
    for i in range(len(Na)):
        fa = net_flow_against(Na[i], Nb, w, p)
        fb = np.array([net_flow_against(Nb[h], np.vstack([np.delete(Nb, h, axis=0), Na[i]]), w, p) for h in range(k1)])
        out[i] = int((fa >= fb).sum())
    return out


def quantile_classes(S: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Klassen als Rangquantile der gewichteten Summe innerhalb der Menge (k gleich große Klassen; relativ)."""
    s = sc.normalize(S) @ np.asarray(w, dtype=float)
    ranks = np.argsort(np.argsort(s, kind="stable"), kind="stable")
    return (ranks * k) // len(S)                                                  # Rang <= n - 1, also Klasse <= k - 1


def data_profiles(S: np.ndarray, k: int) -> np.ndarray:
    """Profile aus der Menge: je Kriterium die k - 1 gleich verteilten Quantile (Rohwerte)."""
    qs = [(h + 1) / k for h in range(k - 1)]
    return np.quantile(np.asarray(S, dtype=float), qs, axis=0)


def classify(method: str, S: np.ndarray, w: np.ndarray, Sp: np.ndarray, k: int, lam: float = C.DEFAULT_LAM, veto: float | None = None) -> np.ndarray:
    """Klassen nach dem Verfahren `method` (feste Profile `Sp`; Quantilklassen und Profile aus den Quantilen ignorieren `Sp`)."""
    if method == "summe":
        return class_by_sum(S, w, Sp)
    if method == "tri_pess":
        return electre_tri(S, w, Sp, lam, veto, "pess")
    if method == "tri_opt":
        return electre_tri(S, w, Sp, lam, veto, "opt")
    if method == "flowsort":
        return flowsort(S, w, Sp)
    if method == "quantil":
        return quantile_classes(S, w, k)
    if method == "datenprofile":
        return class_by_sum(S, w, data_profiles(S, k))
    raise ValueError(f"unbekanntes Verfahren: {method}")
