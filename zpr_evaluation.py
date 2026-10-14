"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import zpr_constants as C
import zpr_profiles as pf
import zpr_score as sc
import zpr_sorting as so
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    k: int = C.DEFAULT_K
    method: str = C.DEFAULT_METHOD
    lam: float = C.DEFAULT_LAM
    veto: float = C.DEFAULT_VETO
    delta: float = C.DEFAULT_DELTA
    decider: str = C.DEFAULT_DECIDER
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    S: np.ndarray              # signierte Rohwerte der Alternativen
    G: np.ndarray              # Güten
    w: np.ndarray              # Gewichte des Entscheiders
    utility: np.ndarray
    profile_S: np.ndarray      # Profile der Verfahren (um delta verschoben), signierte Rohwerte
    profile_G: np.ndarray
    true_class: np.ndarray
    predicted: np.ndarray
    other: dict                # alle Verfahren: Klassen (mit denselben Einstellungen)
    pess: np.ndarray           # ELECTRE TRI pessimistisch und optimistisch mit den Einstellungen lam / veto
    opt: np.ndarray
    changed: np.ndarray        # Klassenwechsel der alten Alternativen, wenn eine zufällige dazukommt (Maske)
    new_class: int             # Klasse der neuen Alternative nach dem gewählten Verfahren
    thresholds: np.ndarray     # Nutzen der Profile an den Grenzen des Entscheiders (unverschoben)


def _veto(v: float) -> float | None:
    return None if v <= 0 else v


def make_world(n: int, m: int, seed: int):
    """n Alternativen und eine weitere (die (n + 1)-te Zeile derselben Matrix)."""
    x = make_matrix(n + 1, m, 0.0, SplitMix64(seed))
    return x


def analyse(settings: Settings) -> Analysis:
    s = settings
    x = make_world(s.n, s.m, s.seed)
    S_all = np.asarray(x.signed, dtype=float)
    G_all = np.asarray(x.goodness, dtype=float)
    S, G, w = S_all[: s.n], G_all[: s.n], np.asarray(x.weights, dtype=float)
    u = pf.decider_utility(s.decider, G, S, w)
    tc = pf.true_class(s.decider, G, S, w, s.k)
    Sp, Gp = pf.profile_values(pf.profile_levels(s.k, s.delta), s.m)
    veto = _veto(s.veto)
    other = {k_: so.classify(k_, S, w, Sp, s.k, s.lam, veto) for k_ in C.METHODS}
    pred = other[s.method]
    new_all = so.classify(s.method, S_all, w, Sp, s.k, s.lam, veto)
    Sb, Gb = pf.profile_values(pf.profile_levels(s.k), s.m)
    return Analysis(s, S, G, w, u, Sp, Gp, tc, pred, other, other["tri_pess"], other["tri_opt"], new_all[: s.n] != pred, int(new_all[s.n]), pf.decider_utility(s.decider, Gb, Sb, w))


# ----------------------------------------------------------------------------- Hilfen für die Experimente

ACC_METHODS = ("summe", "tri_pess", "tri_opt", "tri_pess_veto", "flowsort", "quantil", "datenprofile")
ACC_LABELS = {**C.METHOD_SHORT, "tri_pess_veto": "TRI pessimistisch, Veto 0,3"}


def _classify_acc(name: str, S, w, Sp, k) -> np.ndarray:
    if name == "tri_pess_veto":
        return so.electre_tri(S, w, Sp, 0.6, 0.3, "pess")
    return so.classify(name, S, w, Sp, k, 0.6, None)


def experiment_accuracy(ks=C.K_CHOICES, deciders=C.DECIDERS, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Anteil richtig zugeordneter Alternativen und mittlerer Klassenabstand je Verfahren und Entscheider."""
    rows = []
    for k in ks:
        Sp, _ = pf.profile_values(pf.profile_levels(k), m)
        acc = {(d, nme): ([], []) for d in deciders for nme in ACC_METHODS}
        for seed in seeds:
            x = make_world(n, m, seed)
            S, G, w = np.asarray(x.signed, dtype=float)[:n], np.asarray(x.goodness, dtype=float)[:n], np.asarray(x.weights, dtype=float)
            preds = {nme: _classify_acc(nme, S, w, Sp, k) for nme in ACC_METHODS}
            for d in deciders:
                tc = pf.true_class(d, G, S, w, k)
                for nme, p in preds.items():
                    acc[(d, nme)][0].append(float((p == tc).mean()))
                    acc[(d, nme)][1].append(float(np.abs(p - tc).mean()))
        for (d, nme), (a, e) in acc.items():
            rows.append({"k": k, "decider": d, "method": nme, "acc": float(np.mean(a)), "dist": float(np.mean(e))})
    return rows


def experiment_stability(k=3, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Eine Alternative kommt dazu: Anteil der alten mit Klassenwechsel (zufällig und bei einem Ausreißer in einem Kriterium) und Matrizen mit mindestens einem Wechsel."""
    Sp, _ = pf.profile_values(pf.profile_levels(k), m)
    names = ("summe", "tri_pess", "tri_opt", "flowsort", "quantil", "datenprofile")
    chg = {nme: [] for nme in names}
    anyc = {nme: [] for nme in names}
    out = {nme: [] for nme in names}
    for seed in seeds:
        x = make_world(n, m, seed)
        S_all = np.asarray(x.signed, dtype=float)
        w = np.asarray(x.weights, dtype=float)
        S_old, S_new = S_all[:n], S_all[: n + 1]
        lo, hi = S_old.min(axis=0), S_old.max(axis=0)
        rng = SplitMix64(seed + C.OFFSET)
        kk = int(rng.uniform(1)[0] * n)
        jj = int(rng.uniform(1)[0] * m)
        z = S_old[kk].copy()
        z[jj] = hi[jj] + C.OUTLIER_SHARE * (hi[jj] - lo[jj])
        S_out = np.vstack([S_old, z])
        for nme in names:
            c0 = so.classify(nme, S_old, w, Sp, k)
            c1 = so.classify(nme, S_new, w, Sp, k)[:n]
            c2 = so.classify(nme, S_out, w, Sp, k)[:n]
            chg[nme].append(float((c0 != c1).mean()))
            anyc[nme].append(bool((c0 != c1).any()))
            out[nme].append(float((c0 != c2).mean()))
    return [{"method": nme, "changed": float(np.mean(chg[nme])), "any": float(np.mean(anyc[nme])), "outlier": float(np.mean(out[nme]))} for nme in names]


def experiment_shift(deltas=C.DELTA_CHOICES, k=3, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Die Profile liegen um delta (Güte) neben den Grenzen des Entscheiders (Güte/linear): Anteil richtig für Summe, TRI pessimistisch und FlowSort; Anteil der obersten Klasse."""
    rows = []
    for d in deltas:
        Sp, _ = pf.profile_values(pf.profile_levels(k, d), m)
        a = {"summe": [], "tri_pess": [], "flowsort": []}
        top, true_top = [], []
        for seed in seeds:
            x = make_world(n, m, seed)
            S, G, w = np.asarray(x.signed, dtype=float)[:n], np.asarray(x.goodness, dtype=float)[:n], np.asarray(x.weights, dtype=float)
            tc = pf.true_class("guete_linear", G, S, w, k)
            for nme in a:
                a[nme].append(float((so.classify(nme, S, w, Sp, k, 0.6, None) == tc).mean()))
            c = so.class_by_sum(S, w, Sp)
            top.append(float((c == k - 1).mean()))
            true_top.append(float((tc == k - 1).mean()))
        rows.append({"delta": d, **{nme: float(np.mean(v)) for nme, v in a.items()}, "top_share": float(np.mean(top)), "true_top_share": float(np.mean(true_top))})
    return rows


def experiment_resolution(ks=C.K_CHOICES, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Was die Zuordnung an Ordnung hergibt: Anteil der Paare in verschiedenen Klassen und davon richtig geordnet (Summe gegen Profile, Güte/linear)."""
    rows = []
    iu = np.triu_indices(n, 1)
    for k in ks:
        Sp, _ = pf.profile_values(pf.profile_levels(k), m)
        res, cor = [], []
        for seed in seeds:
            x = make_world(n, m, seed)
            S, G, w = np.asarray(x.signed, dtype=float)[:n], np.asarray(x.goodness, dtype=float)[:n], np.asarray(x.weights, dtype=float)
            c = so.class_by_sum(S, w, Sp)
            u = G @ w
            dc = np.sign(c[:, None] - c[None, :])[iu]
            du = np.sign(u[:, None] - u[None, :])[iu]
            mask = dc != 0
            res.append(float(mask.mean()))
            cor.append(float((dc[mask] == du[mask]).mean()) if mask.any() else 1.0)
        rows.append({"k": k, "resolved": float(np.mean(res)), "correct": float(np.mean(cor))})
    return rows


def experiment_tri(lams=C.EXP_LAMS, vetos=C.EXP_VETOS, k=3, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """ELECTRE TRI: Schwelle lam und Veto: Anteil richtig (pessimistisch, optimistisch) für den Güte/linear- und den Untergrenzen-Entscheider; pessimistisch ungleich optimistisch."""
    Sp, _ = pf.profile_values(pf.profile_levels(k), m)
    rows = []
    for lam in lams:
        for veto in vetos:
            r = {"pg": [], "og": [], "pu": [], "ou": [], "diff": []}
            for seed in seeds:
                x = make_world(n, m, seed)
                S, G, w = np.asarray(x.signed, dtype=float)[:n], np.asarray(x.goodness, dtype=float)[:n], np.asarray(x.weights, dtype=float)
                tg = pf.true_class("guete_linear", G, S, w, k)
                tu = pf.true_class("untergrenze", G, S, w, k)
                pe = so.electre_tri(S, w, Sp, lam, veto, "pess")
                op = so.electre_tri(S, w, Sp, lam, veto, "opt")
                r["pg"].append(float((pe == tg).mean()))
                r["og"].append(float((op == tg).mean()))
                r["pu"].append(float((pe == tu).mean()))
                r["ou"].append(float((op == tu).mean()))
                r["diff"].append(float((pe != op).mean()))
            rows.append({"lam": lam, "veto": veto, **{kk: float(np.mean(v)) for kk, v in r.items()}})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_accuracy(rows: list[dict]) -> tuple[str, str]:
    s = _row(rows, k=3, decider="guete_linear", method="summe")["acc"]
    t = _row(rows, k=3, decider="guete_linear", method="tri_pess")["acc"]
    f = _row(rows, k=3, decider="guete_linear", method="flowsort")["acc"]
    u_v = _row(rows, k=3, decider="untergrenze", method="tri_pess_veto")["acc"]
    u_n = _row(rows, k=3, decider="untergrenze", method="tri_pess")["acc"]
    k5 = _row(rows, k=5, decider="guete_linear", method="summe")["acc"]
    return "warning", (f"Bei 3 Klassen ordnet die Summe gegen Profile {de(100 * s, 1)} % der Alternativen eines Entscheiders mit linearem Güte-Nutzen richtig zu, FlowSort {de(100 * f, 1)} %, "
                       f"ELECTRE TRI (pessimistisch, λ = 0,6) {de(100 * t, 1)} %: Die Aggregationsregel ist eine Annahme über den Entscheider. Das Veto hilft nur beim Entscheider mit "
                       f"Untergrenze ({de(100 * u_v, 1)} gegen {de(100 * u_n, 1)} %). Bei 5 Klassen sinkt die Summe auf {de(100 * k5, 1)} %.")


def verdict_stability(rows: list[dict]) -> tuple[str, str]:
    fixed = [r for r in rows if C.METHOD_FIXED[r["method"]]]
    rel = [r for r in rows if not C.METHOD_FIXED[r["method"]]]
    return "info", (f"Gegen feste Profile wechselt keine alte Alternative die Klasse, wenn eine dazukommt (höchstens {de(100 * max(r['changed'] for r in fixed), 2)} %, "
                    f"auch nicht bei einem Ausreißer: {de(100 * max(r['outlier'] for r in fixed), 2)} %). Kommen Klassen oder Profile aus der Menge, kehrt die Abhängigkeit zurück: "
                    f"{de(100 * min(r['changed'] for r in rel), 1)} bis {de(100 * max(r['changed'] for r in rel), 1)} % der alten Alternativen wechseln, in "
                    f"{de(100 * min(r['any'] for r in rel), 0)} bis {de(100 * max(r['any'] for r in rel), 0)} % der Matrizen mindestens eine.")


def verdict_price(shift: list[dict], res: list[dict]) -> tuple[str, str]:
    at0 = _row(shift, delta=0.0)["summe"]
    lo = min(_row(shift, delta=0.05)["summe"], _row(shift, delta=-0.05)["summe"])
    far = min(_row(shift, delta=0.1)["summe"], _row(shift, delta=-0.1)["summe"])
    r3 = _row(res, k=3)
    return "warning", (f"Liegen die Profile um 0,05 Güte neben den Grenzen des Entscheiders, trifft die Summe gegen Profile bis zu {de(100 * (at0 - lo), 1)} Punkte weniger ({de(100 * lo, 1)} gegen "
                       f"{de(100 * at0, 1)} %), bei 0,10 bis zu {de(100 * (at0 - far), 1)} Punkte. Bei 3 Klassen stehen {de(100 * (1 - r3['resolved']), 1)} % der Paare in derselben Klasse "
                       f"(ungeordnet); die aufgelösten sind zu {de(100 * r3['correct'], 1)} % richtig geordnet.")


def verdict_tri(rows: list[dict]) -> tuple[str, str]:
    base = _row(rows, lam=0.5, veto=None)
    veto = _row(rows, lam=0.5, veto=0.3)
    hi = _row(rows, lam=0.8, veto=None)
    return "info", (f"Bei λ = 0,5 und ohne Veto sind pessimistische und optimistische Zuordnung gleich ({de(100 * base['diff'], 1)} % verschieden); bei λ = 0,8 weichen sie in "
                    f"{de(100 * hi['diff'], 1)} % ab und der Anteil richtig fällt auf {de(100 * hi['pg'], 1)} %. Das Veto 0,3 hebt die pessimistische Zuordnung des Entscheiders mit Untergrenze "
                    f"von {de(100 * base['pu'], 1)} auf {de(100 * veto['pu'], 1)} %, senkt sie für den kompensatorischen Entscheider von {de(100 * base['pg'], 1)} auf {de(100 * veto['pg'], 1)} %.")
