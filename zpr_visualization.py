"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import zpr_constants as C
import zpr_score as sc
import zpr_sorting as so
from zpr_evaluation import ACC_LABELS, ACC_METHODS, Analysis, de

CLASS_NAMES = {3: ("niedrig", "mittel", "hoch"), 5: ("sehr niedrig", "niedrig", "mittel", "hoch", "sehr hoch")}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def class_label(k: int, c: int) -> str:
    return f"{c + 1} ({CLASS_NAMES[k][c]})"


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_strip(a: Analysis) -> go.Figure:
    """Nutzen des Entscheiders je Alternative, Klasse nach dem Verfahren als Höhe; senkrechte Linien: Grenzen des Entscheiders; rot: falsch zugeordnet."""
    s = a.settings
    names = alt_names(s.n)
    finite = a.utility[a.utility > -1e8]
    floor = float(finite.min()) - 0.05 if len(finite) else 0.0                 # Untergrenze verletzt: links von allen anderen eingezeichnet
    u = np.where(a.utility < -1e8, floor, a.utility)
    wrong = a.predicted != a.true_class
    fig = go.Figure()
    for c in range(s.k):
        sel = np.flatnonzero((a.predicted == c) & ~wrong)
        fig.add_trace(go.Scatter(x=u[sel], y=np.full(len(sel), c) + 0.0, mode="markers+text", text=[names[i] for i in sel], textposition="top center", textfont=dict(size=9),
                                 marker=dict(size=11, color=C.COLOR_CLASS[c]), name=f"Klasse {class_label(s.k, c)}", hovertemplate="%{text}: Nutzen %{x:.3f}<extra></extra>"))
    w_idx = np.flatnonzero(wrong)
    fig.add_trace(go.Scatter(x=u[w_idx], y=a.predicted[w_idx].astype(float), mode="markers+text", text=[names[i] for i in w_idx], textposition="top center", textfont=dict(size=9),
                             marker=dict(size=12, color=C.COLOR_WRONG, symbol="x"), name="falsch zugeordnet", hovertemplate="%{text}: Nutzen %{x:.3f}<extra></extra>"))
    for t in a.thresholds:
        fig.add_vline(x=float(t), line=dict(color=C.COLOR_LINE, dash="dot"))
    fig.update_yaxes(title="Klasse nach dem Verfahren", tickmode="array", tickvals=list(range(s.k)), ticktext=[class_label(s.k, c) for c in range(s.k)], range=[-0.5, s.k - 0.5])
    fig.update_xaxes(title="Nutzen des Entscheiders (Linien: seine Klassengrenzen)")
    fig.update_layout(title=dict(text="Zuordnung der Alternativen", font=dict(size=13)))
    return _fixed(fig, 380)


def build_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    names = alt_names(s.n)
    df = pd.DataFrame({"Alternative": names, "Nutzen (Entscheider)": np.where(a.utility < -1e8, np.nan, a.utility),
                       "wahre Klasse": [class_label(s.k, int(c)) for c in a.true_class], "Klasse (Verfahren)": [class_label(s.k, int(c)) for c in a.predicted],
                       "richtig": np.where(a.predicted == a.true_class, "ja", "nein"), "bei neuer Alternative": np.where(a.changed, "wechselt", "bleibt")})
    return df


def style_table(df: pd.DataFrame):
    return df.style.format({"Nutzen (Entscheider)": lambda v: "unter der Untergrenze" if v != v else de(v, 3)})


def profile_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    out = {"Profil": [f"b{h + 1}" for h in range(len(a.profile_S))], "Güte (verschoben)": [float(g[0]) for g in a.profile_G], "Nutzen an den Grenzen des Entscheiders": a.thresholds}
    for j in range(s.m):
        out[f"K{j + 1}"] = a.profile_S[:, j]
    return pd.DataFrame(out)


def style_profiles(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 2)) for c in df.columns if c != "Profil"})


def method_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    rows = []
    for k_ in C.METHODS:
        p = a.other[k_]
        rows.append({"Verfahren": C.METHOD_LABELS[k_], "Referenz": "fest (Profile)" if C.METHOD_FIXED[k_] else "aus der Menge", "richtig": f"{int((p == a.true_class).sum())} von {s.n}",
                     "mittlerer Klassenabstand": de(float(np.abs(p - a.true_class).mean()), 3)})
    return pd.DataFrame(rows)


def pair_details(a: Analysis, i: int) -> pd.DataFrame:
    """Je Profil: Konkordanz der Alternative gegen das Profil und umgekehrt, Veto, ob die Alternative das Profil übertrifft (ELECTRE TRI mit den Einstellungen)."""
    s = a.settings
    Na = sc.normalize(a.S)[i]
    Nb = sc.normalize(a.profile_S)
    veto = None if s.veto <= 0 else s.veto
    rows = []
    for h in range(len(Nb)):
        conc = float(((Na >= Nb[h]) * a.w).sum())
        back = float(((Nb[h] >= Na) * a.w).sum())
        vetoed = veto is not None and bool(((Nb[h] - Na) > veto).any())
        rows.append({"Profil": f"b{h + 1}", "Konkordanz a über b": conc, "Konkordanz b über a": back, "Veto gegen a über b": "ja" if vetoed else "nein",
                     "a übertrifft b": "ja" if so.outranks(Na, Nb[h], a.w, s.lam, veto) else "nein", "b übertrifft a": "ja" if so.outranks(Nb[h], Na, a.w, s.lam, veto) else "nein"})
    return pd.DataFrame(rows)


def style_pair(df: pd.DataFrame):
    return df.style.format({"Konkordanz a über b": lambda v: de(v, 2), "Konkordanz b über a": lambda v: de(v, 2)})


# ----------------------------------------------------------------------------- Experimente

def accuracy_table(rows: list[dict], k: int) -> pd.DataFrame:
    deciders = list(dict.fromkeys(r["decider"] for r in rows))
    out = []
    for nme in ACC_METHODS:
        row = {"Verfahren": ACC_LABELS[nme]}
        for d in deciders:
            r = next(x for x in rows if x["k"] == k and x["decider"] == d and x["method"] == nme)
            row[C.DECIDER_SHORT[d]] = f"{de(100 * r['acc'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def build_accuracy(rows: list[dict], k: int = 3) -> go.Figure:
    deciders = list(dict.fromkeys(r["decider"] for r in rows))
    fig = go.Figure()
    for d in deciders:
        fig.add_trace(go.Bar(x=[ACC_LABELS[n] for n in ACC_METHODS], y=[100 * next(x for x in rows if x["k"] == k and x["decider"] == d and x["method"] == n)["acc"] for n in ACC_METHODS],
                             name=C.DECIDER_SHORT[d]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="richtig zugeordnet (%)", range=[0, 102])
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 420)


def stability_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": C.METHOD_SHORT[r["method"]], "Referenz": "fest" if C.METHOD_FIXED[r["method"]] else "aus der Menge",
                          "alte mit Klassenwechsel (zufällige neue)": f"{de(100 * r['changed'], 2)} %", "Matrizen mit Wechsel": f"{de(100 * r['any'], 1)} %",
                          "alte mit Wechsel (Ausreißer)": f"{de(100 * r['outlier'], 2)} %"} for r in rows])


def shift_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verschiebung δ (Güte)": de(r["delta"], 2), "Summe gegen Profile": f"{de(100 * r['summe'], 1)} %", "ELECTRE TRI pessimistisch": f"{de(100 * r['tri_pess'], 1)} %",
                          "FlowSort": f"{de(100 * r['flowsort'], 1)} %", "Anteil oberste Klasse (Summe)": f"{de(100 * r['top_share'], 1)} %",
                          "wahre oberste Klasse": f"{de(100 * r['true_top_share'], 1)} %"} for r in rows])


def build_shift(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for key, label in (("summe", "Summe gegen Profile"), ("tri_pess", "ELECTRE TRI pessimistisch"), ("flowsort", "FlowSort")):
        fig.add_trace(go.Scatter(x=[r["delta"] for r in rows], y=[100 * r[key] for r in rows], mode="lines+markers", name=label, line=dict(width=3)))
    fig.update_xaxes(title="Verschiebung der Profile gegenüber den Grenzen des Entscheiders (Güte)")
    fig.update_yaxes(title="richtig zugeordnet (%)", range=[0, 102])
    return _fixed(fig, 360)


def resolution_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Klassen k": r["k"], "Paare in verschiedenen Klassen": f"{de(100 * r['resolved'], 1)} %", "davon richtig geordnet": f"{de(100 * r['correct'], 1)} %",
                          "Paare ohne Ordnung": f"{de(100 * (1 - r['resolved']), 1)} %"} for r in rows])


def tri_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"λ": de(r["lam"], 1), "Veto": "kein" if r["veto"] is None else de(r["veto"], 1), "pessimistisch richtig (Güte/linear)": f"{de(100 * r['pg'], 1)} %",
                          "optimistisch richtig (Güte/linear)": f"{de(100 * r['og'], 1)} %", "pessimistisch richtig (Untergrenze)": f"{de(100 * r['pu'], 1)} %",
                          "optimistisch richtig (Untergrenze)": f"{de(100 * r['ou'], 1)} %", "pessimistisch ≠ optimistisch": f"{de(100 * r['diff'], 1)} %"} for r in rows])
