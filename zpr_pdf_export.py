"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import zpr_constants as C
from zpr_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "λ": "lambda", "δ": "delta", "≥": ">=", "≤": "<=", "≠": "!=", "−": "-", "€": "EUR", "·": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Zuordnung statt Rangfolge: vergleichen gegen feste Profile"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Alternativen, {s.m} Kriterien, {s.k} Klassen, Verfahren: {C.METHOD_LABELS[s.method]}. ELECTRE TRI: lambda = {de(s.lam, 1)}, "
                            f"Veto {'kein' if s.veto <= 0 else de(s.veto, 1)}. Profile um delta = {de(s.delta, 2)} verschoben. Entscheider: {C.DECIDER_LABELS[s.decider]}. "
                            f"Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    right = int((a.predicted == a.true_class).sum())
    pdf.multi_cell(0, 5, _s(f"Richtig zugeordnet: {right} von {s.n}; mittlerer Klassenabstand {de(float(np.abs(a.predicted - a.true_class).mean()), 3)}. "
                            f"Klassenwechsel der alten Alternativen bei einer zufällig hinzukommenden: {int(a.changed.sum())}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Alternativen (Nutzen, wahre Klasse, Klasse nach dem Verfahren)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i in range(s.n):
        u = "unter der Untergrenze" if a.utility[i] < -1e8 else de(float(a.utility[i]), 3)
        pdf.cell(0, 5, _s(f"{names[i]}: Nutzen {u}, wahre Klasse {int(a.true_class[i]) + 1}, Verfahren {int(a.predicted[i]) + 1}"
                          f"{'' if a.predicted[i] == a.true_class[i] else ' (falsch)'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: zuordnung-profile-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
