"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 Kriterien, w = (0,5; 0,3; 0,2), NORMIERTE Werte (Nenn-Spannweite); Profile b1 = (0,4; 0,4; 0,4), b2 = (0,7; 0,7; 0,7) (Summen 0,4 und 0,7).
    a1 = (0,9; 0,9; 0): Summe 0,72 -> Klasse 2; TRI lambda 0,6: Konkordanz gegen b2 = 0,8 -> pessimistisch 2, optimistisch 2; mit Veto 0,5: gegen b2 ist 0,7 - 0 = 0,7 > 0,5 (K3)
    -> pessimistisch 1 (gegen b1: 0,4 <= 0,5), optimistisch 2. FlowSort (V 0,5): Phi(a1) = (0,64 + 0,12)/2 = 0,38 gegen Phi(b1) = -0,62, Phi(b2) = 0,24 -> Klasse 2.
    a2 = (0,5; 0,5; 0,5): Summe 0,5 -> 1; TRI 1 / 1; FlowSort -0,10 gegen -0,40 und 0,50 -> 1. a3 = (0,3; 0,3; 0,3): Summe 0,3 -> 0; TRI 0 / 0; FlowSort -0,50 gegen -0,20 und 0,70 -> 0."""
    w = np.array([0.5, 0.3, 0.2])
    B = np.array([[0.4, 0.4, 0.4], [0.7, 0.7, 0.7]])
    A = np.array([[0.9, 0.9, 0.0], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    return w, B, A


def to_signed(N, m=3):
    """Normierte Werte (Nenn-Spannweite) in vorzeichenbehaftete Rohwerte zurückrechnen, damit die Funktionen der App sie normieren können."""
    import zpr_score as sc

    lo, hi = sc.nominal_bounds(m)
    return lo + np.asarray(N, dtype=float) * (hi - lo)


@pytest.fixture
def mini_signed(mini):
    w, B, A = mini
    return w, to_signed(B), to_signed(A)
