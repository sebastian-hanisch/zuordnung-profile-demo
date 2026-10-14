"""Nenn-Spannweite, Normierung und Kendalls tau-b (Handwerte)."""

import numpy as np
import pytest

import zpr_score as sc


def test_nominal_bounds_follow_the_direction():
    lo, hi = sc.nominal_bounds(2)
    assert lo.tolist() == [0.0, -5000.0] and hi.tolist() == [100.0, -800.0]
    assert sc.directions(3).tolist() == [1.0, -1.0, 1.0]


def test_normalize_by_hand_and_independent_of_other_rows():
    S = np.array([[50.0, -2000.0], [100.0, -800.0], [0.0, -5000.0]])
    N = sc.normalize(S)
    assert N.ravel().tolist() == pytest.approx([0.5, 3000 / 4200, 1.0, 1.0, 0.0, 0.0])
    assert sc.normalize(S[:1]).ravel().tolist() == pytest.approx(N[0].tolist())
    assert sc.normalize(np.array([[150.0, -300.0]])).ravel().tolist() == pytest.approx([1.5, 4700 / 4200])       # außerhalb der Nenn-Spannweite


def test_kendall_tau_b_by_hand():
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(2 / np.sqrt(2 * 3))
