"""Profile und wahre Klasse des Entscheiders."""

import numpy as np
import pytest

import zpr_constants as C
import zpr_profiles as pf
import zpr_score as sc


def test_profile_levels_and_shift():
    assert pf.profile_levels(3) == (0.44, 0.56)
    assert pf.profile_levels(5) == (0.38, 0.47, 0.53, 0.62)
    assert pf.profile_levels(3, 0.1) == pytest.approx((0.54, 0.66))
    assert pf.profile_levels(3, -0.05) == pytest.approx((0.39, 0.51))


def test_profile_values_are_uniform_in_goodness_and_monotone_in_raw_values():
    Sp, Gp = pf.profile_values((0.44, 0.56), 5)
    assert Gp.tolist() == [[0.44] * 5, [0.56] * 5]
    assert Sp.shape == (2, 5)
    assert (Sp[1] > Sp[0]).all()                                    # höher ist besser im vorzeichenbehafteten Raum, auch bei ↓-Kriterien
    assert sc.normalize(Sp)[0, 0] == pytest.approx(0.44)            # ↑-Kriterium 1 linear: Güte = normierter Wert


def test_true_class_counts_the_profiles_reached():
    m = 5
    Sp, Gp = pf.profile_values(pf.profile_levels(3), m)
    w = np.full(m, 0.2)
    G = np.array([[0.3] * m, [0.44] * m, [0.5] * m, [0.56] * m, [0.9] * m])
    S, _ = pf.profile_values(tuple(G[:, 0]), m)                    # eine Alternative je Güte (gleichförmig)
    assert pf.true_class("guete_linear", G, S, w, 3).tolist() == [0, 1, 1, 2, 2]


def test_floor_decider_puts_an_alternative_with_one_bad_criterion_in_the_lowest_class():
    m = 5
    w = np.full(m, 0.2)
    G = np.array([[0.9, 0.9, 0.9, 0.9, 0.05], [0.9, 0.9, 0.9, 0.9, 0.1], [0.9] * 5])
    S, _ = pf.profile_values((0.9, 0.9, 0.9), m)
    S = S[:3].copy()
    assert pf.true_class("untergrenze", G, S, w, 3).tolist() == [0, 2, 2]       # 0,1 liegt genau auf der Untergrenze (inklusive)
    assert pf.true_class("guete_linear", G, S, w, 3).tolist() == [2, 2, 2]


def test_raw_value_decider_equals_the_nominal_sum_class():
    rng = np.random.default_rng(2)
    m = 5
    S = np.column_stack([rng.random(10) * 100, -(rng.random(10) * 4000 + 900), rng.random(10) * 100, -(rng.random(10) * 500 + 20), rng.random(10) * 100])
    w = rng.dirichlet(np.ones(m))
    G = rng.random((10, m))
    from zpr_sorting import class_by_sum

    Sp, _ = pf.profile_values(pf.profile_levels(3), m)
    assert pf.true_class("roh_linear", G, S, w, 3).tolist() == class_by_sum(S, w, Sp).tolist()


def test_decider_utility_forms_are_monotone_in_goodness():
    G = np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7]])
    w = np.full(5, 0.2)
    for kind in ("guete_linear", "guete_konkav", "guete_cobb"):
        u = pf.decider_utility(kind, G, np.zeros((2, 5)), w)
        assert u[1] > u[0], kind


def test_constants_are_consistent():
    assert set(C.LEVELS) == set(C.K_CHOICES)
    for k, lv in C.LEVELS.items():
        assert len(lv) == k - 1 and list(lv) == sorted(lv)
    assert set(C.METHOD_LABELS) == set(C.METHODS) == set(C.METHOD_SHORT) == set(C.METHOD_FIXED)
