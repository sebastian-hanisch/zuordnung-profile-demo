"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 1, n = 2, Alternative genau auf dem Profil, Konkordanz genau auf der Schwelle, k = 2, lauter gleiche Alternativen, Veto 0."""

import numpy as np
import pytest

import zpr_evaluation as ev
import zpr_profiles as pf
import zpr_score as sc
import zpr_sorting as so

M = 5


def alt(g):
    """Eine Alternative mit gleicher Güte g in jedem Kriterium (signierte Rohwerte)."""
    return pf.profile_values((g,), M)[0]


def test_single_alternative_and_two_alternatives():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    w = np.full(M, 0.2)
    one = alt(0.5)
    for method in ("summe", "tri_pess", "tri_opt", "flowsort", "quantil", "datenprofile"):
        c = so.classify(method, one, w, Sp, 3)
        assert c.shape == (1,) and 0 <= c[0] <= 2, method
    assert so.quantile_classes(one, w, 3).tolist() == [0]
    two = np.vstack([alt(0.3), alt(0.7)])
    assert so.quantile_classes(two, w, 3).tolist() == [0, 1]
    assert so.class_by_sum(two, w, Sp).tolist() == [0, 2]


def test_an_alternative_exactly_on_a_profile_reaches_it():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    w = np.full(M, 0.2)
    on = Sp.copy()
    assert so.class_by_sum(on, w, Sp).tolist() == [1, 2]
    assert so.electre_tri(on, w, Sp, 0.6, None, "pess").tolist() == [1, 2]              # Konkordanz 1,0 gegen das eigene Profil
    assert so.flowsort(on, w, Sp).tolist() == [1, 2]


def test_concordance_exactly_on_the_threshold_counts():
    Na = np.array([1.0, 0.0])
    Nb = np.array([0.5, 0.5])
    w = np.array([0.6, 0.4])
    assert so.outranks(Na, Nb, w, 0.6, None)                                              # 0,6 >= 0,6
    assert not so.outranks(Na, Nb, w, 0.600001, None)
    assert so.outranks(np.array([0.5, 0.5]), np.array([0.5, 0.5]), w, 1.0, 0.0)            # Gleichheit: Konkordanz 1, kein Veto bei Abstand 0 (nicht strikt größer)


def test_two_classes_use_one_profile():
    Sp = pf.profile_values((0.5,), M)[0]
    w = np.full(M, 0.2)
    S = np.vstack([alt(0.2), alt(0.5), alt(0.8)])
    assert so.class_by_sum(S, w, Sp).tolist() == [0, 1, 1]
    assert so.electre_tri(S, w, Sp, 0.6, None, "pess").tolist() == [0, 1, 1]
    assert so.electre_tri(S, w, Sp, 0.6, None, "opt").tolist() == [0, 1, 1]
    assert so.data_profiles(S, 2).shape == (1, M)


def test_all_alternatives_equal():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    w = np.full(M, 0.2)
    S = np.tile(alt(0.5), (6, 1))
    for method in ("summe", "tri_pess", "tri_opt", "flowsort"):
        c = so.classify(method, S, w, Sp, 3)
        assert len(set(c.tolist())) == 1, method
    assert sorted(np.bincount(so.quantile_classes(S, w, 3)).tolist()) == [2, 2, 2]        # Gleichstand wird nach Reihenfolge in Klassen verteilt (Rangquantile)


def test_veto_zero_is_a_strict_veto_on_any_shortfall():
    Na = np.array([0.5, 0.5, 0.49])
    Nb = np.array([0.5, 0.5, 0.5])
    w = np.array([0.4, 0.4, 0.2])
    assert so.outranks(Na, Nb, w, 0.6, None)
    assert not so.outranks(Na, Nb, w, 0.6, 0.0)                                           # jedes Defizit > 0 greift
    assert so.outranks(Na, Nb, w, 0.6, 0.02)                                              # Defizit 0,01 liegt unter dem Veto 0,02


def test_class_counts_and_deciders_in_the_edge_of_the_world():
    a = ev.analyse(ev.Settings(n=2, m=3, k=5, method="tri_pess", lam=0.8, veto=0.3, delta=-0.1, decider="untergrenze", seed=2))
    assert a.predicted.shape == (2,) and a.thresholds.shape == (4,)
    b = ev.analyse(ev.Settings(n=40, m=8, k=3, method="flowsort", seed=7))
    assert b.predicted.shape == (40,)


def test_normalization_outside_the_nominal_range_stays_finite():
    S = np.array([[500.0, -50.0, 900.0, -3.0, 1e4]])
    N = sc.normalize(S)
    assert np.isfinite(N).all() and (N > 1.0).any()
