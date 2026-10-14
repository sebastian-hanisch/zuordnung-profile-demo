"""Zuordnungsregeln: Handwerte der Mini-Instanz, Schwellen inklusive, Veto, pessimistisch gegen optimistisch."""

import numpy as np
import pytest

import zpr_score as sc
import zpr_sorting as so


def test_class_by_sum_by_hand(mini_signed):
    w, Bs, As = mini_signed
    assert so.class_by_sum(As, w, Bs).tolist() == [2, 1, 0]


def test_the_class_threshold_is_inclusive(mini_signed):
    w, Bs, _ = mini_signed
    assert so.class_by_sum(Bs, w, Bs).tolist() == [1, 2]                 # ein Profil gegen sich selbst: Summe gleich, erreicht = Klasse


def test_electre_tri_pessimistic_and_optimistic_by_hand(mini_signed):
    w, Bs, As = mini_signed
    assert so.electre_tri(As, w, Bs, 0.6, None, "pess").tolist() == [2, 1, 0]
    assert so.electre_tri(As, w, Bs, 0.6, None, "opt").tolist() == [2, 1, 0]


def test_veto_demotes_an_alternative_with_one_very_bad_criterion(mini_signed):
    w, Bs, As = mini_signed
    assert so.electre_tri(As, w, Bs, 0.6, 0.5, "pess").tolist() == [1, 1, 0]
    assert so.electre_tri(As, w, Bs, 0.6, 0.5, "opt").tolist() == [2, 1, 0]


def test_outranks_threshold_and_veto_are_inclusive_where_documented(mini):
    w, B, A = mini
    # Konkordanz a1 gegen b1 = 0,8; mit lam = 0,8 gilt die Schwelle inklusive
    assert so.outranks(A[0], B[0], w, 0.8, None)
    assert not so.outranks(A[0], B[0], w, 0.81, None)
    # Veto: b - a im dritten Kriterium = 0,4: bei Veto 0,4 greift es nicht (strikt größer), bei 0,39 schon
    assert so.outranks(A[0], B[0], w, 0.6, 0.4)
    assert not so.outranks(A[0], B[0], w, 0.6, 0.39)


def test_electre_tri_rejects_an_unknown_mode(mini_signed):
    w, Bs, As = mini_signed
    with pytest.raises(ValueError):
        so.electre_tri(As, w, Bs, 0.6, None, "mittel")


def test_flowsort_by_hand(mini, mini_signed):
    w, B, A = mini
    _, Bs, As = mini_signed
    assert so.flowsort(As, w, Bs).tolist() == [2, 1, 0]
    assert so.net_flow_against(A[0], B, w) == pytest.approx(0.38)
    assert so.net_flow_against(B[0], np.vstack([B[1:], A[0]]), w) == pytest.approx(-0.62)
    assert so.net_flow_against(B[1], np.vstack([B[:1], A[0]]), w) == pytest.approx(0.24)


def test_quantile_classes_depend_on_the_set_and_fixed_cuts_do_not():
    s_old = np.array([0.9, 0.6, 0.5, 0.1])
    s_new = np.array([0.9, 0.6, 0.5, 0.1, 0.95])
    quant = lambda s, k=2: ((np.argsort(np.argsort(s, kind="stable"), kind="stable") * k) // len(s)).tolist()  # noqa: E731
    assert quant(s_old) == [1, 1, 0, 0]
    assert quant(s_new)[:4] == [1, 0, 0, 0]
    assert (s_old >= 0.55).astype(int).tolist() == (s_new >= 0.55).astype(int).tolist()[:4]


def test_quantile_classes_function_and_data_profiles():
    rng = np.random.default_rng(5)
    S = np.column_stack([rng.random(9) * 100, -(rng.random(9) * 4000 + 900), rng.random(9) * 100])
    w = np.array([0.4, 0.3, 0.3])
    c = so.quantile_classes(S, w, 3)
    assert sorted(np.bincount(c).tolist()) == [3, 3, 3]                       # 9 Alternativen: 3 je Klasse
    Sp = so.data_profiles(S, 3)
    assert Sp.shape == (2, 3) and (Sp[0] <= Sp[1]).all()
    assert Sp[0].tolist() == pytest.approx(np.quantile(S, 1 / 3, axis=0).tolist())


def test_classify_dispatch_and_unknown_method(mini_signed):
    w, Bs, As = mini_signed
    for method in ("summe", "tri_pess", "tri_opt", "flowsort", "quantil", "datenprofile"):
        assert so.classify(method, As, w, Bs, 3, 0.6, None).shape == (3,)
    with pytest.raises(ValueError):
        so.classify("orakel", As, w, Bs, 3)


def test_normalize_is_independent_of_the_set(mini_signed):
    _, _, As = mini_signed
    assert sc.normalize(As[:1]).ravel().tolist() == pytest.approx(sc.normalize(As)[0].tolist())
