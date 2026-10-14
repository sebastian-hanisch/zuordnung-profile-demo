"""Auswertung: Fall der App, Grenzfälle und die Verdikt-Texte (an von Hand gesetzten Zeilen: jede Zahl im Text)."""

import numpy as np
import pytest

import zpr_constants as C
import zpr_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    s = a.settings
    assert a.S.shape == (s.n, s.m) and len(a.true_class) == s.n and len(a.predicted) == s.n
    assert set(a.other) == set(C.METHODS) and (a.predicted == a.other[s.method]).all()
    assert set(a.true_class.tolist()) <= {0, 1, 2}
    assert len(a.thresholds) == s.k - 1 and a.profile_S.shape == (s.k - 1, s.m)
    assert (a.pess == a.other["tri_pess"]).all() and (a.opt == a.other["tri_opt"]).all()


def test_fixed_methods_never_change_class_when_an_alternative_is_added():
    for method in ("summe", "tri_pess", "tri_opt", "flowsort"):
        for seed in range(6):
            assert not ev.analyse(ev.Settings(method=method, seed=seed)).changed.any(), (method, seed)


def test_profiles_from_the_data_change_class_in_some_cases():
    assert any(ev.analyse(ev.Settings(method="datenprofile", seed=s)).changed.any() for s in range(10))
    assert any(ev.analyse(ev.Settings(method="quantil", seed=s)).changed.any() for s in range(10))


def test_shifted_profiles_move_the_assignments():
    base = ev.analyse(ev.Settings(method="summe", delta=0.0, seed=3))
    up = ev.analyse(ev.Settings(method="summe", delta=0.1, seed=3))
    assert (up.predicted <= base.predicted).all() and (up.predicted < base.predicted).any()
    assert (up.true_class == base.true_class).all()                             # die Grenzen des Entscheiders bleiben


def test_veto_setting_zero_means_no_veto():
    a = ev.analyse(ev.Settings(method="tri_pess", veto=0.0, lam=0.6, seed=2))
    from zpr_sorting import electre_tri

    assert a.pess.tolist() == electre_tri(a.S, a.w, a.profile_S, 0.6, None, "pess").tolist()


def test_every_method_decider_and_class_count_analyses_without_error():
    for method in C.METHODS:
        for dec in C.DECIDERS:
            for k in C.K_CHOICES:
                a = ev.analyse(ev.Settings(n=6, m=4, k=k, method=method, decider=dec, seed=4))
                assert a.predicted.max() <= k - 1


def test_two_alternatives_analyse():
    a = ev.analyse(ev.Settings(n=2, m=3, method="quantil", seed=1))
    assert a.predicted.shape == (2,)


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:6]
    acc = ev.experiment_accuracy(ks=(3,), deciders=("guete_linear", "untergrenze"), seeds=seeds)
    assert len(acc) == 2 * len(ev.ACC_METHODS) and all(0.0 <= r["acc"] <= 1.0 for r in acc)
    assert len(ev.experiment_stability(seeds=seeds)) == 6
    assert len(ev.experiment_shift(deltas=(0.0, 0.1), seeds=seeds)) == 2
    assert len(ev.experiment_resolution(seeds=seeds)) == 2
    assert len(ev.experiment_tri(lams=(0.5,), vetos=(None, 0.3), seeds=seeds)) == 2


# ----------------------------------------------------------------------------- Verdikte an von Hand gesetzten Zeilen

def _acc_rows(over=None):
    over = over or {}
    rows = []
    for k in (3, 5):
        for d in C.DECIDERS:
            for m in ev.ACC_METHODS:
                r = {"k": k, "decider": d, "method": m, "acc": 0.5, "dist": 0.5}
                r.update(over.get((k, d, m), {}))
                rows.append(r)
    return rows


def test_verdict_accuracy_names_each_number():
    rows = _acc_rows({(3, "guete_linear", "summe"): {"acc": 0.84}, (3, "guete_linear", "tri_pess"): {"acc": 0.70}, (3, "guete_linear", "flowsort"): {"acc": 0.854},
                        (3, "untergrenze", "tri_pess_veto"): {"acc": 0.714}, (3, "untergrenze", "tri_pess"): {"acc": 0.651}, (5, "guete_linear", "summe"): {"acc": 0.736}})
    kind, text = ev.verdict_accuracy(rows)
    assert kind == "warning"
    assert "ordnet die Summe gegen Profile 84,0 % der Alternativen" in text and "FlowSort 85,4 %" in text and "(pessimistisch, λ = 0,6) 70,0 %" in text
    assert "(71,4 gegen 65,1 %)" in text and "Bei 5 Klassen sinkt die Summe auf 73,6 %" in text


def _stab_rows(over=None):
    over = over or {}
    rows = []
    for m in ("summe", "tri_pess", "tri_opt", "flowsort", "quantil", "datenprofile"):
        r = {"method": m, "changed": 0.0, "any": 0.0, "outlier": 0.0}
        r.update(over.get(m, {}))
        rows.append(r)
    return rows


def test_verdict_stability_separates_fixed_and_relative_methods():
    rows = _stab_rows({"quantil": {"changed": 0.0488, "any": 0.62, "outlier": 0.0215}, "datenprofile": {"changed": 0.0438, "any": 0.565, "outlier": 0.0395},
                       "flowsort": {"changed": 0.0, "outlier": 0.0}})
    kind, text = ev.verdict_stability(rows)
    assert kind == "info"
    assert "(höchstens 0,00 %, auch nicht bei einem Ausreißer: 0,00 %)" in text
    assert "4,4 bis 4,9 % der alten Alternativen wechseln, in 56 bis 62 % der Matrizen" in text
    rows2 = _stab_rows({"summe": {"changed": 0.01, "outlier": 0.02}, "quantil": {"changed": 0.05, "any": 0.6}, "datenprofile": {"changed": 0.04, "any": 0.5}})
    assert "höchstens 1,00 %, auch nicht bei einem Ausreißer: 2,00 %" in ev.verdict_stability(rows2)[1]


def test_verdict_price_takes_the_worse_side_of_the_shift():
    shift = [{"delta": d, "summe": v} for d, v in ((-0.1, 0.715), (-0.05, 0.841), (0.0, 0.84), (0.05, 0.686), (0.1, 0.541))]
    res = [{"k": 3, "resolved": 0.638, "correct": 0.977}, {"k": 5, "resolved": 0.77, "correct": 0.97}]
    kind, text = ev.verdict_price(shift, res)
    assert kind == "warning"
    assert "bis zu 15,4 Punkte weniger (68,6 gegen 84,0 %)" in text and "bei 0,10 bis zu 29,9 Punkte" in text
    assert "36,2 % der Paare in derselben Klasse" in text and "zu 97,7 % richtig geordnet" in text


def test_verdict_tri_names_the_veto_effect():
    rows = []
    for lam, veto, pg, pu, diff in ((0.5, None, 0.731, 0.64, 0.0), (0.5, 0.3, 0.559, 0.748, 0.604), (0.8, None, 0.527, 0.614, 0.722)):
        rows.append({"lam": lam, "veto": veto, "pg": pg, "og": pg, "pu": pu, "ou": pu, "diff": diff})
    kind, text = ev.verdict_tri(rows)
    assert kind == "info"
    assert "gleich (0,0 % verschieden)" in text and "bei λ = 0,8 weichen sie in 72,2 % ab und der Anteil richtig fällt auf 52,7 %" in text
    assert "von 64,0 auf 74,8 %" in text and "von 73,1 auf 55,9 %" in text
