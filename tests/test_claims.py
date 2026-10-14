"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), 20 Alternativen, 5 Kriterien, nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte (Klassenwechsel 0,02),
damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler eines Anteils ist 1,0 bis 2,5 Punkte: Unterschiede unter etwa 5 Punkten sind NICHT
belastbar und stehen nirgends als Aussage.
"""

import pytest

import zpr_constants as C
import zpr_evaluation as ev


@pytest.fixture(scope="module")
def acc():
    return ev.experiment_accuracy()


@pytest.fixture(scope="module")
def stab():
    return ev.experiment_stability()


@pytest.fixture(scope="module")
def shift():
    return ev.experiment_shift()


@pytest.fixture(scope="module")
def res():
    return ev.experiment_resolution()


@pytest.fixture(scope="module")
def tri():
    return ev.experiment_tri()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")


# ----------------------------------------------------------------------------- Experiment 1: wie gut trifft welche Regel

@pytest.mark.parametrize("k,method,expected", [
    (3, "summe", [84.0, 88.1, 80.3, 100.0, 76.7]), (3, "flowsort", [85.4, 87.1, 79.3, 98.4, 75.9]), (3, "tri_pess", [70.0, 73.0, 71.9, 72.1, 65.1]),
    (3, "tri_opt", [70.5, 64.5, 59.0, 64.5, 60.0]), (3, "tri_pess_veto", [52.5, 61.9, 71.2, 61.2, 71.4]), (3, "quantil", [76.6, 76.3, 68.7, 79.1, 66.9]),
    (3, "datenprofile", [62.3, 62.5, 58.0, 63.0, 52.2]),
    (5, "summe", [73.6, 78.0, 64.9, 100.0, 65.5]), (5, "flowsort", [75.6, 76.6, 64.0, 96.5, 64.5]), (5, "tri_pess", [53.6, 56.8, 57.6, 55.9, 52.8]),
    (5, "quantil", [59.3, 56.6, 46.5, 60.8, 48.8]), (5, "datenprofile", [34.1, 33.2, 28.0, 33.4, 24.5]),
])
def test_share_assigned_correctly(acc, k, method, expected):
    assert [pct(cell(acc, k=k, decider=d, method=method)["acc"]) for d in DECIDERS] == pytest.approx(expected, abs=0.51)


def test_the_raw_value_decider_makes_the_sum_exactly_right(acc):
    assert cell(acc, k=3, decider="roh_linear", method="summe")["acc"] == 1.0
    assert cell(acc, k=3, decider="roh_linear", method="summe")["dist"] == 0.0


def test_mean_class_distance(acc):
    assert [round(cell(acc, k=3, decider=d, method="summe")["dist"], 3) for d in DECIDERS] == pytest.approx([0.167, 0.120, 0.208, 0.0, 0.304], abs=0.01)
    assert [round(cell(acc, k=3, decider=d, method="tri_pess")["dist"], 3) for d in DECIDERS] == pytest.approx([0.336, 0.295, 0.334, 0.311, 0.476], abs=0.01)


def test_flowsort_is_level_with_the_sum_and_tri_trails_for_a_compensatory_decider(acc):
    for d in DECIDERS:
        assert abs(pct(cell(acc, k=3, decider=d, method="flowsort")["acc"]) - pct(cell(acc, k=3, decider=d, method="summe")["acc"])) <= 1.7
    assert pct(cell(acc, k=3, decider="guete_linear", method="summe")["acc"]) - pct(cell(acc, k=3, decider="guete_linear", method="tri_pess")["acc"]) > 10


def test_veto_helps_only_the_decider_with_a_floor(acc):
    assert pct(cell(acc, k=3, decider="untergrenze", method="tri_pess_veto")["acc"]) > pct(cell(acc, k=3, decider="untergrenze", method="tri_pess")["acc"]) + 5
    assert pct(cell(acc, k=3, decider="guete_linear", method="tri_pess_veto")["acc"]) < pct(cell(acc, k=3, decider="guete_linear", method="tri_pess")["acc"]) - 10


# ----------------------------------------------------------------------------- Experiment 2: hängt die Zuordnung an den anderen

def test_fixed_profiles_never_change_class(stab):
    for method in ("summe", "tri_pess", "tri_opt", "flowsort"):
        r = cell(stab, method=method)
        assert r["changed"] == 0.0 and r["any"] == 0.0 and r["outlier"] == 0.0


def test_quantile_classes_and_profiles_from_the_data_bring_the_dependence_back(stab):
    q = cell(stab, method="quantil")
    d = cell(stab, method="datenprofile")
    assert (pct(q["changed"]), pct(q["any"]), pct(q["outlier"])) == pytest.approx((4.88, 62.0, 2.15), abs=0.51)
    assert (pct(d["changed"]), pct(d["any"]), pct(d["outlier"])) == pytest.approx((4.38, 56.5, 3.95), abs=0.51)


# ----------------------------------------------------------------------------- Experiment 3: was die Profile kosten

def test_shifted_profiles_cost_accuracy(shift):
    rows = {r["delta"]: r for r in shift}
    assert [pct(rows[d]["summe"]) for d in (-0.1, -0.05, 0.0, 0.05, 0.1)] == pytest.approx([71.5, 84.1, 84.0, 68.6, 54.1], abs=0.51)
    assert [pct(rows[d]["tri_pess"]) for d in (-0.1, -0.05, 0.0, 0.05, 0.1)] == pytest.approx([70.2, 71.5, 70.0, 65.7, 60.8], abs=0.51)
    assert [pct(rows[d]["flowsort"]) for d in (-0.1, -0.05, 0.0, 0.05, 0.1)] == pytest.approx([71.2, 84.4, 85.4, 69.8, 55.1], abs=0.51)
    assert [pct(rows[d]["top_share"]) for d in (-0.1, -0.05, 0.0, 0.05, 0.1)] == pytest.approx([51.8, 41.4, 31.7, 23.2, 15.7], abs=0.51)
    assert pct(rows[0.0]["true_top_share"]) == pytest.approx(38.4, abs=0.51)


def test_the_order_inside_a_class_is_lost(res):
    r3, r5 = cell(res, k=3), cell(res, k=5)
    assert (pct(r3["resolved"]), pct(r3["correct"])) == pytest.approx((63.8, 97.7), abs=0.51)
    assert (pct(r5["resolved"]), pct(r5["correct"])) == pytest.approx((77.0, 97.0), abs=0.51)


# ----------------------------------------------------------------------------- Experiment 4: ELECTRE TRI

@pytest.mark.parametrize("lam,veto,expected", [
    (0.5, None, (73.1, 73.1, 64.0, 64.0, 0.0)), (0.5, 0.3, (55.9, 61.6, 74.8, 50.5, 60.4)), (0.6, None, (70.0, 70.5, 65.1, 60.0, 26.8)), (0.6, 0.3, (52.5, 58.8, 71.4, 47.6, 68.5)),
    (0.7, None, (62.8, 63.1, 63.9, 52.7, 51.7)), (0.7, 0.3, (48.8, 53.5, 67.7, 43.1, 76.8)), (0.8, None, (52.7, 54.0, 61.4, 43.9, 72.2)), (0.8, 0.3, (44.7, 48.2, 63.6, 38.2, 84.8)),
])
def test_tri_threshold_and_veto(tri, lam, veto, expected):
    r = cell(tri, lam=lam, veto=veto)
    assert tuple(pct(r[k]) for k in ("pg", "og", "pu", "ou", "diff")) == pytest.approx(expected, abs=0.51)


def test_without_veto_and_lambda_one_half_pessimistic_equals_optimistic(tri):
    r = cell(tri, lam=0.5, veto=None)
    assert r["diff"] == 0.0 and r["pg"] == r["og"] and r["pu"] == r["ou"]
