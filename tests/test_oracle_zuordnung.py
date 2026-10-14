"""Orakel: Zuordnungsregeln gegen einen unabhängigen Rechenweg (DEMO-PLAYBOOK §12).

1. Feste Profile: kein Klassenwechsel beim Hinzufügen beliebiger Alternativen (Summe, ELECTRE TRI beide Modi, FlowSort); Quantilklassen und Profile aus den Daten wechseln (Gegenprobe).
2. Rohwert-Entscheider: Summe gegen Profile = wahre Klasse; ohne Veto und mit lambda = 0,5 sind pessimistisch und optimistisch gleich.
3. Monotonie: ein höherer Wert in einem Kriterium senkt die Klasse nie (Summe, ELECTRE TRI pessimistisch ohne Veto, FlowSort).
4. ELECTRE TRI und FlowSort gegen Definitionsschleifen.
"""

import numpy as np
import pytest

import zpr_profiles as pf
import zpr_score as sc
import zpr_sorting as so

M = 5
SEEDS = tuple(range(300000, 300050))


def world(seed, n=12):
    rng = np.random.default_rng(seed)
    S = np.column_stack([rng.random(n) * 100, -(rng.random(n) * 4200 + 800), rng.random(n) * 100, -(rng.random(n) * 595 + 5), rng.random(n) * 100])
    w = rng.dirichlet(np.ones(M))
    return S, w


def test_fixed_profiles_never_change_class_when_alternatives_are_added():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    for seed in SEEDS:
        S, w = world(seed, 14)
        old, new = S[:12], S
        for method in ("summe", "tri_pess", "tri_opt", "flowsort"):
            c0 = so.classify(method, old, w, Sp, 3, 0.6, None)
            c1 = so.classify(method, new, w, Sp, 3, 0.6, None)[:12]
            assert c0.tolist() == c1.tolist(), method
            assert so.classify(method, old, w, Sp, 3, 0.6, 0.3).tolist() == so.classify(method, new, w, Sp, 3, 0.6, 0.3)[:12].tolist()


def test_quantile_classes_and_data_profiles_do_change_in_some_cases():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    changed = {"quantil": 0, "datenprofile": 0}
    for seed in SEEDS:
        S, w = world(seed, 14)
        for method in changed:
            if so.classify(method, S[:12], w, Sp, 3).tolist() != so.classify(method, S, w, Sp, 3)[:12].tolist():
                changed[method] += 1
    assert changed["quantil"] > 10 and changed["datenprofile"] > 10


def test_raw_value_decider_gives_the_sum_class_exactly():
    Sp, _ = pf.profile_values(pf.profile_levels(5), M)
    for seed in SEEDS:
        S, w = world(seed)
        G = np.random.default_rng(seed).random(S.shape)
        assert pf.true_class("roh_linear", G, S, w, 5).tolist() == so.class_by_sum(S, w, Sp).tolist()


def test_lambda_one_half_without_veto_makes_pessimistic_equal_optimistic():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    for seed in SEEDS:
        S, w = world(seed)
        assert so.electre_tri(S, w, Sp, 0.5, None, "pess").tolist() == so.electre_tri(S, w, Sp, 0.5, None, "opt").tolist()


def test_a_better_value_never_lowers_the_class():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    rng = np.random.default_rng(31)
    lo, hi = sc.nominal_bounds(M)
    for seed in SEEDS:
        S, w = world(seed)
        S2 = S.copy()
        j = int(rng.integers(M))
        S2[:, j] = np.minimum(S2[:, j] + 0.2 * (hi[j] - lo[j]), hi[j] * 1.2 if hi[j] > 0 else hi[j] + 0.2 * (hi[j] - lo[j]))
        S2[:, j] = np.maximum(S2[:, j], S[:, j])
        for method in ("summe", "tri_pess", "flowsort"):
            assert (so.classify(method, S2, w, Sp, 3, 0.6, None) >= so.classify(method, S, w, Sp, 3, 0.6, None)).all(), method


def test_electre_tri_follows_the_definition_loop():
    Sp, _ = pf.profile_values(pf.profile_levels(5), M)
    Nb = sc.normalize(Sp)
    for seed in SEEDS[:20]:
        S, w = world(seed, 8)
        Na = sc.normalize(S)
        for lam, veto in ((0.6, None), (0.7, 0.3)):
            def s_rel(a, b):
                conc = sum(w[j] for j in range(M) if a[j] >= b[j])
                if veto is not None and any(b[j] - a[j] > veto for j in range(M)):
                    return False
                return conc >= lam - 1e-12
            pess = []
            opt = []
            for a in Na:
                c = 0
                for h in range(len(Nb) - 1, -1, -1):
                    if s_rel(a, Nb[h]):
                        c = h + 1
                        break
                pess.append(c)
                o = len(Nb)
                for h in range(len(Nb)):
                    if s_rel(Nb[h], a) and not s_rel(a, Nb[h]):
                        o = h
                        break
                opt.append(o)
            assert so.electre_tri(S, w, Sp, lam, veto, "pess").tolist() == pess
            assert so.electre_tri(S, w, Sp, lam, veto, "opt").tolist() == opt


def test_flowsort_follows_the_definition_loop():
    Sp, _ = pf.profile_values(pf.profile_levels(3), M)
    Nb = sc.normalize(Sp)
    p = 0.5
    for seed in SEEDS[:20]:
        S, w = world(seed, 8)
        Na = sc.normalize(S)

        def flow(x, R):
            total = 0.0
            for r in R:
                for j in range(M):
                    d = x[j] - r[j]
                    total += w[j] * (min(1.0, max(0.0, d / p)) - min(1.0, max(0.0, -d / p)))
            return total / len(R)

        expected = []
        for a in Na:
            fa = flow(a, list(Nb))
            c = 0
            for h in range(len(Nb)):
                others = [Nb[i] for i in range(len(Nb)) if i != h] + [a]
                if fa >= flow(Nb[h], others):
                    c += 1
            expected.append(c)
        assert so.flowsort(S, w, Sp, p).tolist() == expected
