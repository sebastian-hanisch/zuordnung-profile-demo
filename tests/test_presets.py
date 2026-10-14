"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import zpr_constants as C
import zpr_evaluation as ev
import zpr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["k"] in C.K_CHOICES and p["method"] in C.METHODS and p["lam"] in C.LAM_CHOICES and p["veto"] in C.VETO_CHOICES, name
        assert p["delta"] in C.DELTA_CHOICES and p["decider"] in C.DECIDERS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Summe gegen Profile, 3 Klassen)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.METHOD_LABELS) == set(C.METHODS) == set(C.METHOD_SHORT) == set(C.METHOD_FIXED)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (Summe gegen Profile, 3 Klassen)")
    right = int((a.predicted == a.true_class).sum())
    assert 9 <= right < a.settings.n and not a.changed.any()

    a = _analysis("Veto")
    b = _analysis("Veto", veto=0.0)
    assert int((a.predicted == a.true_class).sum()) >= int((b.predicted == b.true_class).sum()) + 2

    a = _analysis("Pessimistisch gegen optimistisch")
    assert int((a.pess != a.opt).sum()) >= 3

    a = _analysis("Profile aus den Daten")
    assert a.changed.any() and not _analysis("Profile aus den Daten", method="summe").changed.any()

    a = _analysis("Profil daneben")
    assert int((a.predicted < a.true_class).sum()) >= 3 and not (a.predicted > a.true_class).any()
