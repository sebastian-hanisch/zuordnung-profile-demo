"""Konstanten: Regler-Grenzen, Klassen und Profile, Verfahren, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 12
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 3           # je Preset geprüft (tests/test_presets.py)

K_CHOICES = (3, 5)                            # Zahl der Klassen
DEFAULT_K = 3
# Güte der gleichförmigen Profile (gleiche Güte in jedem Kriterium): etwa Terzile bzw. Quintile des Nutzens bei m = 5
LEVELS = {3: (0.44, 0.56), 5: (0.38, 0.47, 0.53, 0.62)}
THETA = 0.1                                   # Untergrenze der Güte des Entscheiders mit Untergrenze

METHODS = ("summe", "tri_pess", "tri_opt", "flowsort", "quantil", "datenprofile")
METHOD_LABELS = {
    "summe": "Summe gegen Profile (gewichtete Summe gegen die der Profile)",
    "tri_pess": "ELECTRE TRI, pessimistisch (nicht kompensatorisch)",
    "tri_opt": "ELECTRE TRI, optimistisch (nicht kompensatorisch)",
    "flowsort": "FlowSort (Nettofluss gegen die Profile, V-Form p = 0,5; eine Lesart)",
    "quantil": "Quantilklassen (Klassen sind Rangquantile der Menge: relativ)",
    "datenprofile": "Summe gegen Profile aus den Quantilen der Menge (relativ)",
}
METHOD_SHORT = {"summe": "Summe gegen Profile", "tri_pess": "TRI pessimistisch", "tri_opt": "TRI optimistisch", "flowsort": "FlowSort", "quantil": "Quantilklassen",
                "datenprofile": "Profile aus Quantilen"}
METHOD_FIXED = {"summe": True, "tri_pess": True, "tri_opt": True, "flowsort": True, "quantil": False, "datenprofile": False}   # feste Referenz (Profile) oder aus der Menge
DEFAULT_METHOD = "summe"

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite)",
    "untergrenze": "Linear in der Güte, mit Untergrenze (kein Kriterium unter 0,1)",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear", "untergrenze": "Güte/Untergrenze"}
DEFAULT_DECIDER = "guete_linear"

LAM_CHOICES = (0.5, 0.6, 0.7, 0.8)           # Konkordanzschwelle von ELECTRE TRI
DEFAULT_LAM = 0.6
VETO_CHOICES = (0.0, 0.3)                    # Veto: ein Kriterium, in dem die Alternative um mehr als diesen Anteil der Nenn-Spannweite unter dem Profil liegt; 0 = kein Veto
DEFAULT_VETO = 0.0
DELTA_CHOICES = (-0.1, -0.05, 0.0, 0.05, 0.1)    # Verschiebung der Profile (Güte) gegenüber den Grenzen des Entscheiders
DEFAULT_DELTA = 0.0
FLOW_P = 0.5                                 # V-Form von FlowSort: Anteil der Nenn-Spannweite

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20
EXP_M = 5
EXP_LAMS = (0.5, 0.6, 0.7, 0.8)
EXP_VETOS = (None, 0.3)
OFFSET = 7919                                # Strom des Ausreißers: SplitMix64(seed + OFFSET)
OUTLIER_SHARE = 0.5

COLOR_CLASS = ("#8A96A6", "#4c78a8", "#2F6B65", "#D68A2E", "#7a5195")
COLOR_WRONG = "#c0392b"
COLOR_LINE = "rgba(80,80,80,0.7)"

PRESETS = {
    "Standardfall (Summe gegen Profile, 3 Klassen)": {"n": DEFAULT_N, "m": DEFAULT_M, "k": 3, "method": "summe", "lam": 0.6, "veto": 0.0, "delta": 0.0, "decider": "guete_linear", "seed": 3},
    "Veto": {"n": 12, "m": 5, "k": 3, "method": "tri_pess", "lam": 0.5, "veto": 0.3, "delta": 0.0, "decider": "untergrenze", "seed": 2},
    "Pessimistisch gegen optimistisch": {"n": 12, "m": 5, "k": 3, "method": "tri_pess", "lam": 0.7, "veto": 0.0, "delta": 0.0, "decider": "guete_linear", "seed": 1},
    "Profile aus den Daten": {"n": 12, "m": 5, "k": 3, "method": "datenprofile", "lam": 0.6, "veto": 0.0, "delta": 0.0, "decider": "guete_linear", "seed": 1},
    "Profil daneben": {"n": 12, "m": 5, "k": 3, "method": "summe", "lam": 0.6, "veto": 0.0, "delta": 0.1, "decider": "guete_linear", "seed": 1},
}
PRESET_HELP = {
    "Standardfall (Summe gegen Profile, 3 Klassen)": "12 Alternativen, 5 Kriterien, 3 Klassen mit zwei festen Grenzprofilen; die gewichtete Summe gegen die der Profile ordnet die meisten richtig zu.",
    "Veto": "Entscheider mit Untergrenze: ELECTRE TRI mit Veto schickt eine Alternative mit einem sehr schlechten Kriterium in die niedrigere Klasse, ohne Veto landet sie zu hoch.",
    "Pessimistisch gegen optimistisch": "Konkordanzschwelle 0,7: ELECTRE TRI pessimistisch und optimistisch ordnen einen Teil der Alternativen verschiedenen Klassen zu.",
    "Profile aus den Daten": "Die Profile sind Quantile der Menge: Kommt eine Alternative dazu, wechseln alte Alternativen die Klasse. Bei festen Profilen passiert das nie.",
    "Profil daneben": "Die Profile liegen um 0,10 Güte über den Grenzen des Entscheiders: ein Teil der Alternativen landet eine Klasse zu niedrig.",
}
