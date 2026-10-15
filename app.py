"""Zuordnung statt Rangfolge: vergleichen gegen feste Profile - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Linie Mehrkriterien-Ranking (Kind der Rangumkehr). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seinen Nutzen und seine Klassengrenzen.
Jede Alternative wird gegen feste Grenzprofile einer von k geordneten Klassen zugeordnet, ohne sie mit den anderen zu vergleichen: Summe gegen Profile, ELECTRE TRI, FlowSort.
Kern: zpr_sorting.py, zpr_profiles.py, zpr_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import zpr_constants as C
from zpr_evaluation import (
    Settings,
    analyse,
    de,
    experiment_accuracy,
    experiment_resolution,
    experiment_shift,
    experiment_stability,
    experiment_tri,
    verdict_accuracy,
    verdict_price,
    verdict_stability,
    verdict_tri,
)
from zpr_pdf_export import build_pdf
from zpr_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from zpr_visualization import (
    accuracy_table,
    alt_names,
    build_accuracy,
    build_shift,
    build_strip,
    build_table,
    method_table,
    pair_details,
    profile_table,
    resolution_table,
    shift_table,
    stability_table,
    style_pair,
    style_profiles,
    style_table,
    tri_table,
)

st.set_page_config(page_title="Zuordnung statt Rangfolge – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {"accuracy": experiment_accuracy(), "stability": experiment_stability(), "shift": experiment_shift(), "resolution": experiment_resolution(), "tri": experiment_tri()}


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎯 Zuordnung statt Rangfolge: vergleichen gegen feste Profile")
st.markdown(
    """
**Eine Zuordnung gegen feste Profile hängt nicht von den anderen Alternativen ab, aber sie setzt Profile und eine Zuordnungsregel voraus.** Dies ist das zehnte Stück der Linie
**Mehrkriterien-Ranking**: Statt Alternativen untereinander zu ordnen (und bei jeder neuen die Ordnung zu riskieren, Stück 9), wird jede einzeln mit **Grenzprofilen** verglichen und
einer von k geordneten **Klassen** zugeordnet: mit der gewichteten Summe, mit **ELECTRE TRI** (Konkordanz, Veto, pessimistisch und optimistisch) oder mit **FlowSort** (Fluss gegen die
Profile). Die Demo misst, wie gut das trifft, was die Profile kosten und ob die Abhängigkeit zurückkehrt, wenn die Profile aus der Menge kommen. Der Entscheider ist simuliert.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Zuordnung", expanded=True):
    st.markdown(
        """
1. **Profile:** k − 1 Grenzalternativen mit gleicher Güte in jedem Kriterium trennen k Klassen (1 = niedrigste). Die wahre Klasse des Entscheiders ist die Zahl der Profile, deren Nutzen
   sein Nutzen der Alternative erreicht.
2. **Summe gegen Profile:** die gewichtete Summe der Alternative gegen die der Profile (Nenn-Spannweite als Skala).
3. **ELECTRE TRI:** a übertrifft ein Profil, wenn die Gewichte der Kriterien, in denen a mindestens so gut ist, die Schwelle λ erreichen und kein Veto greift. *Pessimistisch:* die höchste
   Klasse, deren unteres Profil a übertrifft; *optimistisch:* unterhalb des ersten Profils, das a strikt übertrifft.
4. **FlowSort (eine Lesart):** Nettofluss der Alternative gegen die Profile (V-Form), verglichen mit den Flüssen der Profile.
5. **Absolut gegen relativ:** Mit festen Profilen wechselt nie eine Klasse, wenn eine Alternative dazukommt. Kommen Klassen oder Profile aus der Menge (Quantile), kehrt die Abhängigkeit zurück.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zugeordnet werden.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    k = st.selectbox("Klassen k", C.K_CHOICES, key="k_select", help="In wie viele geordnete Klassen zugeordnet wird; die Profile liegen bei Güte 0,44 / 0,56 (3) bzw. 0,38 / 0,47 / 0,53 / 0,62 (5).")
    st.markdown("**Zuordnungsregel**")
    method = st.selectbox("Verfahren", C.METHODS, key="method_select", format_func=lambda x: C.METHOD_LABELS[x],
                          help="Feste Referenz (Profile) oder aus der Menge (Quantile): nur die ersten vier hängen nicht von den anderen Alternativen ab.")
    uses_tri = method in ("tri_pess", "tri_opt")
    lam = st.selectbox("Konkordanzschwelle λ", C.LAM_CHOICES, key="lam_select", format_func=lambda v: de(v, 1), disabled=not uses_tri,
                       help="Wie viel Gewicht die Kriterien haben müssen, in denen die Alternative mindestens so gut ist wie das Profil. Nur bei ELECTRE TRI.")
    veto = st.selectbox("Veto", C.VETO_CHOICES, key="veto_select", format_func=lambda v: "kein Veto" if v == 0 else f"{de(v, 1)} der Nenn-Spannweite", disabled=not uses_tri,
                        help="Ein Kriterium, in dem die Alternative um mehr als diesen Anteil unter dem Profil liegt, verhindert das Übertreffen. Nur bei ELECTRE TRI.")
    delta = st.selectbox("Profile verschoben um δ (Güte)", C.DELTA_CHOICES, key="delta_select", format_func=lambda v: de(v, 2),
                         help="Wie weit die Profile der Verfahren neben den Grenzen des Entscheiders liegen: der Preis, Profile setzen zu müssen.")
    st.markdown("**Entscheider (simuliert)**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda x: C.DECIDER_LABELS[x],
                           help="Wie der verborgene Entscheider bewertet; seine Klassengrenzen sind der Nutzen der unverschobenen Profile.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "k_select": int(k), "method_select": method, "lam_select": float(lam), "veto_select": float(veto),
                   "delta_select": float(delta), "decider_select": decider, "seed_input": int(seed)})

settings = Settings(int(n), int(m), int(k), method, float(lam), float(veto), float(delta), decider, int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
right = int((a.predicted == a.true_class).sum())
dist = float(np.abs(a.predicted - a.true_class).mean())
fixed = C.METHOD_FIXED[settings.method]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Richtig zugeordnet", f"{right} von {settings.n}", delta=f"mittlerer Klassenabstand {de(dist, 2)}", delta_color="off",
          help="Alternativen, deren Klasse nach dem Verfahren der wahren Klasse des Entscheiders entspricht.")
c2.metric("Zu hoch / zu niedrig", f"{int((a.predicted > a.true_class).sum())} / {int((a.predicted < a.true_class).sum())}",
          delta=f"verschobene Profile: δ = {de(settings.delta, 2)}", delta_color="off", help="Falsch zugeordnete Alternativen nach der Richtung des Fehlers.")
c3.metric("Pessimistisch ≠ optimistisch", f"{int((a.pess != a.opt).sum())} von {settings.n}", delta=f"λ = {de(settings.lam, 1)}, {'kein Veto' if settings.veto <= 0 else 'Veto ' + de(settings.veto, 1)}",
          delta_color="off", help="Alternativen, die ELECTRE TRI pessimistisch und optimistisch verschiedenen Klassen zuordnet (mit den Einstellungen links).")
c4.metric("Neue Alternative: Klassenwechsel", f"{int(a.changed.sum())} von {settings.n}", delta="fest: nie" if fixed else "relativ: kehrt zurück", delta_color="off",
          help="Alte Alternativen, die die Klasse wechseln, wenn eine weitere (die (n + 1)-te Zeile der Matrix) dazukommt. Gegen feste Profile nie.")

left, right_col = st.columns([3, 2])
with left:
    st.plotly_chart(build_strip(a), width="stretch", key="strip_chart")
with right_col:
    st.markdown("**Grenzprofile** (Rohwerte; Nutzen an den Grenzen des Entscheiders)")
    st.dataframe(style_profiles(profile_table(a)), hide_index=True, width="stretch")
    st.caption("Die Profile haben in jedem Kriterium dieselbe Güte; ↓-Kriterien (Euro, Minuten) sind mit Vorzeichen angegeben, höher ist besser.")

with st.expander("Eine Alternative gegen die Profile: Konkordanz und Veto"):
    ia = st.selectbox("Alternative", list(range(settings.n)), index=0, format_func=lambda i: names[i], key="pair_alt")
    st.dataframe(style_pair(pair_details(a, int(ia))), hide_index=True, width="stretch")
    st.caption(f"{names[int(ia)]}: wahre Klasse {int(a.true_class[int(ia)]) + 1}, Klasse nach dem Verfahren {int(a.predicted[int(ia)]) + 1}; ELECTRE TRI pessimistisch "
               f"{int(a.pess[int(ia)]) + 1}, optimistisch {int(a.opt[int(ia)]) + 1}.")

tab_t, tab_m = st.tabs(["Alle Alternativen", "Verfahren im Vergleich"])
with tab_t:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch")
    if settings.decider == "untergrenze":
        st.caption("Entscheider mit Untergrenze: ein Kriterium unter Güte 0,1 gibt die niedrigste Klasse (Nutzen „unter der Untergrenze“).")
with tab_m:
    st.dataframe(method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider, alle sechs Verfahren (mit den Einstellungen links). Quantilklassen und Profile aus den Quantilen bewerten relativ zur Menge.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="zuordnung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet „gegen feste Profile“?")
if fixed:
    st.info(f"Das Verfahren bewertet jede Alternative für sich gegen feste Profile: Kommt eine Alternative dazu, wechselt keine alte die Klasse ({int(a.changed.sum())} von {settings.n}). "
            f"Dafür sitzt die Genauigkeit an den Profilen und an der Regel: {right} von {settings.n} richtig"
            + (f", mit um {de(settings.delta, 2)} verschobenen Profilen." if settings.delta else "."))
else:
    st.warning(f"Die Klassen kommen aus der Menge (Quantile): Kommt eine Alternative dazu, wechseln {int(a.changed.sum())} von {settings.n} alten Alternativen die Klasse. Die Abhängigkeit von den anderen "
               "ist nicht verschwunden, sondern in die Herkunft der Profile gewandert.")
if settings.k == 3 and settings.method in ("tri_pess", "tri_opt") and int((a.pess != a.opt).sum()):
    st.caption(f"Pessimistisch und optimistisch ordnen {int((a.pess != a.opt).sum())} Alternativen verschiedenen Klassen zu: Die Regel lässt hier ein Intervall von Klassen, kein Urteil.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), 20 Alternativen, 5 Kriterien, wahre Gewichte, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 30 Sekunden und starten erst auf "
           "Klick. Standardfehler 1,0 bis 2,5 Punkte: Unterschiede unter etwa 5 Punkten sind nicht belastbar.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Wie gut trifft welche Regel")
    st.caption("Anteil richtig zugeordneter Alternativen, 3 Klassen, je Entscheider. In der Rohwert-Welt ist die Summe gegen Profile per Definition richtig (der Entscheider ist auf der "
               "Nenn-Spannweite definiert).")
    st.plotly_chart(build_accuracy(exp["accuracy"], 3), width="stretch", key="accuracy_chart")
    verdict(verdict_accuracy(exp["accuracy"]))
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**3 Klassen**")
        st.dataframe(accuracy_table(exp["accuracy"], 3), hide_index=True, width="stretch")
    with cb:
        st.markdown("**5 Klassen**")
        st.dataframe(accuracy_table(exp["accuracy"], 5), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Hängt die Zuordnung an den anderen Alternativen?")
    st.caption("Eine Alternative kommt dazu (zufällig bzw. als Ausreißer in einem Kriterium): Anteil der alten Alternativen mit Klassenwechsel.")
    verdict(verdict_stability(exp["stability"]))
    st.dataframe(stability_table(exp["stability"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Was die Profile kosten")
    st.caption("Die Profile liegen um δ Güte neben den Grenzen des Entscheiders (Güte/linear, 3 Klassen); darunter, was die Zuordnung an Ordnung hergibt.")
    st.plotly_chart(build_shift(exp["shift"]), width="stretch", key="shift_chart")
    verdict(verdict_price(exp["shift"], exp["resolution"]))
    cc, cd = st.columns(2)
    with cc:
        st.dataframe(shift_table(exp["shift"]), hide_index=True, width="stretch")
    with cd:
        st.dataframe(resolution_table(exp["resolution"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 4: ELECTRE TRI, Schwelle und Veto")
    st.caption("Anteil richtig für den Entscheider mit linearem Güte-Nutzen und den mit Untergrenze, pessimistisch und optimistisch, und wie oft beide verschieden zuordnen (3 Klassen).")
    verdict(verdict_tri(exp["tri"]))
    st.dataframe(tri_table(exp["tri"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Profile passen per Konstruktion** | Der Entscheider definiert seine Klassengrenzen durch den Nutzen genau dieser Profile; echte Profile sind Meinungen. Experiment 3 zeigt, was Verschieben kostet, ersetzt aber keine Erhebung. | Gewichte erheben |
| **Gleichförmige Profile** | Gleiche Güte in jedem Kriterium; echte Profile unterscheiden sich je Kriterium und haben Streuung. | – |
| **FlowSort in einer Lesart** | Grenzprofile als Referenzmenge, V-Form: Aussagen gelten für diese Variante, nicht für die Originalfassung. | – |
| **Ein Veto-Wert, vier Schwellen** | Das Veto ist relativ zum Profil; Schwellen je Kriterium (q, p) sind nicht Teil. | ELECTRE |
| **Der Entscheider ist ein Orakel** | Kein Mensch, der Profile setzt, zögert oder seine Meinung ändert. | – |
| **Absolut gegen relativ** | Dass feste Profile nie von den anderen abhängen, ist eine Eigenschaft; wie man die Profile festlegt, entscheidet, ob die Abhängigkeit zurückkehrt (Experiment 2). | Verfahrensvergleich |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Profile und Klassen.** $k - 1$ Grenzprofile $b_1 < \dots < b_{k-1}$ (Alternativen mit gleicher Güte $g_h$ in jedem Kriterium), Klassen $0, \dots, k - 1$. Die wahre Klasse einer
Alternative $a$ ist $|\{h : u(a) \ge u(b_h)\}|$ mit dem Nutzen $u$ des Entscheiders.

**Summe gegen Profile.** $s(a) = \sum_j w_j\, n_j(a)$ mit der Nenn-Spannweite als Normierung; Klasse $|\{h : s(a) \ge s(b_h)\}|$.

**ELECTRE TRI.** Konkordanz $c(a, b) = \sum_j w_j\, \mathbb{1}[n_j(a) \ge n_j(b)]$; $a\, S\, b$, wenn $c(a, b) \ge \lambda$ und (mit Veto $v$) kein $j$ mit $n_j(b) - n_j(a) > v$.
*Pessimistisch:* $\max\{h : a\, S\, b_h\}$ (sonst 0). *Optimistisch:* der kleinste Index $h$ mit $b_h\, S\, a$ und nicht $a\, S\, b_h$ (die Klasse liegt unter $h$); sonst die oberste Klasse.
Bei $\lambda = 0{,}5$ ohne Veto gilt $a\, S\, b$ oder $b\, S\, a$; pessimistisch und optimistisch sind dann gleich.

**FlowSort (eine Lesart).** $P(d) = \min(1, \max(0, d/p))$; $\Phi(x \mid R) = \frac{1}{|R|} \sum_{r \in R} \sum_j w_j\, [P(n_j(x) - n_j(r)) - P(n_j(r) - n_j(x))]$.
Klasse von $a$: $|\{h : \Phi(a \mid \{b_1, \dots, b_{k-1}\}) \ge \Phi(b_h \mid \{b_i\}_{i \ne h} \cup \{a\})\}|$.

**Feste gegen relative Referenz.** Hängt die Klasse von $a$ nur von $a$ und den Profilen ab, ändert eine neue Alternative keine Klasse. Quantilklassen und Profile aus den Quantilen der Menge
hängen von allen Alternativen ab.

**Literatur:** ELECTRE TRI (Yu; Roy und Mousseau) und FlowSort (Nemery de Bellevue und Lamboray) sind hier nicht im Original geprüft und deshalb nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
