# Zuordnung statt Rangfolge: vergleichen gegen feste Profile – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-zuordnung-profile-demo.streamlit.app/)**

Zehntes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind der
[Rangumkehr](https://github.com/sebastian-hanisch/rangumkehr-demo). Statt Alternativen untereinander zu ordnen, wird jede einzeln gegen feste **Grenzprofile** einer von k geordneten
**Klassen** zugeordnet: mit der gewichteten Summe, mit **ELECTRE TRI** (Konkordanz, Veto, pessimistisch und optimistisch) oder mit **FlowSort** (Fluss gegen die Profile).

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seinen Nutzen
(linear, konkav oder Cobb-Douglas in der Güte, linear in den Rohwerten, oder mit Untergrenze); seine Klassengrenzen sind der Nutzen der Profile. Er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**Eine Zuordnung gegen feste Profile hängt nicht von den anderen Alternativen ab, aber sie setzt Profile und eine Zuordnungsregel voraus.** Die Kernfrage der Demo: Was kostet „gegen feste
Profile“, und kehrt die Abhängigkeit zurück, wenn die Profile aus der Menge kommen?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Hypothese „die Rangumkehr verschwindet“ stimmt nur für feste Profile.** Dort wechselt keine alte Alternative die Klasse, wenn eine dazukommt (0,00 %, Identität). Kommen Klassen oder
  Profile aus der Menge (Quantile), wechseln wieder 4,4 bis 4,9 % der alten Alternativen, in 56 bis 62 % der Matrizen.
- **Zuordnung ist nicht automatisch besser als Rangfolge.** Die Summe gegen Profile trifft bei 3 Klassen 84,0 % (Entscheider linear in der Güte), ELECTRE TRI pessimistisch 70,0 %: Die
  Aggregationsregel ist eine Annahme über den Entscheider.
- **Das Veto gehört zum Entscheider mit Untergrenze:** +10,8 Punkte dort (74,8 gegen 64,0 %, λ = 0,5), beim kompensatorischen Entscheider 8 bis 17,5 Punkte weniger.
- **FlowSort ist gleichauf mit der Summe** (höchstens 1,6 Punkte Abstand bei allen fünf Entscheidern). Die Variante hier ist **eine** Lesart (Grenzprofile als Referenzmenge) und nicht belegt
  als die Originalfassung.
- **Profile kosten:** Liegen sie um 0,05 Güte neben den Grenzen des Entscheiders, sinkt die Summe gegen Profile von 84,0 auf 68,6 bis 84,1 %, bei 0,10 auf 54,1 bis 71,5 %. 3 statt 5 Klassen
  heißt 84,0 statt 73,6 %.
- **Die Ordnung innerhalb der Klasse geht verloren:** Bei 3 Klassen stehen 36,2 % der Paare in derselben Klasse; die aufgelösten sind zu 97,7 % richtig geordnet.
- **Die Experimente starten auf Klick** (rund 30 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40, Standard 12), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; eine weitere Alternative (die (n + 1)-te Zeile) für den Klassenwechsel.
- **Profile:** k − 1 Alternativen mit gleicher Güte in jedem Kriterium (3 Klassen: 0,44 / 0,56; 5 Klassen: 0,38 / 0,47 / 0,53 / 0,62), Rohwerte aus dem Generator; um δ verschiebbar.
- **Wahre Klasse:** Zahl der Profile (an den Grenzen des Entscheiders), deren Nutzen sein Nutzen der Alternative erreicht; Untergrenze: ein Kriterium unter Güte 0,1 gibt die niedrigste Klasse.
- **Verfahren:** Summe gegen Profile (Nenn-Spannweite als Skala); ELECTRE TRI (Konkordanz ≥ λ, Veto, pessimistisch / optimistisch); FlowSort (V-Form p = 0,5, Nettofluss gegen die Profile);
  Quantilklassen (relativ); Summe gegen Profile aus den Quantilen der Menge (relativ).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), 20 Alternativen, 5 Kriterien, wahre Gewichte, unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Regeln:** Vergleiche zwischen Regeln sind gepaart.
2. **Messgröße ist die Zuordnung** (Anteil richtig, mittlerer Klassenabstand), nicht eine Rangfolge.
3. **Absolut gegen relativ:** feste Profile gegen Klassen und Profile aus der Menge, Klassenwechsel beim Hinzufügen (zufällig und als Ausreißer in einem Kriterium).
4. **Preis der Profile:** Verschiebung, Klassenzahl, aufgelöste Paare.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, 20 Alternativen, 5 Kriterien, wahre Gewichte. **Der Standardfehler eines Anteils ist 1,0 bis 2,5 Punkte; Unterschiede unter etwa
5 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Anteil richtig, 3 Klassen (Güte/linear, konkav, Cobb-Douglas, Rohwerte, Untergrenze) | **Summe gegen Profile 84,0 / 88,1 / 80,3 / 100,0 / 76,7 %** (Rohwerte: der Entscheider ist auf der Nenn-Spannweite definiert, Tautologie); **FlowSort 85,4 / 87,1 / 79,3 / 98,4 / 75,9 %**; ELECTRE TRI pessimistisch 70,0 / 73,0 / 71,9 / 72,1 / 65,1 %, optimistisch 70,5 / 64,5 / 59,0 / 64,5 / 60,0 %; **TRI pessimistisch mit Veto 0,3: 52,5 / 61,9 / 71,2 / 61,2 / 71,4 %**; Quantilklassen 76,6 / 76,3 / 68,7 / 79,1 / 66,9 %; Summe gegen Profile aus den Quantilen 62,3 / 62,5 / 58,0 / 63,0 / 52,2 %. | `test_share_assigned_correctly`, `test_the_raw_value_decider_makes_the_sum_exactly_right` |
| Dasselbe mit 5 Klassen | Summe 73,6 / 78,0 / 64,9 / 100,0 / 65,5 %; FlowSort 75,6 / 76,6 / 64,0 / 96,5 / 64,5 %; TRI pessimistisch 53,6 / 56,8 / 57,6 / 55,9 / 52,8 %; Quantilklassen 59,3 / 56,6 / 46,5 / 60,8 / 48,8 %; Profile aus Quantilen 34,1 / 33,2 / 28,0 / 33,4 / 24,5 %. | `test_share_assigned_correctly` |
| Eine Alternative kommt dazu: Klassenwechsel der alten (zufällig) / Matrizen mit Wechsel / bei Ausreißer | Feste Profile (Summe, TRI beide Modi, FlowSort): **0,00 / 0,0 / 0,00 %**; **Quantilklassen 4,88 / 62,0 / 2,15 %; Profile aus Quantilen 4,38 / 56,5 / 3,95 %.** | `test_fixed_profiles_never_change_class`, `test_quantile_classes_and_profiles_from_the_data_bring_the_dependence_back` |
| Profile um δ Güte verschoben (3 Klassen, Güte/linear): Summe / TRI pessimistisch / FlowSort; Anteil oberste Klasse (Summe) | δ = −0,10: 71,5 / 70,2 / 71,2 % (51,8 %); −0,05: 84,1 / 71,5 / 84,4 % (41,4 %); **0: 84,0 / 70,0 / 85,4 % (31,7 %)**; +0,05: 68,6 / 65,7 / 69,8 % (23,2 %); +0,10: 54,1 / 60,8 / 55,1 % (15,7 %); wahre oberste Klasse 38,4 %. | `test_shifted_profiles_cost_accuracy` |
| Was die Zuordnung an Ordnung hergibt (Summe gegen Profile) | 3 Klassen: Paare in verschiedenen Klassen **63,8 %**, davon richtig geordnet 97,7 %; 5 Klassen: 77,0 %, davon 97,0 %. | `test_the_order_inside_a_class_is_lost` |
| ELECTRE TRI: λ und Veto (pessimistisch / optimistisch richtig für Güte/linear; dasselbe für Untergrenze; pessimistisch ≠ optimistisch) | λ = 0,5 ohne Veto: 73,1 / 73,1; 64,0 / 64,0; **0,0 %** (Identität); λ = 0,6: 70,0 / 70,5; 65,1 / 60,0; 26,8 %; λ = 0,7: 62,8 / 63,1; 63,9 / 52,7; 51,7 %; λ = 0,8: 52,7 / 54,0; 61,4 / 43,9; 72,2 %; **mit Veto 0,3 bei λ = 0,5: 55,9 / 61,6; 74,8 / 50,5; 60,4 %.** | `test_tri_threshold_and_veto`, `test_without_veto_and_lambda_one_half_pessimistic_equals_optimistic` |

## Ehrliche Grenzen

- **Die Profile passen per Konstruktion.** Der Entscheider definiert seine Klassengrenzen durch den Nutzen genau dieser Profile; echte Profile sind Meinungen. Die Verschiebung zeigt den Preis,
  ersetzt aber keine Erhebung der Profile.
- **Gleichförmige Profile** (gleiche Güte in jedem Kriterium); echte Profile unterscheiden sich je Kriterium und haben Streuung.
- **FlowSort in einer Lesart,** ohne die Originalfassung geprüft; Aussagen über „FlowSort“ gelten für diese Variante.
- **Ein Veto-Wert und vier λ.** Das Veto ist relativ zum Profil; Schwellen q, p je Kriterium sind nicht Teil.
- **Kleine Unterschiede sind Rauschen.** Summe gegen FlowSort (höchstens 1,6 Punkte) und die Spalten der Güte-Entscheider liegen innerhalb von zwei Standardfehlern. Quellenangaben nur aus
  Hintergrundwissen: ELECTRE TRI (Yu; Roy und Mousseau) und FlowSort (Nemery de Bellevue und Lamboray) sind hier nicht im Original geprüft.

## Verifikation

- **Kern:** Profile, wahre Klasse, Summe gegen Profile, ELECTRE TRI (beide Modi, Schwelle, Veto), FlowSort und Quantilklassen gegen Handrechnung auf der Mini-Instanz (`tests/conftest.py`:
  drei Alternativen und zwei Profile mit Summen 0,72 / 0,5 / 0,3 gegen 0,4 / 0,7, Konkordanzen 0,8, Veto, Flüsse 0,38 / −0,10 / −0,50).
- **Orakel** (`tests/test_oracle_zuordnung.py`): feste Profile ändern nie eine Klasse beim Hinzufügen (Gegenprobe: Quantilklassen und Profile aus der Menge ändern sie); Rohwert-Entscheider gleich
  Summe gegen Profile; λ = 0,5 ohne Veto: pessimistisch gleich optimistisch; Monotonie; ELECTRE TRI und FlowSort gegen Definitionsschleifen.
- **Randfälle** (`tests/test_edge_cases.py`): n = 1 und 2, Alternative genau auf dem Profil, Konkordanz genau auf der Schwelle, 2 Klassen, lauter gleiche Alternativen, Veto 0.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an von Hand gesetzten Zeilen in `test_evaluation.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jedes Verfahren, jeder Entscheider, Klassenzahl, Verschiebung, deaktivierte Regler,
  Permalink, PDF und Experimente erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `zpr_constants.py` | Regler-Grenzen, Klassen und Profile, Verfahren, Farben, Presets |
| `zpr_score.py` | Nenn-Spannweite, Normierung, Kendalls τ-b |
| `zpr_profiles.py` | Profile (Güte → Rohwerte), Nutzen und wahre Klasse des Entscheiders |
| `zpr_sorting.py` | Summe gegen Profile, ELECTRE TRI, FlowSort, Quantilklassen, Profile aus den Quantilen |
| `zpr_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `zpr_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `zpr_visualization.py` | Streifendiagramm mit Klassengrenzen, Tabellen, Plotly-Abbildungen (Achsen gesperrt) |
| `zpr_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, Zuordnungsregeln, Profile, Score, Orakel, Randfälle, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Profile je Kriterium und mit Streuung:** gleichförmige Profile, ein Regler für die Verschiebung.
- **Schwellen q, p je Kriterium und gemischte Präferenzfunktionen:** ein Regler für alle Kriterien.
- **ELECTRE TRI mit Unschärfe (λ-Intervalle, ELECTRE TRI-nC) und FlowSort-Varianten mit zentralen Profilen:** eine Lesart je Verfahren.
- **Profile aus Beispielen lernen:** die Profile sind gegeben; das Lernen steht in einem eigenen Stück der Linie.
- **Vergleich über viele Verfahren:** hier nur Zuordnungsregeln; der Vergleich steht im letzten Stück der Linie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
