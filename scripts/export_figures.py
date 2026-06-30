#!/usr/bin/env python3
"""
Génère et exporte tous les graphiques du notebook Getaround_Pricing_GV.ipynb
dans le répertoire exports/ au format PNG (statique) et HTML (interactif).

Usage (depuis la racine du projet) :
    conda run -n getaround python scripts/export_figures.py
"""

import os
import warnings
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS  = os.path.join(ROOT, "exports")
DATA_DIR = os.path.join(ROOT, "data")

os.makedirs(EXPORTS, exist_ok=True)


def save(fig, name: str):
    """Sauvegarde fig en PNG et HTML dans exports/."""
    png_path  = os.path.join(EXPORTS, f"{name}.png")
    html_path = os.path.join(EXPORTS, f"{name}.html")
    fig.write_image(png_path, width=1100, height=550, scale=2)
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"  ✓ {name}.png / .html")


print("── Chargement des données pricing ──")
df_pricing = pd.read_csv(os.path.join(DATA_DIR, "get_around_pricing_project.csv"), index_col=0)
# Nettoyage identique au notebook
df_model = df_pricing[(df_pricing["mileage"] >= 0) & (df_pricing["engine_power"] > 0)].copy()

print("── Partie 1 — Pricing ──")

# 1. Distribution du prix de location / jour
fig = px.histogram(
    df_model, x="rental_price_per_day", nbins=50,
    title="Distribution du prix de location / jour",
    labels={"rental_price_per_day": "Prix (€/jour)"},
)
save(fig, "01_pricing_distribution_prix")

# 2. Prix par type de véhicule
fig = px.box(
    df_model, x="car_type", y="rental_price_per_day",
    title="Prix par type de véhicule", points="outliers",
    labels={"rental_price_per_day": "Prix (€/jour)", "car_type": "Type"},
)
save(fig, "02_pricing_prix_par_type")

# 3. Prix par type de carburant
fig = px.box(
    df_model, x="fuel", y="rental_price_per_day",
    title="Prix par type de carburant", points="outliers",
    labels={"rental_price_per_day": "Prix (€/jour)", "fuel": "Carburant"},
)
save(fig, "03_pricing_prix_par_carburant")

# 4. Kilométrage vs prix
fig = px.scatter(
    df_model, x="mileage", y="rental_price_per_day",
    color="fuel", opacity=0.5,
    title="Kilométrage vs prix de location",
    labels={"mileage": "Kilométrage (km)", "rental_price_per_day": "Prix (€/jour)"},
)
save(fig, "04_pricing_kilometrage_vs_prix")

# 5. Puissance moteur vs prix
fig = px.scatter(
    df_model, x="engine_power", y="rental_price_per_day",
    color="car_type", opacity=0.5,
    title="Puissance moteur vs prix de location",
    labels={"engine_power": "Puissance (ch)", "rental_price_per_day": "Prix (€/jour)"},
)
save(fig, "05_pricing_puissance_vs_prix")

print("\n── Partie 2 — Analyse des retards ──")
df_delay = pd.read_excel(
    os.path.join(DATA_DIR, "get_around_delay_analysis.xlsx"),
    sheet_name="rentals_data",
)

# Filtre : locations non annulées
df_ended = df_delay[df_delay["state"] != "canceled"].copy()

# Jointure A (retard) ← B (location suivante)
df_B = df_ended[df_ended["previous_ended_rental_id"].notna()][[
    "previous_ended_rental_id",
    "time_delta_with_previous_rental_in_minutes",
    "checkin_type",
]].rename(columns={
    "previous_ended_rental_id":                  "rental_id",
    "time_delta_with_previous_rental_in_minutes": "B_time_delta",
    "checkin_type":                              "B_checkin_type",
})
df_A = df_ended[df_ended["delay_at_checkout_in_minutes"] > 0].copy()
df_impact = df_A.merge(df_B, on="rental_id", how="inner")
df_impact["caused_problem"] = (
    df_impact["delay_at_checkout_in_minutes"] >= df_impact["B_time_delta"]
)
df_caused = df_impact[df_impact["caused_problem"]].copy()

# 6. Répartition connect / mobile des retards impactants
counts  = df_caused["checkin_type"].value_counts()
pcts    = df_caused["checkin_type"].value_counts(normalize=True).mul(100).round(1)
summary = pd.DataFrame({"nb": counts, "pct_%": pcts}).reset_index()
summary.columns = ["checkin_type", "nb", "pct_%"]
summary["text"] = summary["pct_%"].astype(str) + " %"

fig = px.bar(
    summary, x="checkin_type", y="nb",
    color="checkin_type",
    text="text",
    title="Retards de A ayant impacté B — connect vs mobile",
    labels={"checkin_type": "Type", "nb": "Nb locations"},
    color_discrete_map={"connect": "steelblue", "mobile": "darkorange"},
)
fig.update_traces(texttemplate="%{text}", textposition="outside")
save(fig, "06_retards_connect_vs_mobile")

# 7. Boxplot des retards impactants
CAP = 600
df_viz = df_caused[df_caused["delay_at_checkout_in_minutes"] <= CAP].copy()

fig = px.box(
    df_viz, x="checkin_type", y="delay_at_checkout_in_minutes",
    color="checkin_type", points="outliers",
    title=f"Distribution des retards impactants par type (≤ {CAP} min)",
    labels={"delay_at_checkout_in_minutes": "Retard (min)", "checkin_type": "Type"},
    color_discrete_map={"connect": "steelblue", "mobile": "darkorange"},
)
save(fig, "07_retards_boxplot_par_type")

# 8. Histogramme des retards impactants
fig = px.histogram(
    df_viz, x="delay_at_checkout_in_minutes",
    color="checkin_type", barmode="overlay",
    nbins=60, opacity=0.7,
    title=f"Distribution des retards impactants (≤ {CAP} min)",
    labels={"delay_at_checkout_in_minutes": "Retard (min)", "checkin_type": "Type"},
    color_discrete_map={"connect": "steelblue", "mobile": "darkorange"},
)
save(fig, "08_retards_histogram_par_type")

# 9. Courbe percentile → délai optimal
percentiles = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
rows = []
for ct, grp in df_caused.groupby("checkin_type"):
    for p in percentiles:
        val = grp["delay_at_checkout_in_minutes"].quantile(p)
        rows.append({"checkin_type": ct, "percentile": f"p{int(p*100)}", "délai_min": round(val)})

fig = px.line(
    pd.DataFrame(rows), x="percentile", y="délai_min",
    color="checkin_type", markers=True,
    title="Délai nécessaire pour couvrir X% des retards impactants (connect vs mobile)",
    labels={"délai_min": "Délai minimum (min)", "percentile": "Percentile", "checkin_type": "Type"},
    color_discrete_map={"connect": "steelblue", "mobile": "darkorange"},
)
save(fig, "09_retards_percentiles_delai_optimal")

print(f"\n✅ {len(os.listdir(EXPORTS))//2} graphiques exportés dans {EXPORTS}/")
