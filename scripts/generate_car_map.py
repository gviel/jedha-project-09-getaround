#!/usr/bin/env python3
"""
Génère data/car_id_price_map.csv : association déterministe entre les car_id
du dataset de retards et les indices de véhicules du dataset de prix.

Contrainte :
  - si un car_id a au moins une location 'connect', il est associé à un
    véhicule du dataset pricing ayant has_getaround_connect == True.
  - sinon, il est associé à n'importe quel véhicule valide du dataset pricing.

Utilisation (depuis la racine du projet) :
  conda run -n getaround python scripts/generate_car_map.py
"""
import os
import pandas as pd

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELAY_FILE   = os.path.join(ROOT, "data", "get_around_delay_analysis.xlsx")
PRICING_FILE = os.path.join(ROOT, "data", "get_around_pricing_project.csv")
OUT_FILE     = os.path.join(ROOT, "data", "car_id_price_map.csv")

df_delay   = pd.read_excel(DELAY_FILE, sheet_name="rentals_data")
df_pricing = pd.read_csv(PRICING_FILE, index_col=0)

# Filtrer les véhicules invalides (même filtre que l'API)
df_valid = df_pricing[(df_pricing["mileage"] >= 0) & (df_pricing["engine_power"] > 0)]

connect_idx = df_valid[df_valid["has_getaround_connect"] == True].index.tolist()
all_idx     = df_valid.index.tolist()

# Pour chaque car_id, détecter si au moins une location est 'connect'
car_has_connect = (
    df_delay.groupby("car_id")["checkin_type"]
    .apply(lambda x: "connect" in x.values)
    .rename("is_connect")
)

rows = []
for i, (car_id, is_connect) in enumerate(car_has_connect.items()):
    pool       = connect_idx if is_connect else all_idx
    pricing_idx = pool[i % len(pool)]
    rows.append({
        "car_id":      int(car_id),
        "pricing_idx": int(pricing_idx),
        "is_connect":  bool(is_connect),
    })

df_map = pd.DataFrame(rows)
df_map.to_csv(OUT_FILE, index=False)

print(f"Mapping généré : {len(df_map)} car_id → {OUT_FILE}")
print(f"  avec connect : {df_map['is_connect'].sum()} (pool : {len(connect_idx)} véhicules)")
print(f"  mobile only  : {(~df_map['is_connect']).sum()} (pool : {len(all_idx)} véhicules)")
print(f"  pricing_idx uniques à prédire : {df_map['pricing_idx'].nunique()}")
