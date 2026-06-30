import json
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Getaround — Dashboard", layout="wide")

DATA_DIR       = os.getenv("DATA_DIR",       "data")
API_URL        = os.getenv("API_URL",        "http://localhost:8000")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))

PRICING_CSV = os.path.join(DATA_DIR, "get_around_pricing_project.csv")
CAR_MAP_CSV = os.path.join(DATA_DIR, "car_id_price_map.csv")

_FEATURES = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color", "car_type",
    "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires",
]


# ── Chargement données ───────────────────────────────────────────────────────

@st.cache_data
def load_delay_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "get_around_delay_analysis.xlsx")
    return pd.read_excel(path, sheet_name="rentals_data")


@st.cache_data
def load_pricing_meta() -> dict:
    """Valeurs uniques des colonnes catégorielles du dataset pricing."""
    df = pd.read_csv(PRICING_CSV, index_col=0)
    return {
        "model_key":   sorted(df["model_key"].dropna().unique().tolist()),
        "fuel":        sorted(df["fuel"].dropna().unique().tolist()),
        "paint_color": sorted(df["paint_color"].dropna().unique().tolist()),
        "car_type":    sorted(df["car_type"].dropna().unique().tolist()),
    }


@st.cache_data(show_spinner="Chargement des prix via l'API…")
def fetch_prices() -> str:
    """
    Charge le fichier d'association car_id→pricing_idx, prédit le prix de chaque
    véhicule unique via /predict/batch, et renvoie un JSON {car_id: prix}.
    Le résultat est mis en cache pour toute la session.
    """
    mapping    = pd.read_csv(CAR_MAP_CSV)
    df_pricing = pd.read_csv(PRICING_CSV, index_col=0)

    unique_idx = mapping["pricing_idx"].unique()
    df_to_pred = df_pricing.loc[unique_idx].copy()
    df_to_pred.index.name = "orig_idx"
    df_to_pred = df_to_pred.reset_index()  # colonne "orig_idx" = pricing_idx original

    idx_to_price: dict[int, float] = {}
    rows_list = list(df_to_pred.itertuples(index=False))

    for start in range(0, len(rows_list), MAX_BATCH_SIZE):
        batch = rows_list[start : start + MAX_BATCH_SIZE]
        vehicles = []
        for row in batch:
            v: dict = {"vehicle_id": str(int(row.orig_idx))}
            for col in _FEATURES:
                val = getattr(row, col)
                v[col] = val.item() if hasattr(val, "item") else val
            vehicles.append(v)
        resp = requests.post(
            f"{API_URL}/predict/batch",
            json={"vehicles": vehicles},
            timeout=30,
        )
        resp.raise_for_status()
        for pred in resp.json()["predictions"]:
            idx_to_price[int(pred["vehicle_id"])] = pred["predicted_price_per_day"]

    # Construire le dict final car_id → prix prédit
    result: dict[str, float] = {}
    for _, row in mapping.iterrows():
        car_id   = str(int(row["car_id"]))
        pidx     = int(row["pricing_idx"])
        result[car_id] = idx_to_price.get(pidx, 0.0)

    return json.dumps(result)


@st.cache_data
def precompute_curve(checkin_type: str, prices_json: str) -> pd.DataFrame:
    """
    Précalcule pour chaque seuil t ∈ [0, 720] (pas 10 min) :
    - nb de réservations bloquées (time_delta < t)
    - CA estimé perdu (somme des prix prédits des résas bloquées)
    """
    prices = json.loads(prices_json)
    df     = load_delay_data()

    df_ended = df[
        (df["checkin_type"] == checkin_type) &
        (df["state"]        == "ended")
    ].copy()
    df_ended["price"] = df_ended["car_id"].astype(str).map(prices).fillna(0.0)

    df_scope = df_ended[
        df_ended["previous_ended_rental_id"].notna() &
        df_ended["time_delta_with_previous_rental_in_minutes"].notna()
    ].copy()

    total    = len(df_ended)
    ca_total = float(df_ended["price"].sum())

    rows = []
    for t in range(0, 721, 10):
        mask     = df_scope["time_delta_with_previous_rental_in_minutes"] < t
        impacted = int(mask.sum())
        ca_perdu = float(df_scope.loc[mask, "price"].sum())
        rows.append({
            "threshold": t,
            "impacted":  impacted,
            "total":     total,
            "pct_resa":  round(impacted / total * 100, 2) if total else 0,
            "ca_perdu":  round(ca_perdu, 0),
            "ca_total":  round(ca_total, 0),
        })
    return pd.DataFrame(rows)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image("getaround_logo.png", use_container_width=True)
st.sidebar.header("Analyse des retards")
st.sidebar.markdown("---")

include_connect = st.sidebar.checkbox("Connect", value=True)
threshold_connect = (
    st.sidebar.slider("Seuil Connect (min)", 0, 720, 30, 10) if include_connect else 0
)

include_mobile = st.sidebar.checkbox("Mobile", value=True)
threshold_mobile = (
    st.sidebar.slider("Seuil Mobile (min)", 0, 720, 60, 10) if include_mobile else 0
)


# ── Onglets ───────────────────────────────────────────────────────────────────

tab_delay, tab_predict = st.tabs(["📊 Analyse des retards", "💰 Prédiction du prix"])


# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Analyse des retards
# ═══════════════════════════════════════════════════════════════════════════════

with tab_delay:
    st.title("Impact d'un délai minimum entre réservations")
    st.markdown(
        "<p style='font-size:18px'>Simule l'effet d'un seuil de délai obligatoire "
        "entre deux locations consécutives sur le chiffre d'affaires et le nombre "
        "de réservations. Les prix sont estimés par le modèle ML pour chaque véhicule.</p>",
        unsafe_allow_html=True,
    )

    if not include_connect and not include_mobile:
        st.warning("Sélectionner au moins un type de checkin dans la sidebar.")
        st.stop()

    # ── Chargement des prix ──────────────────────────────────────────────────

    try:
        prices_json = fetch_prices()
    except Exception as e:
        st.error(
            f"Impossible de contacter l'API ({API_URL}/predict/batch). "
            f"Vérifiez que l'API est démarrée.\n\n`{e}`"
        )
        st.stop()

    prices    = json.loads(prices_json)
    avg_price = sum(prices.values()) / len(prices) if prices else 0

    # ── Précalcul des courbes ────────────────────────────────────────────────

    curve_c = precompute_curve("connect", prices_json) if include_connect else None
    curve_m = precompute_curve("mobile",  prices_json) if include_mobile  else None

    # ── KPIs combinés ────────────────────────────────────────────────────────

    total_ended = 0
    impacted    = 0
    ca_total    = 0.0
    ca_perdu    = 0.0

    if include_connect and curve_c is not None:
        r = curve_c[curve_c["threshold"] == threshold_connect].iloc[0]
        total_ended += int(r["total"])
        impacted    += int(r["impacted"])
        ca_total    += float(r["ca_total"])
        ca_perdu    += float(r["ca_perdu"])

    if include_mobile and curve_m is not None:
        r = curve_m[curve_m["threshold"] == threshold_mobile].iloc[0]
        total_ended += int(r["total"])
        impacted    += int(r["impacted"])
        ca_total    += float(r["ca_total"])
        ca_perdu    += float(r["ca_perdu"])

    pct_resa = impacted / total_ended * 100 if total_ended else 0

    st.subheader("Impact combiné")

    label_seuils = []
    if include_connect:
        label_seuils.append(f"Connect : {threshold_connect} min")
    if include_mobile:
        label_seuils.append(f"Mobile : {threshold_mobile} min")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Réservations totales",  f"{total_ended:,}")
    col2.metric("Réservations bloquées", f"{impacted:,}",
                delta=f"{pct_resa:.1f}%", delta_color="inverse")
    col3.metric("CA estimé total",       f"{ca_total:,.0f} €")
    col4.metric("CA estimé perdu",       f"{ca_perdu:,.0f} €",
                delta=f"-{pct_resa:.1f}%", delta_color="inverse")

    st.caption(
        f"Seuils appliqués — {' · '.join(label_seuils)}  ·  "
        f"Prix moyen prédit : {avg_price:.0f} €/jour "
        f"({len(prices)} véhicules, estimations modèle ML)"
    )

    st.divider()

    # ── Courbes d'impact par type ────────────────────────────────────────────

    st.subheader("Sensibilité au seuil par type de checkin")

    curves_to_show = []
    if include_connect and curve_c is not None:
        curves_to_show.append(("Connect", curve_c, threshold_connect, "steelblue"))
    if include_mobile and curve_m is not None:
        curves_to_show.append(("Mobile", curve_m, threshold_mobile, "darkorange"))

    for label, curve, threshold, color in curves_to_show:
        row_t = curve[curve["threshold"] == threshold].iloc[0]
        st.markdown(f"**{label}** — seuil : {threshold} min")

        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = px.line(
                curve, x="threshold", y="pct_resa",
                title=f"{label} — % réservations bloquées",
                labels={"threshold": "Seuil (min)", "pct_resa": "% bloquées"},
                color_discrete_sequence=[color],
            )
            fig1.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"{threshold} min",
                           annotation_position="top right")
            fig1.add_hline(y=float(row_t["pct_resa"]), line_dash="dot",
                           line_color="orange")
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            fig2 = px.line(
                curve, x="threshold", y="ca_perdu",
                title=f"{label} — CA estimé perdu (€)",
                labels={"threshold": "Seuil (min)", "ca_perdu": "CA perdu (€)"},
                color_discrete_sequence=[color],
            )
            fig2.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"{threshold} min",
                           annotation_position="top right")
            fig2.add_hline(y=float(row_t["ca_perdu"]), line_dash="dot",
                           line_color="orange")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Distribution des retards ─────────────────────────────────────────────

    st.subheader("Distribution des retards au checkout")

    df_raw  = load_delay_data()
    types   = tuple(t for t, f in [("connect", include_connect), ("mobile", include_mobile)] if f)
    df_late = df_raw[
        (df_raw["checkin_type"].isin(types)) &
        (df_raw["state"]       == "ended") &
        (df_raw["delay_at_checkout_in_minutes"] > 0)
    ]

    if df_late.empty:
        st.info("Aucun retard pour les types sélectionnés.")
    else:
        CAP    = 600
        df_viz = df_late[df_late["delay_at_checkout_in_minutes"] <= CAP]

        fig3 = px.histogram(
            df_viz, x="delay_at_checkout_in_minutes", color="checkin_type",
            barmode="overlay", nbins=60, opacity=0.7,
            title=f"Distribution des retards (≤ {CAP} min)",
            labels={"delay_at_checkout_in_minutes": "Retard (min)",
                    "checkin_type": "Type"},
            color_discrete_map={"connect": "steelblue", "mobile": "darkorange"},
        )
        if include_connect:
            fig3.add_vline(x=threshold_connect, line_dash="dash",
                           line_color="steelblue",
                           annotation_text=f"Connect : {threshold_connect} min",
                           annotation_position="top right")
        if include_mobile:
            fig3.add_vline(x=threshold_mobile, line_dash="dash",
                           line_color="darkorange",
                           annotation_text=f"Mobile : {threshold_mobile} min",
                           annotation_position="top left")
        st.plotly_chart(fig3, use_container_width=True)

        stats = df_late.groupby("checkin_type")["delay_at_checkout_in_minutes"].agg(
            count="count",
            mean="mean",
            std="std",
            p50=lambda x: x.quantile(0.50),
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
        ).round(0).astype(int)

        col = df_late["delay_at_checkout_in_minutes"]
        global_row = pd.Series({
            "count": col.count(),
            "mean":  col.mean(),
            "std":   col.std(),
            "p50":   col.quantile(0.50),
            "p75":   col.quantile(0.75),
            "p90":   col.quantile(0.90),
        }, name="global").round(0).astype(int)
        stats = pd.concat([stats, global_row.to_frame().T])

        stats.columns = ["Nb retards", "Moyenne (min)", "Std", "p50 (min)", "p75 (min)", "p90 (min)"]
        stats.index.name = "checkin_type"

        st.markdown(
            "<style>"
            ".stats-table th, .stats-table td { text-align: center !important; padding: 6px 14px; }"
            ".stats-table { width: 100%; border-collapse: collapse; font-size: 14px; }"
            ".stats-table thead tr { border-bottom: 2px solid #e0e0e0; }"
            ".stats-table tbody tr:nth-child(even) { background: #f8f8f8; }"
            "</style>",
            unsafe_allow_html=True,
        )
        st.markdown(stats.to_html(classes="stats-table"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Prédiction du prix d'un véhicule
# ═══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.title("Prédiction du prix de location")
    st.markdown(
        "<p style='font-size:18px'>Estimez le prix journalier optimal d'un véhicule "
        "à partir de ses caractéristiques.</p>",
        unsafe_allow_html=True,
    )

    meta = load_pricing_meta()

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Caractéristiques du véhicule")
            model_key   = st.selectbox("Marque",       meta["model_key"])
            fuel        = st.selectbox("Carburant",    meta["fuel"])
            car_type    = st.selectbox("Type",         meta["car_type"])
            paint_color = st.selectbox("Couleur",      meta["paint_color"])
            mileage     = st.number_input("Kilométrage (km)", min_value=0, value=50000, step=1000)
            engine_power = st.number_input("Puissance (ch)",  min_value=1, value=120,   step=10)

        with col2:
            st.subheader("Équipements")
            private_parking      = st.checkbox("Parking privé")
            has_gps              = st.checkbox("GPS")
            has_air_conditioning = st.checkbox("Climatisation")
            automatic_car        = st.checkbox("Boîte automatique")
            has_getaround_connect = st.checkbox("Getaround Connect")
            has_speed_regulator  = st.checkbox("Régulateur de vitesse")
            winter_tires         = st.checkbox("Pneus hiver")

        submitted = st.form_submit_button("Estimer le prix", use_container_width=True)

    if submitted:
        payload = {
            "model_key":                  model_key,
            "mileage":                    int(mileage),
            "engine_power":               int(engine_power),
            "fuel":                       fuel,
            "paint_color":                paint_color,
            "car_type":                   car_type,
            "private_parking_available":  private_parking,
            "has_gps":                    has_gps,
            "has_air_conditioning":       has_air_conditioning,
            "automatic_car":              automatic_car,
            "has_getaround_connect":      has_getaround_connect,
            "has_speed_regulator":        has_speed_regulator,
            "winter_tires":               winter_tires,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data  = resp.json()
            price = data["predicted_price_per_day"]

            st.success(f"### Prix estimé : **{price:.0f} €/jour**")
            st.caption(f"Modèle utilisé : {data.get('model_info', 'n/a')}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Connexion refusée vers {API_URL}. "
                "Vérifiez que l'API est démarrée (`bash scripts/docker_api.sh`)."
            )
        except Exception as e:
            st.error(f"Erreur API : {e}")
