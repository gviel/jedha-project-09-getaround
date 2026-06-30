import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Literal

import mlflow.pyfunc
from mlflow.client import MlflowClient
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

MODEL_PATH     = os.getenv("MODEL_PATH",     "models/getaround_pricing_model.pkl")
MLFLOW_URI     = os.getenv("MLFLOW_URI",     "https://gviel-mlflow37.hf.space/")
MODEL_NAME     = os.getenv("MODEL_NAME",     "getaround_pricing_lr")
MODEL_ENV      = os.getenv("MODEL_ENV",      "prod")    # prod | staging | test
MODEL_STATUS   = os.getenv("MODEL_STATUS",   "best")    # best | challenger | worst
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))

model = None
_model_info = ""


def load_model():
    """
    MODEL_ENV=test  → chargement local depuis MODEL_PATH (joblib)
    MODEL_ENV=prod|staging → MLflow client, filtre sur tags env + status
    """
    global model, _model_info

    if MODEL_ENV == "test":
        model = joblib.load(MODEL_PATH)
        _model_info = f"local:{MODEL_PATH}"
        print(f"Modèle chargé localement : {MODEL_PATH}")
        return

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    filter_str = (
        f"name='{MODEL_NAME}'"
        f" and tags.env = '{MODEL_ENV}'"
        f" and tags.status = '{MODEL_STATUS}'"
    )
    versions = client.search_model_versions(filter_str)
    if not versions:
        raise RuntimeError(
            f"Aucune version de '{MODEL_NAME}' avec les tags "
            f"env={MODEL_ENV}, status={MODEL_STATUS} sur {MLFLOW_URI}"
        )

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    uri = f"models:/{MODEL_NAME}/{latest.version}"
    model = mlflow.pyfunc.load_model(uri)
    _model_info = f"{MODEL_NAME} v{latest.version} [env={MODEL_ENV}, status={MODEL_STATUS}]"
    print(f"Modèle chargé depuis MLflow : {_model_info}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Getaround Pricing API",
    description="""
## Prédiction du prix de location journalier d'un véhicule Getaround

Modèle ML (scikit-learn Pipeline) chargé depuis MLflow ou en local (mode test).

### Endpoints
- **POST `/predict`** — prix estimé pour un véhicule
- **POST `/predict/batch`** — prix estimés pour une liste de véhicules (max `MAX_BATCH_SIZE`, défaut 20)
- **GET `/`** — health check
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas ──────────────────────────────────────────────────────────────────

_FEATURES = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color", "car_type",
    "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires",
]


class VehicleFeatures(BaseModel):
    vehicle_id: str | None = Field(
        None,
        description="Identifiant optionnel du véhicule, renvoyé tel quel dans la réponse",
        examples=["car_42"],
    )
    model_key: str = Field(..., description="Marque du véhicule", examples=["Peugeot"])
    mileage: float = Field(..., ge=0,  description="Kilométrage (km)",       examples=[50000])
    engine_power: float = Field(..., gt=0, description="Puissance moteur (ch)", examples=[120])
    fuel: Literal["diesel", "petrol", "hybrid_petrol", "electro"] = Field(
        ..., description="Type de carburant", examples=["diesel"],
    )
    paint_color: Literal["black", "grey", "white", "red", "silver", "blue", "orange", "beige", "brown", "green"] = Field(
        ..., description="Couleur de la carrosserie", examples=["white"],
    )
    car_type: Literal["convertible", "coupe", "estate", "hatchback", "sedan", "subcompact", "suv", "van"] = Field(
        ..., description="Type de carrosserie", examples=["sedan"],
    )
    private_parking_available: bool = Field(..., description="Parking privé disponible", examples=[True])
    has_gps: bool                   = Field(..., description="GPS intégré",               examples=[True])
    has_air_conditioning: bool      = Field(..., description="Climatisation",              examples=[True])
    automatic_car: bool             = Field(..., description="Boîte automatique",          examples=[False])
    has_getaround_connect: bool     = Field(..., description="Getaround Connect",          examples=[True])
    has_speed_regulator: bool       = Field(..., description="Régulateur de vitesse",      examples=[False])
    winter_tires: bool              = Field(..., description="Pneus hiver",                examples=[False])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vehicle_id": "car_42",
                    "model_key": "Peugeot",
                    "mileage": 50000,
                    "engine_power": 120,
                    "fuel": "diesel",
                    "paint_color": "white",
                    "car_type": "sedan",
                    "private_parking_available": True,
                    "has_gps": True,
                    "has_air_conditioning": True,
                    "automatic_car": False,
                    "has_getaround_connect": True,
                    "has_speed_regulator": False,
                    "winter_tires": False,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    vehicle_id: str | None = Field(None,  description="Identifiant du véhicule, si fourni en entrée")
    predicted_price_per_day: float = Field(..., description="Prix estimé en €/jour")
    currency: str   = Field(default="EUR")
    model_info: str = Field(default="")


class BatchPredictionRequest(BaseModel):
    vehicles: list[VehicleFeatures] = Field(
        ...,
        description=f"Liste de véhicules à évaluer (max {MAX_BATCH_SIZE})",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vehicles": [
                        {
                            "vehicle_id": "car_1",
                            "model_key": "Peugeot", "mileage": 50000, "engine_power": 120,
                            "fuel": "diesel", "paint_color": "white", "car_type": "sedan",
                            "private_parking_available": True, "has_gps": True,
                            "has_air_conditioning": True, "automatic_car": False,
                            "has_getaround_connect": True, "has_speed_regulator": False,
                            "winter_tires": False,
                        },
                        {
                            "vehicle_id": "car_2",
                            "model_key": "Renault", "mileage": 80000, "engine_power": 90,
                            "fuel": "petrol", "paint_color": "grey", "car_type": "hatchback",
                            "private_parking_available": False, "has_gps": False,
                            "has_air_conditioning": True, "automatic_car": False,
                            "has_getaround_connect": False, "has_speed_regulator": False,
                            "winter_tires": False,
                        },
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        ..., description="Prix estimés dans l'ordre de la requête, avec vehicle_id si fourni",
    )
    count: int      = Field(..., description="Nombre de véhicules traités")
    model_info: str = Field(default="")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check."""
    return {"status": "ok", "model_loaded": model is not None, "model_info": _model_info}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Prédit le prix de location journalier",
)
def predict(vehicle: VehicleFeatures):
    """
    Retourne le prix de location estimé en €/jour pour un véhicule donné.

    **Exemple de requête :**
    ```json
    {
      "model_key": "Peugeot",
      "mileage": 50000,
      "engine_power": 120,
      "fuel": "diesel",
      "paint_color": "white",
      "car_type": "sedan",
      "private_parking_available": true,
      "has_gps": true,
      "has_air_conditioning": true,
      "automatic_car": false,
      "has_getaround_connect": true,
      "has_speed_regulator": false,
      "winter_tires": false
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    row = pd.DataFrame([vehicle.model_dump(include=set(_FEATURES))])
    price = max(round(float(model.predict(row)[0]), 2), 0.0)

    return PredictionResponse(
        vehicle_id=vehicle.vehicle_id,
        predicted_price_per_day=price,
        model_info=_model_info,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary=f"Prédit le prix de location pour une liste de véhicules (max {MAX_BATCH_SIZE})",
)
def predict_batch(request: BatchPredictionRequest):
    """
    Retourne les prix estimés en €/jour pour une liste de véhicules.

    Les véhicules sont traités en une seule inférence (un seul appel `model.predict`).
    Les résultats sont retournés dans le même ordre que la requête.
    Le champ `vehicle_id` est renvoyé tel quel s'il a été fourni en entrée.

    Limite configurable via la variable d'environnement `MAX_BATCH_SIZE` (défaut : 20).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(request.vehicles) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Trop de véhicules : {len(request.vehicles)} > MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    df = pd.DataFrame([v.model_dump(include=set(_FEATURES)) for v in request.vehicles])
    raw_prices = model.predict(df)

    predictions = [
        PredictionResponse(
            vehicle_id=v.vehicle_id,
            predicted_price_per_day=max(round(float(p), 2), 0.0),
            model_info=_model_info,
        )
        for v, p in zip(request.vehicles, raw_prices)
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_info=_model_info,
    )
