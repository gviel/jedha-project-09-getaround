# Getaround — Pricing ML & Delay Analysis

**JEDHA Certification RNCP6 CSCD — Bloc 4**

Projet en deux parties : (1) modèle ML pour estimer le prix de location journalier d'un véhicule, (2) analyse des retards de restitution et impact sur le chiffre d'affaires selon le seuil de délai entre réservations.

---

## Architecture

| Composant | Technologie | Rôle |
|---|---|---|
| `Getaround_Pricing_GV.ipynb` | Jupyter | EDA, entraînement XGBoost, suivi MLflow |
| `api/` | FastAPI + Docker | Endpoints `/predict` et `/predict/batch` |
| `dashboard/` | Streamlit + Docker | Analyse délais + prédiction prix |

---

## Installation locale

### Prérequis
- [Conda](https://docs.conda.io/) (Python 3.12)
- Docker (optionnel, pour les tests conteneurisés)

### Environnement

```bash
conda env create -f env_getaround.yml
conda activate getaround
```

### Données

Les fichiers sources sont dans `data/`. Générer le fichier d'association car_id → prix (une seule fois) :

```bash
conda run -n getaround python scripts/generate_car_map.py
```

### Notebook

```bash
conda run -n getaround jupyter notebook
```

Ouvrir `Getaround_Pricing_GV.ipynb`.

---

## Lancer l'API et le dashboard

### Sans Docker (développement)

```bash
# API (depuis api/)
conda run -n getaround uvicorn app:app --reload

# Dashboard (depuis dashboard/)
conda run -n getaround streamlit run app.py
```

### Avec Docker

```bash
# API — mode prod (MLflow + S3) ou test (pkl local)
bash scripts/docker_api.sh              # port 8000, mode prod
bash scripts/docker_api.sh 8000 test   # mode test (pickle local)

# Dashboard — nécessite --network host pour joindre l'API sur localhost
bash scripts/docker_dashboard.sh        # port 8501
```

> **Note :** copier `api/.env_template` → `api/.env` et renseigner les credentials AWS avant de lancer en mode prod.

---

## Déploiement (Render)

Le fichier `render.yaml` définit les deux services Docker (Blueprint Render).

1. Pousser le repo sur GitHub
2. render.com → **New → Blueprint** → pointer sur le repo
3. Dans l'UI Render → service `getaround-api` → **Environment** : saisir `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY`

L'URL de l'API est injectée automatiquement dans le dashboard via `render.yaml`.

---

## Variables d'environnement

### API (`api/.env`)

| Variable | Exemple | Description |
|---|---|---|
| `MODEL_ENV` | `prod` | `prod` = MLflow, `test` = pickle local |
| `MODEL_STATUS` | `best` | Tag MLflow à cibler |
| `MODEL_NAME` | `getaround_pricing` | Nom du modèle dans le registre |
| `MLFLOW_URI` | `https://…` | URL du serveur MLflow |
| `AWS_ACCESS_KEY_ID` | — | Accès au store S3 (mode prod) |
| `AWS_SECRET_ACCESS_KEY` | — | Accès au store S3 (mode prod) |

### Dashboard

| Variable | Défaut | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | URL de l'API FastAPI |
| `MAX_BATCH_SIZE` | `20` | Taille des lots pour `/predict/batch` |

---

## MLflow

Serveur distant : <https://huggingface.co/spaces/gviel/mlflow37>  
Client requis : `mlflow==3.7.0`

Les modèles sont taggés `env=prod` et `status=best|challenger|worst` (calculé automatiquement à l'entraînement).
