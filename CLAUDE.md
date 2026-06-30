# Getaround — CLAUDE.md

## Projet
JEDHA Certification RNCP6 CSCD — Bloc 4  
Specs complètes : `specs.md`

## Structure
```
Project_09_Getaround/
├── Getaround_Pricing_GV.ipynb   ← notebook EDA + entraînement ML
├── data/                        ← fichiers source
│   ├── get_around_pricing_project.csv
│   ├── get_around_delay_analysis.xlsx
│   └── car_id_price_map.csv     ← association car_id (delay) → pricing_idx (généré par scripts/generate_car_map.py)
├── models/                      ← modèle local (mode test)
│   └── getaround_pricing_model.pkl
├── api/                         ← FastAPI → Render (Docker)
│   ├── app.py
│   ├── requirements.txt
│   ├── .env                     ← credentials (non versionné)
│   ├── .env_template            ← template sans valeurs sensibles
│   └── Dockerfile               ← FROM python:3.12-slim, port $PORT (défaut 8000)
├── dashboard/                   ← Streamlit → Render (Docker)
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile               ← FROM python:3.12-slim, port $PORT (défaut 8501)
├── env_getaround.yml            ← conda local (tout-en-un, Python 3.12)
└── scripts/                     ← outils manipulation notebook + Docker
    ├── generate_car_map.py      ← génère data/car_id_price_map.csv
    ├── docker_api.sh            ← build + run API (mode prod par défaut)
    └── docker_dashboard.sh      ← build + run dashboard (--network host)
```

## Données
| Fichier | Contenu |
|---|---|
| `data/get_around_pricing_project.csv` | 4 843 véhicules · 14 features · target : `rental_price_per_day` |
| `data/get_around_delay_analysis.xlsx` onglet `rentals_data` | 21 310 locations · checkin type, délai retard |
| `data/get_around_delay_analysis.xlsx` onglet `Documentation` | Descriptif des champs |
| `data/car_id_price_map.csv` | Association déterministe car_id (delay) → pricing_idx ; contrainte connect voir specs.md |

## Environnement local
```bash
conda activate getaround   # Python 3.12
conda run -n getaround jupyter notebook
```

## MLFlow distant
URL : https://huggingface.co/spaces/gviel/mlflow37  
Version client requise : `mlflow==3.7.0`

## Déploiement (Render)
Les deux composants utilisent Docker (`python:3.12-slim`) et lisent `$PORT` au démarrage.
La configuration de déploiement est dans `render.yaml` (Blueprint Render).

```bash
# Générer le fichier d'association car_id (une seule fois)
conda run -n getaround python scripts/generate_car_map.py

# Build et test local API (context = racine du projet)
bash scripts/docker_api.sh              # port 8000, MODEL_ENV=prod (via api/.env)
bash scripts/docker_api.sh 8001         # port personnalisé
bash scripts/docker_api.sh 8000 test    # mode test (pkl local, pas de MLflow/S3)

# Build et test local Dashboard
# ⚠️  --network host requis pour joindre l'API sur localhost
bash scripts/docker_dashboard.sh        # port 8501
bash scripts/docker_dashboard.sh 8502   # port personnalisé
```

### Déploiement prod sur Render
1. Pousser le repo sur GitHub
2. render.com → **New → Blueprint** → pointer sur le repo → Render lit `render.yaml`
3. Dans l'UI Render → service `getaround-api` → **Environment** :
   - Saisir `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY` (variables secrètes)
4. `API_URL` du dashboard est injecté automatiquement via `fromService` dans `render.yaml`
5. ⚠️ Plan gratuit : cold start après 15 min d'inactivité (modèle rechargé depuis MLflow/S3)

## Scripts notebook
```bash
python scripts/nb_inspect.py Getaround_Pricing_GV.ipynb --headers
python scripts/nb_cat.py Getaround_Pricing_GV.ipynb <idx>
python scripts/nb_grep.py Getaround_Pricing_GV.ipynb "<pattern>"
```
Voir `scripts/README.md` pour la liste complète.
