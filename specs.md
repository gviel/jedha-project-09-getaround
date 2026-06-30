# JEDHA - Project Get Around - bloc 4

Certification RNCP6 CSCD

---

Enoncé du projet dans : 01-Getaround_analysis.ipynb

Fichiers :
- data/get_around_pricing_project.csv : contient les données des véhicules avec leurs caractéristiques et leurs prix de location
- data/get_around_delay_analysis.xlsx :
  - onglet rentals_data : contient un historique des locations qui permet d'analyser les délais de retard
  - onglet Documentation : contient le descriptif des champs de l'onglet rental_data

---

## Structure du projet

Le projet contient plusieurs sous-projets :
- notebook @GetAround_Pricing_GV.ipynb : EDA + modélisation ML
- dans le dossier @api : API FastAPI de prédicition de prix de location et estimation de CA 
- dans le dosser @dashbaord : un dashboard streamlit qui permet de prédire un prix de location de véhicule + l'impact sur le CA de l'ajout d'un délai entre locations

Déploiement :
- on ne fera le déploiement et tests qu'en local pour l'instant; on verra la prod plus tard
- le déploiement en prod se fera sur Render (par contre se désactive au bout de 15min)

---

## Partie 1 : ML pour optimiser le prix des locations

dans le notebook avec le fichier `get_around_pricing_project.csv`
- analyses de valeurs null et outliers :
    - regarder notamment les anomalies / outliers pour kilométrage (a priori un seul véhicule avec -64km à exclure du modèle)
    - regarde si d'autres problèmes susceptibles de poser problème pour l'analyse comme des valeurs à 0 (ex: puissance moteur=0)
    - on excluera ces valeurs pour avoir un dataset propre
- puis tester plusieurs modèles
    - un modèle de régression linéaire avec encodage des valeurs pour prédire au mieux le prix d'un véhicule en fonction de ses caractéristiques
        - variante mais en enlevant les outliers sur les valeurs numériques `mileage` et `engine_power` et `rental_price_per_day`
        - variante avec lasso puis ridge    
    - essaye un modèle XGBoost (type RandomForest)
    - conclure sur le choix du modèle
- il faudra envoyer le modèle
    - pour la prod : vers un serveur MLFlow https://huggingface.co/spaces/gviel/mlflow37 en version 3.7.0 ; variables de config dans .env
    - pour les tests en local : mettre les modèles dans un fichier en local
- **Convention de tags MLflow sur les versions de modèle** (2 tags obligatoires) :

    | Tag | Valeurs possibles | Qui le pose |
    |-----|-------------------|-------------|
    | `env` | `prod` \| `staging` \| `test` | notebook (variable `MODEL_ENV`) |
    | `status` | `best` \| `challenger` \| `worst` | notebook (calcul automatique sur R²) |

    - **Règle de calcul du `status`** (comparaison du R² de la nouvelle version vs toutes les versions existantes du même modèle dans MLflow) :
        - `best`       : R² > max des R² existants, **ou** premier enregistrement du modèle
        - `worst`      : R² < min des R² existants
        - `challenger` : R² intermédiaire (ni meilleur ni pire)
    - **Sélection par l'API** : `MlflowClient.search_model_versions()` filtre sur `name`, `tags.env` et `tags.status` ; parmi les versions retournées, la plus récente (numéro de version le plus élevé) est chargée
    - **Mode test** : si `MODEL_ENV=test`, l'API charge le modèle localement via `joblib` (fichier `MODEL_PATH`) sans contacter MLflow
    - **Variables d'environnement API** (fichier `api/.env`, template dans `api/.env_template`) :
        ```
        MODEL_ENV=prod                          # prod | staging | test
        MODEL_STATUS=best                       # best | challenger | worst
        MODEL_NAME=getaround_pricing
        MLFLOW_URI=https://gviel-mlflow37.hf.space/
        MODEL_PATH=../models/getaround_pricing_model.pkl   # test uniquement
        AWS_ACCESS_KEY_ID=<clé>                 # requis si MODEL_ENV=prod|staging
        AWS_SECRET_ACCESS_KEY=<secret>          # artifact store = s3://aws-s3-mlflow (eu-west-3)
        AWS_DEFAULT_REGION=eu-west-3
        ```
- faire ensuite une API pour faire l'inférence avec le meilleur modèle (tagger `env=prod, status=best` sur MLFlow)
    - on récupèrera le modèle dans le serveur MLFlow via `MlflowClient.search_model_versions()` (ou en local pour les tests)
    - faire un endpoint `/predict` (POST JSON) pour un véhicule unique
    - faire un endpoint `/predict/batch` (POST JSON) pour une liste de véhicules
    - faire un endpoint `/docs` pour documentation swagger détaillée avec des exemples
    - préparer les tests en local pour les endpoints
    - on fera d'abord des déploiements en local avec une image docker de type slim pour faire les tests

- **Endpoint `/predict/batch`** :
    - accepte une liste de `VehicleFeatures` (max `MAX_BATCH_SIZE`, défaut 20, configurable via env var)
    - chaque véhicule peut porter un `vehicle_id` optionnel (string) qui est renvoyé tel quel dans la réponse — permet au dashboard de faire la correspondance sans index
    - une seule inférence `model.predict(df)` sur le DataFrame complet (pas de boucle)
    - réponse : `{ predictions: [{vehicle_id, predicted_price_per_day, ...}], count, model_info }`

- **Fichier d'association `data/car_id_price_map.csv`** (généré une seule fois par `scripts/generate_car_map.py`) :
    - Les deux datasets (delay et pricing) **ne peuvent pas être joints** : espaces d'identifiants différents
    - Pour chaque `car_id` unique du dataset delay :
        - si le véhicule a au moins une location `connect` → associé à un véhicule pricing avec `has_getaround_connect=True` (pool de 2 230 véhicules)
        - sinon → associé à n'importe quel véhicule pricing valide (pool de 4 841 véhicules)
        - affectation déterministe : `pool[enumerate_index % len(pool)]`
    - Colonnes : `car_id`, `pricing_idx`, `is_connect`
    - Ce fichier est versionné et copié dans l'image Docker dashboard

- **Stratégie de prix dans le dashboard** (pour le calcul d'impact CA sur `get_around_delay_analysis.xlsx`) :
    - Charger `car_id_price_map.csv` → obtenir les `pricing_idx` uniques (~4 683)
    - Charger le dataset pricing → extraire les features de ces véhicules uniques
    - Envoyer en batch par paquets de `MAX_BATCH_SIZE` à `/predict/batch` (vehicle_id = pricing_idx)
    - Construire un dict `{car_id: prix_prédit}` et le stocker en **cache JSON** (string) pour le cache Streamlit
    - Utiliser `df["car_id"].map(prices)` pour affecter un prix à chaque location
    - Calculer le CA total = somme des prix prédits de toutes les locations ended du type sélectionné

---

## Partie 2 : analyse des retards

dans le notebook partie 2 sur le fichier `get_around_delay_analysis.xslx` (onglet `rentals_data`):
- faire une analyse du délai de retard > 0
  - il faut trouver les véhicules rendus en retard et ayant impacté la location suivante
    - regarder la proportion de véhicules de type 'connect' ou 'mobile' ayant des retards
    - je pense qu'il faut avoir les moyennes, mediane, std et faire boxplot et/ou graphe de distribution pour estimer le délai qu'il faut appliquer entre 2 véhicules (à décliner selon type connect/mobile)


---

## Partie 3 : dashboard streamlit

Le dashboard comporte **deux onglets** :

### Onglet 1 — Analyse des retards
- Calcule l'impact d'un délai minimum entre réservations sur le CA et le nombre de réservations bloquées
- Les prix sont estimés par le modèle ML via l'API (`/predict/batch`), en utilisant `car_id_price_map.csv`
- **Deux sliders distincts** (un par type de checkin, cf. conclusion notebook §2) :
    - Seuil Connect (min) — défaut 30 min
    - Seuil Mobile (min) — défaut 60 min
- Cases à cocher pour filtrer les types : Connect et/ou Mobile
- KPIs combinés selon les types sélectionnés : nb réservations totales, bloquées, CA total, CA perdu
- Courbes d'impact séparées par type (sensibilité au seuil)
- Histogramme de distribution des retards avec lignes verticales par type
- Tableau de statistiques des retards (count, moyenne, std, p50, p75, p90)

### Onglet 2 — Prédiction du prix
- Formulaire de saisie de toutes les features d'un véhicule (sélecteurs + cases)
- Appel à `POST /predict` → affichage du prix estimé
- Variables d'environnement dashboard :
    ```
    API_URL=http://localhost:8000      # URL de l'API FastAPI
    MAX_BATCH_SIZE=20                  # taille des paquets pour /predict/batch
    DATA_DIR=data                      # chemin vers les données
    ```
