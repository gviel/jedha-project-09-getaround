# models/

Ce dossier contient le modèle ML entraîné localement (non versionné dans git).

| Fichier | Description |
|---|---|
| `getaround_pricing_model.pkl` | Pipeline scikit-learn sérialisé via joblib — prédit le prix de location journalier |

## Comment l'obtenir

**En local** : exécuter le notebook `Getaround_Pricing_GV.ipynb` (section entraînement),
le modèle est sauvegardé automatiquement dans ce dossier.

**En production (Render)** : le modèle est chargé depuis le registre MLflow
(`MODEL_ENV=prod`) — ce dossier n'est pas nécessaire.

**Mode test Docker** :
```bash
bash scripts/docker_api.sh 8000 test
```
Le pickle local est alors monté depuis ce dossier dans le conteneur.
