#!/bin/bash
echo " Lancement du pipeline complet..."

echo "Étape 1 : Prétraitement"
python src/data_preprocessing.py --config configs/config.yaml

echo "Étape 2 : Entraînement des modèles"
python src/train_model.py --config configs/config.yaml

echo "Étape 3 : Évaluation"
python src/evaluate_model.py --config configs/config.yaml

echo "Pipeline terminé avec succès."
