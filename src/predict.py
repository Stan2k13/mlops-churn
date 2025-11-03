import pandas as pd
import joblib
import argparse
import yaml
import logging
import os
from pathlib import Path
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# === Utilitaires de chemin : ancrage à la racine du repo (parent de src/) ===
REPO_ROOT = Path(__file__).resolve().parents[1]

def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)

def setup_logger(log_path):
    """Initialise le logger pour suivre les étapes de prédiction."""
    p = _abs(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(p),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path):
    """Charge le fichier de configuration."""
    cfg_path = _abs(config_path)
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_for_inference(df, encoders, scaler):
    # Supprimer les identifiants inutiles
    df = df.drop(columns=['customerID'], errors='ignore')

    # Colonnes numériques à convertir et nettoyer
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Encodage des colonnes catégorielles avec les encoders sauvegardés
    for col, encoder in encoders.items():
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
        # ⚠️ Assure-toi que "Unknown" a bien été vu au fit des encoders côté training
        df[col] = encoder.transform(df[col])

    # Scaler les features
    return scaler.transform(df)

def main(config_path):
    # Charge de la config et initialisation du logger
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_predict'])
    logging.info("Starting prediction ...")

    # Chemins et paramètres depuis la config (résolus en absolu)
    input_path = _abs(config['predict']['input_path'])
    output_path = _abs(config['predict']['output_path'])
    model_registry_name = config['predict']['model_registry_name']
    model_version = config['predict']['version_to_use']

    tracking_uri = config['mlflow']['tracking_uri']
    scaler_path = _abs(config['model']['scaler_path'])
    label_encoder_path = _abs(config['model']['label_encoder_path'])

    # Création du dossier de sortie si besoin
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3 Chargement des nouvelles données brutes à prédire
    df_raw = pd.read_csv(input_path)
    logging.info(f"Données brutes chargées depuis {input_path}. Shape: {df_raw.shape}")

    # 4 Connexion et chargement du modèle depuis MLflow Model Registry
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_registry_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info(f"Modèle chargé depuis {model_uri}.")

    # 5 Réupération du runid du modèle que l'on vient de charger 
    client = MlflowClient()
    model_version_info = client.get_model_version(name=model_registry_name, version=model_version)
    source_run_id = model_version_info.run_id

    # 6 Télécharger les artéfacts de préprocessing
    preprocessing_dir = "preprocessing_artifacts"
    os.makedirs(preprocessing_dir, exist_ok=True)
    client.download_artifacts(source_run_id, "preprocessing/scaler.pkl", preprocessing_dir)
    client.download_artifacts(source_run_id, "preprocessing/label_encoders.pkl", preprocessing_dir)
    
    # 7 Charger les objets récupérés depuis mlflow
    scaler = joblib.load(os.path.join(preprocessing_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(preprocessing_dir, "label_encoders.pkl"))

    # 8 Prétraitement des nouvelles données
    df_clean = preprocess_for_inference(df_raw, encoders, scaler)
    logging.info("Prétraitement des nouvelles données effectué.")
                                        
    # 9 Prédiction
    y_pred = model.predict(df_clean)
    df_raw['Churn'] = y_pred

    # 10 Sauvegarde des nouvelles données prédites
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logging.info(f"Prédictions sauvegardées dans {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prédiction de churn')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Chemin du fichier de configuration")
    args = parser.parse_args()
    main(args.config)