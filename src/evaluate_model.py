import pandas as pd
import joblib
import argparse
import yaml
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
import os
import sys

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    logging.info(f"Évaluation du modèle {name} :\n{report}")
    logging.info(f"{name} balanced accuracy: {balanced_acc:.4f}")

def main(config_path):
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_evaluation'])

    # Chargement des données
    df = pd.read_csv(config['data']['processed_path'])
    target_col = config['preprocessing']['target_column']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state']
    )

    # Chargement du scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    if not os.path.exists(scaler_path):
        logging.error(f"Scaler non trouvé à {scaler_path}.")
        sys.exit("Erreur : scaler introuvable, entraînement requis.")
    scaler = joblib.load(scaler_path)

    # Transformation des features
    X_test_scaled = scaler.transform(X_test)

    # Évaluation du modèle baseline
    baseline_model = joblib.load(config['model']['baseline_path'])
    evaluate(baseline_model, X_test_scaled, y_test, "Baseline")

    # Évaluation du modèle XGBoost
    xgb_model = joblib.load(config['model']['xgb_path'])
    evaluate(xgb_model, X_test_scaled, y_test, "XGBoost")

    logging.info("Évaluation des modèles terminée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
