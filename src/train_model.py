import pandas as pd
import joblib
import argparse
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
import os

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_baseline(X, y, config):
    model = LogisticRegression(
        C=config['C'],
        max_iter=config['max_iter'],
        random_state=0
    )
    model.fit(X, y)
    return model

def train_xgb(X, y, config):
    model = XGBClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['learning_rate'],
        reg_lambda=config.get('reg_lambda', 1.0),
        reg_alpha=config.get('reg_alpha', 0.5),
        gamma=config.get('gamma', 0),
        subsample=config.get('subsample', 1.0),
        colsample_bytree=config.get('colsample_bytree', 1.0),
        scale_pos_weight=config.get('scale_pos_weight', 1.0),
        eval_metric='logloss',
        random_state=0
    )
    model.fit(X, y)
    return model

def save_model(model, path, model_name):
    joblib.dump(model, path)
    logging.info(f"{model_name} model saved to {path}")

def main(config_path):
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_training'])

    mode = config['train'].get('mode', 'production')
    target_col = config['preprocessing']['target_column']

    df = pd.read_csv(config['data']['processed_path'])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if mode == 'debug':
        logging.info("Mode: DEBUG – train/test split 80/20")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['train']['test_size'],
            random_state=config['train']['random_state']
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    elif mode == 'production':
        logging.info("Mode: PRODUCTION – toute les données sont utilisées pour l'entraînement")
        X_train, y_train = X, y
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled, y_test = None, None
    else:
        logging.error(f"Mode '{mode}' invalide. Utilisez 'debug' ou 'production'.")
        raise ValueError("Mode d'entraînement invalide")

    # Sauvegarder le scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    # Entraînement Baseline
    baseline_model = train_baseline(X_train_scaled, y_train, config['baseline_model'])
    save_model(baseline_model, config['model']['baseline_path'], "Baseline")

    if mode == 'debug':
        y_pred_baseline = baseline_model.predict(X_test_scaled)
        acc_baseline = balanced_accuracy_score(y_test, y_pred_baseline)
        logging.info(f"Baseline accuracy: {acc_baseline:.4f}")

    # Entraînement XGBoost
    xgb_model = train_xgb(X_train_scaled, y_train, config['xgb_model'])
    save_model(xgb_model, config['model']['xgb_path'], "XGBoost")

    if mode == 'debug':
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        acc_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
        logging.info(f"XGBoost accuracy: {acc_xgb:.4f}")

    logging.info("Training process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help="Chemin du fichier de configuration")
    args = parser.parse_args()
    main(args.config)
