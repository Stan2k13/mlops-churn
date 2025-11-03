import pandas as pd
import joblib
import argparse
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import mlflow
from utils import log_run_infos

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

def save_model(model, path, model_name):
    joblib.dump(model, path)
    logging.info(f"{model_name} model saved to {path}")

def main(config_path):
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_training'])

    # initialisation de mlflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # chargement des données
    mode = config['train'].get('mode', 'production')
    target_col = config['preprocessing']['target_column']

    df = pd.read_csv(config['data']['processed_path'])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # prétraitement
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

    with mlflow.start_run(run_name=f'train_baseline_model'):
        mlflow.set_tags(config['mlflow'].get('tags', {}))

        log_run_infos()

        #logging dynamique des  hyperparamètre
        for param, value in config['baseline_model'].items():
            mlflow.log_param(f"baseline_{param}", value)

        # Entraînement Baseline
        baseline_model = train_baseline(X_train_scaled, y_train, config['baseline_model'])
        save_model(baseline_model, config['model']['baseline_path'], "Baseline")

        if mode == 'debug':
            y_pred_baseline = baseline_model.predict(X_test_scaled)
            mlflow.log_metric("baseline_balanced_accuracy", balanced_accuracy_score(y_test, y_pred_baseline))
            mlflow.log_metric("baseline_auc", roc_auc_score(y_test, y_pred_baseline))
            mlflow.log_metric("baseline_f1", f1_score(y_test, y_pred_baseline))
        
            #Courbe ROC    
            y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob_baseline)
            plt.figure()
            plt.plot(fpr, tpr, label= "Baseline ROC")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            fig_name = "roc_baseline.png"
            plt.savefig(config['artifacts']['path']+fig_name)
            mlflow.log_artifact(config['artifacts']['path']+fig_name)
            plt.close()

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred_baseline)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            fig_name = "confusion_matrix_baseline.png"
            plt.savefig(config['artifacts']['path']+fig_name)
            mlflow.log_artifact(config['artifacts']['path']+fig_name)
            plt.close()

    logging.info("Training baseline model process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help="Chemin du fichier de configuration")
    args = parser.parse_args()
    main(args.config)