import pandas as pd
import joblib
import argparse
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from utils import log_run_infos

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

    with mlflow.start_run(run_name=f'log_artifacts_preprocessing'):
        mlflow.set_tags(config['mlflow'].get('tags', {}))

        log_run_infos()

        #logging dynamique des  hyperparamètre
        for param, value in config['xgb_model'].items():
            mlflow.log_param(f"xgb_{param}", value)

        #logguer le scaler
        mlflow.log_artifact(config['model']['scaler_path'], artifact_path = "preprocessing")

        #logguer les encodeurs 
        mlflow.log_artifact(config['model']['label_encoder_path'], artifact_path = "preprocessing")

        # Entraînement XGBoost
        xgb_model = train_xgb(X_train_scaled, y_train, config['xgb_model'])
        save_model(xgb_model, config['model']['xgb_path'], "XGBoost")

        # Sauvegarde du modèle XGBoost dans MLflow
        # Exemple d'entrée pour la documentation du modèle 
        input_example = X_train_scaled[:5]

        ## Signature du modèle
        signature = infer_signature(X_train_scaled, xgb_model.predict(X_train_scaled))


        mlflow.xgboost.log_model(xgb_model, name="xgb_model", input_example = input_example, signature = signature, registered_model_name = "CustomerChurnModelXgboost")

        if mode == 'debug':
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            mlflow.log_metric("xgb_model_balanced_accuracy", balanced_accuracy_score(y_test, y_pred_xgb))
            mlflow.log_metric("xgb_model_auc", roc_auc_score(y_test, y_pred_xgb))
            mlflow.log_metric("xgb_model_f1", f1_score(y_test, y_pred_xgb))
            
            #Courbe ROC    
            y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
            plt.figure()
            plt.plot(fpr, tpr, label= "xgb ROC")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            fig_name = "roc_xgb.png"
            plt.savefig(config['artifacts']['path']+fig_name)
            mlflow.log_artifact(config['artifacts']['path']+fig_name)
            plt.close()

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred_xgb)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            fig_name = "confusion_matrix_xgb.png"
            plt.savefig(config['artifacts']['path']+fig_name)
            mlflow.log_artifact(config['artifacts']['path']+fig_name)
            plt.close()

    logging.info("Training xgb model process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help="Chemin du fichier de configuration")
    args = parser.parse_args()
    main(args.config)