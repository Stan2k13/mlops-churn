import pandas as pd
import argparse
import yaml
import logging
import sys
from sklearn.preprocessing import LabelEncoder

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(df, target_column):
    # Supprimer la colonne identifiant
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    features_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Vérifier qu'il sont tous numériques sinon convertir
    for col in features_numeric:
        if df[col].dtype != 'float64' and df[col].dtype != 'int64':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Vérifier le taux de valeurs manquantes dans la target
    missing_target_ratio = df[target_column].isna().mean()
    if missing_target_ratio > 0.05:
        logging.error(f"Plus de 5% de valeurs manquantes dans la colonne cible '{target_column}' ({missing_target_ratio:.2%}). Arrêt du preprocessing.")
        sys.exit(f"Erreur : trop de valeurs manquantes dans la colonne cible '{target_column}'.")

    # Supprimer les lignes avec target manquante
    df = df.dropna(subset=[target_column])

    # Completer les valeurs manquantes dans les colonnes numériques avec la médiane
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Colonnes catégorielles
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]

    for col in categorical_cols:
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0.05:
            logging.error(f"La colonne catégorielle '{col}' contient plus de 5% de valeurs manquantes ({missing_ratio:.2%}). Arrêt du preprocessing.")
            sys.exit(f"Erreur : trop de valeurs manquantes dans la colonne '{col}'.")
        else:
            df[col] = df[col].fillna("Unknown")

    # Encoder la variable cible (Yes/No)
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})

    # Encodage des colonnes catégorielles
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

def main(config_path):
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_preprocessing'])

    raw_path = config['data']['raw_path']
    output_path = config['data']['processed_path']
    target_col = config['preprocessing']['target_column']

    df = pd.read_csv(raw_path)
    logging.info(f"Raw data loaded from {raw_path}. Shape: {df.shape}")

    df_clean = preprocess_data(df, target_col)
    df_clean.to_csv(output_path, index=False)

    logging.info(f"Preprocessing complete. Clean data saved to {output_path}. Shape: {df_clean.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
