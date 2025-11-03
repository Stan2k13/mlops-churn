import pandas as pd
import joblib
import argparse
import yaml
import logging
from sklearn.preprocessing import LabelEncoder

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_input_data(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

def main(config_path):
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_predict'])

    input_path = config['predict']['input_path']
    output_path = config['predict']['output_path']
    model_choice = config['predict']['model_to_use']

    if model_choice == "xgb":
        model_path = config['model']['xgb_path']
    elif model_choice == "baseline":
        model_path = config['model']['baseline_path']
    else:
        raise ValueError("Unknown model_to_use: choose 'xgb' or 'baseline'.")

    df = pd.read_csv(input_path)
    df_processed = preprocess_input_data(df.copy())

    model = joblib.load(model_path)
    predictions = model.predict(df_processed)

    output_df = df.copy()
    output_df['predicted_churn'] = predictions
    output_df.to_csv(output_path, index=False)

    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
