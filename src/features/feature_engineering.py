import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_yaml_param(path, section, key):
    try:
        logger.info(f"Loading parameter '{key}' from section '{section}' in {path}")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            return config[section][key]
    except FileNotFoundError:
        logger.error(f"YAML file not found: {path}")
        raise
    except KeyError:
        logger.error(f"Missing key '{key}' under section '{section}' in YAML.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def main():
    try:
        logger.info("Loading processed training and testing data...")
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        logger.info("Data loaded successfully.")

        max_features = load_yaml_param('params.yaml', 'Feature_Engineering', 'max_features')
        logger.info(f"Max features for CountVectorizer: {max_features}")

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.info("Applying Bag-of-Words (CountVectorizer)...")
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        data_path = os.path.join('data', 'features')
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False, encoding='utf-8')

        logger.info("Feature engineering completed and CSV files saved.")
    
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
    except pd.errors.ParserError as pe:
        logger.error(f"Pandas CSV parsing error: {pe}")
    except Exception as e:
        logger.exception(f"Unexpected error in feature engineering pipeline: {e}")

if __name__ == '__main__':
    main()
