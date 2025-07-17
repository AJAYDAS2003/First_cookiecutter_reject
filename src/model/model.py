import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import os

from sklearn.ensemble import GradientBoostingClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log", encoding="utf-8"),
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
        logger.info("Loading training feature data...")
        train_data = pd.read_csv('./data/features/train_bow.csv')
        logger.info(f"Training data loaded successfully. Shape: {train_data.shape}")

        n_estimators = load_yaml_param('params.yaml', 'Model_Building', 'n_estimators')
        learning_rate = load_yaml_param('params.yaml', 'Model_Building', 'learning_rate')
        logger.info(f"Using GradientBoostingClassifier with n_estimators={n_estimators}, learning_rate={learning_rate}")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logger.info("Training the GradientBoostingClassifier...")
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        logger.info("Model training completed.")

        model_path = 'models/model.pkl'
        with open(model_path, 'wb') as model_file:
            pickle.dump(clf, model_file)
        logger.info(f"Model saved to {model_path}")

    except FileNotFoundError as fe:
        logger.error(f"File not found: {fe}")
    except pd.errors.ParserError as pe:
        logger.error(f"Pandas parsing error: {pe}")
    except Exception as e:
        logger.exception(f"Unexpected error during model training: {e}")

if __name__ == '__main__':
    main()
