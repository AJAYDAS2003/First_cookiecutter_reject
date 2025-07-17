import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading model from model.pkl...")
        with open('models/model.pkl', 'rb') as model_file:
            clf = pickle.load(model_file)
        logger.info("Model loaded successfully.")

        logger.info("Loading test dataset...")
        test_data = pd.read_csv('./data/features/test_bow.csv')
        logger.info(f"Test data loaded. Shape: {test_data.shape}")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info("Generating predictions...")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.info(f"Metrics: {metrics_dict}")

        metrics_path = 'reports/metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as file:
            json.dump(metrics_dict, file, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
    except pickle.UnpicklingError as pe:
        logger.error(f"Error loading the model: {pe}")
    except pd.errors.ParserError as pe:
        logger.error(f"Error reading CSV file: {pe}")
    except Exception as e:
        logger.exception(f"Unexpected error during evaluation: {e}")

if __name__ == '__main__':
    main()
