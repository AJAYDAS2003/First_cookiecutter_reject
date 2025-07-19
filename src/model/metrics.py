import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score
)

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
        logger.info("Loading model from models/model.pkl...")
        with open('models/model.pkl', 'rb') as model_file:
            clf = pickle.load(model_file)
        logger.info("Model loaded successfully.")

        logger.info("Loading test dataset...")
        test_data = pd.read_csv('./data/features/test_tfidf.csv')  # or test_tfidf.csv
        logger.info(f"Test data loaded. Shape: {test_data.shape}")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info("Generating predictions...")
        y_pred = clf.predict(X_test)

        # Predict probabilities only if classifier supports it
        try:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba = None
            logger.warning("Model does not support probability prediction. AUC will be skipped.")

        logger.info("Calculating evaluation metrics...")
        average_type = 'binary' if len(np.unique(y_test)) == 2 else 'macro'

        metrics_dict = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, average=average_type), 4),
            'recall': round(recall_score(y_test, y_pred, average=average_type), 4),
            'f1_score': round(f1_score(y_test, y_pred, average=average_type), 4),
        }

        if y_pred_proba is not None and average_type == 'binary':
            metrics_dict['auc'] = round(roc_auc_score(y_test, y_pred_proba), 4)

        logger.info("Evaluation Metrics:")
        for key, val in metrics_dict.items():
            logger.info(f"{key}: {val}")

        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info("Metrics saved to reports/metrics.json")

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
