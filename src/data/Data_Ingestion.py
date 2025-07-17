import numpy as np
import pandas as pd
import os
import yaml
import logging
import sys
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # Ensures output goes to console using UTF-8
    ]
)
logger = logging.getLogger(__name__)

def load_params(path: str) -> float:
    try:
        logger.info(f"Loading parameters from {path}")
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
            test_size = params['Data_Ingestion']['test_size']
            logger.info(f"Loaded test_size={test_size}")
            return test_size
    except FileNotFoundError:
        logger.error(f"'params.yaml' not found at: {path}")
        raise
    except KeyError:
        logger.error(f"Missing 'Data_Ingestion.test_size' in {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {str(e)}")
        raise

def load_data(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {url}")
        df = pd.read_csv(url)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from URL: {url}\n{str(e)}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Processing data...")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.info(f"Processed data shape: {final_df.shape}")
        return final_df
    except KeyError as e:
        logger.error(f"Missing expected column: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.info(f"Saving data to {data_path}")
        os.makedirs(data_path, exist_ok=True)
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info(f"Train and test data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Failed to save data to '{data_path}': {str(e)}")
        raise

def main():
    try:
        logger.info("🚀 Starting data ingestion pipeline...")
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)
        logger.info("✅ Data ingestion and splitting completed successfully.")
    except Exception as e:
        logger.exception("❌ Pipeline failed due to an unexpected error.")

if __name__ == '__main__':
    main()
