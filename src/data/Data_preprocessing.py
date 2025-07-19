import numpy as np
import pandas as pd
import os
import re
import nltk
import logging
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise

# Initialize once globally
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning functions
def lemmatization(text):
    try:
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        return text

def remove_stop_words(text):
    try:
        return " ".join([word for word in str(text).split() if word not in stop_words])
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        return text

def removing_numbers(text):
    try:
        return re.sub(r'\d+', '', text)
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        return text

def lower_case(text):
    try:
        return text.lower()
    except Exception as e:
        logger.error(f"Error converting to lower case: {e}")
        return text

def removing_punctuations(text):
    try:
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text):
    try:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        return text

def normalize_text(df):
    try:
        if 'content' not in df.columns:
            raise KeyError("'content' column not found in DataFrame.")
        
        logger.info("Starting text normalization...")
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(lower_case)\
                                     .apply(remove_stop_words)\
                                     .apply(removing_numbers)\
                                     .apply(removing_punctuations)\
                                     .apply(removing_urls)\
                                     .apply(lemmatization)
        logger.info("Text normalization completed.")
        return df
    except Exception as e:
        logger.error(f"Error in normalize_text(): {e}")
        raise

def main():
    try:
        logger.info("Loading training and testing data...")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Data loaded successfully.")

        logger.info("Preprocessing training data...")
        train_processed = normalize_text(train_data)

        logger.info("Preprocessing testing data...")
        test_processed = normalize_text(test_data)

        data_path = os.path.join('data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False, encoding='utf-8')
        test_processed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False, encoding='utf-8')
        logger.info("Preprocessed data saved successfully.")
    
    except FileNotFoundError as fe:
        logger.error(f"File not found: {fe}")
    except pd.errors.ParserError as pe:
        logger.error(f"Pandas parsing error: {pe}")
    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")

if __name__ == "__main__":
    main()
