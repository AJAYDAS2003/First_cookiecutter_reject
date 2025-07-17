import numpy as np
import pandas as pd
import os
import re
import nltk
import logging
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

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

# Text cleaning functions
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        return text

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in str(text).split() if word not in stop_words])
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        return text

def removing_numbers(text):
    try:
        return ''.join([ch for ch in text if not ch.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        return text

def lower_case(text):
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logger.error(f"Error converting to lower case: {e}")
        return text

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        return " ".join(text.split()).strip()
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        return text

def normalize_text(df):
    try:
        logger.info("Starting text normalization...")
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
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
