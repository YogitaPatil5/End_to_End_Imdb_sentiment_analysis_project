import os
import re
import pickle
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """A class for preprocessing text data for sentiment analysis."""
    
    def __init__(self, max_words: int = 10000, max_len: int = 200):
        """
        Initialize the text preprocessor.
        
        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_len: Maximum sequence length
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self._ensure_nltk_data()
        
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove stopwords
            words = text.split()
            filtered_words = [word for word in words if word not in self.stop_words]
            
            return ' '.join(filtered_words)
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""
            
    def fit_transform(self, texts: List[str], save_path: Optional[str] = None) -> np.ndarray:
        """
        Fit tokenizer and transform texts to sequences.
        
        Args:
            texts: List of texts to process
            save_path: Optional path to save the tokenizer
            
        Returns:
            Padded sequences
        """
        try:
            # Clean texts
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            # Create and fit tokenizer
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(cleaned_texts)
            
            # Convert to sequences
            sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            
            # Pad sequences
            padded_sequences = pad_sequences(
                sequences,
                maxlen=self.max_len,
                padding='post',
                truncating='post'
            )
            
            # Save tokenizer if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(self.tokenizer, f)
                logger.info(f"Tokenizer saved to {save_path}")
                
            return padded_sequences
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted tokenizer.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Padded sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_transform first.")
            
        try:
            cleaned_texts = [self.clean_text(text) for text in texts]
            sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            return pad_sequences(
                sequences,
                maxlen=self.max_len,
                padding='post',
                truncating='post'
            )
            
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise
            
    def load_tokenizer(self, path: str) -> None:
        """
        Load a saved tokenizer.
        
        Args:
            path: Path to saved tokenizer
        """
        try:
            with open(path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logger.info(f"Tokenizer loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

def download_dataset_kaggle(dataset_name="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", dest_folder="data/raw"):
    """
    Download dataset from Kaggle using the official kaggle API.
    """
    try:
        os.makedirs(dest_folder, exist_ok=True)
        # Download the dataset zip
        os.system(f'kaggle datasets download -d {dataset_name} -p {dest_folder} --unzip')
        logger.info(f"Dataset downloaded and extracted to {dest_folder}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False
