import os
import logging
import pandas as pd
from typing import Tuple, Optional
import kagglehub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """A class for downloading and loading the IMDB dataset."""
    
    def __init__(self, dataset_id: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"):
        """
        Initialize the data ingestion.
        
        Args:
            dataset_id: Kaggle dataset ID
        """
        self.dataset_id = dataset_id
        
    def download_dataset(self, dest_folder: str = "data/raw") -> str:
        """
        Download the dataset from Kaggle using kagglehub.
        Returns the path to the downloaded dataset folder, or an empty string on failure.
        """
        try:
            os.makedirs(dest_folder, exist_ok=True)
            path = kagglehub.dataset_download(self.dataset_id)
            logger.info(f"Dataset downloaded to {path}")
            return path
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return ""
            
    def load_data(self, data_path: str = "data/raw/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/IMDB Dataset.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split the dataset into train and test sets.
        
        Args:
            data_path: Path to the dataset file
            
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            
            # Split into train and test (80-20 split)
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            
            logger.info(f"Loaded dataset with {len(df)} samples")
            logger.info(f"Train set: {len(train_df)} samples")
            logger.info(f"Test set: {len(test_df)} samples")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "review",
        label_column: str = "sentiment"
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare the data for training.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        try:
            # Extract features and labels
            X_train = train_df[text_column]
            y_train = (train_df[label_column] == 'positive').astype(int)
            
            X_test = test_df[text_column]
            y_test = (test_df[label_column] == 'positive').astype(int)
            
            logger.info("Data prepared successfully")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

if __name__ == "__main__":
    if download_dataset_kagglehub():
        try:
            df = load_dataset("data/raw/IMDB Dataset.csv")
        except Exception:
            pass