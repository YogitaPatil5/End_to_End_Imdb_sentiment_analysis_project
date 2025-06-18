"""
Data Ingestion Module for IMDB Sentiment Analysis Project.

This module handles downloading and loading the IMDB movie reviews dataset.
It provides functionality for:
1. Downloading the dataset from Kaggle using kagglehub
2. Loading and splitting the data into train/test sets
3. Preparing features and labels for model training
"""

import os  # For file and directory operations
import logging  # For logging messages and debugging information
import pandas as pd  # For data manipulation and CSV handling
from typing import Tuple, Optional  # For type hints to improve code readability
import kagglehub  # For downloading datasets from Kaggle

# Configure logging with detailed format for better debugging
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO to capture important events
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Include timestamp, module, level, and message
)
# Get logger instance for this module
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    A class for handling all data ingestion operations for the IMDB dataset.
    
    This class provides methods to:
    1. Download the dataset from Kaggle
    2. Load and split the data into train/test sets
    3. Prepare features and labels for model training
    
    Attributes:
        dataset_id (str): Kaggle dataset identifier
    """
    
    def __init__(self, dataset_id: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"):
        """
        Initialize the DataIngestion class with Kaggle dataset ID.
        
        Args:
            dataset_id (str): Kaggle dataset identifier. Defaults to IMDB 50k reviews dataset.
        """
        self.dataset_id = dataset_id
    
    def download_dataset(self, dest_folder: str = "data/raw") -> str:
        """
        Download the IMDB dataset from Kaggle using kagglehub.
        
        Args:
            dest_folder (str): Directory where the dataset should be downloaded.
                             Defaults to "data/raw".
        
        Returns:
            str: Path to the downloaded dataset folder, or empty string if download fails.
        
        Note:
            This method creates the destination folder if it doesn't exist.
            It handles download errors gracefully and logs any issues.
        """
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(dest_folder, exist_ok=True)
            
            # Download dataset using kagglehub
            # This handles authentication automatically if credentials are set up
            path = kagglehub.dataset_download(self.dataset_id)
            logger.info(f"Dataset downloaded successfully to {path}")
            return path
            
        except Exception as e:
            # Log error details and return empty string to indicate failure
            logger.error(f"Error downloading dataset: {str(e)}")
            return ""
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split the IMDB dataset into training and test sets.
        
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - train_df: DataFrame with 80% of the data for training
                - test_df: DataFrame with 20% of the data for testing
        
        Raises:
            Exception: If there's an error reading the CSV file or processing the data.
        
        Note:
            Uses random_state=42 for reproducible train/test splits.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(data_path)
            
            # Split into training (80%) and test (20%) sets
            train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
            test_df = df.drop(train_df.index)  # Remaining 20% for testing
            
            # Log dataset statistics
            logger.info(f"Successfully loaded dataset with {len(df)} total samples")
            logger.info(f"Training set: {len(train_df)} samples")
            logger.info(f"Test set: {len(test_df)} samples")
            
            return train_df, test_df
            
        except Exception as e:
            # Log error and re-raise for proper error handling upstream
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "review",
        label_column: str = "sentiment"
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare the data for model training by extracting features and labels.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame
            test_df (pd.DataFrame): Test DataFrame
            text_column (str): Name of the column containing review text. Defaults to "review".
            label_column (str): Name of the column containing sentiment labels. Defaults to "sentiment".
        
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: A tuple containing:
                - X_train: Training features (review texts)
                - y_train: Training labels (binary sentiment)
                - X_test: Test features (review texts)
                - y_test: Test labels (binary sentiment)
        
        Note:
            Converts sentiment labels to binary (1 for positive, 0 for negative)
        """
        try:
            # Extract features (review texts)
            X_train = train_df[text_column]
            X_test = test_df[text_column]
            
            # Convert sentiment labels to binary (1 for positive, 0 for negative)
            y_train = (train_df[label_column] == 'positive').astype(int)
            y_test = (test_df[label_column] == 'positive').astype(int)
            
            logger.info("Data preparation completed successfully")
            logger.info(f"Features shape - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

# Example usage of the DataIngestion class
if __name__ == "__main__":
    # Create an instance of DataIngestion
    ingestion = DataIngestion()
    
    # Download the dataset
    dataset_path = ingestion.download_dataset()
    
    if dataset_path:  # If download was successful
        try:
            # Load and split the data
            train_df, test_df = ingestion.load_data()
            
            # Prepare features and labels
            X_train, y_train, X_test, y_test = ingestion.prepare_data(train_df, test_df)
            
            # Print summary statistics
            print(f"Data loaded successfully:")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")