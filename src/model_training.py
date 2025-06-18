"""
Model Training Module for IMDB Sentiment Analysis Project.

This module implements a deep learning model for sentiment analysis using LSTM.
It provides functionality for:
1. Building an LSTM-based neural network
2. Training the model with early stopping and checkpointing
3. Saving and loading trained models
4. Making predictions on new data

The model architecture consists of:
- Embedding layer for word vector representation
- Two stacked LSTM layers for sequence processing
- Dropout layers for regularization
- Dense layers for classification
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentModel:
    """
    A class for building and training LSTM-based sentiment analysis models.
    
    This class implements a deep learning model that can:
    1. Process text data through an embedding layer
    2. Learn sequential patterns using LSTM layers
    3. Make binary sentiment predictions (positive/negative)
    
    The architecture is optimized for sentiment analysis tasks with:
    - Word embeddings for efficient text representation
    - Stacked LSTM layers for capturing long-term dependencies
    - Dropout for preventing overfitting
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        max_length: int = 200,
        lstm_units: int = 64,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the sentiment model with customizable architecture parameters.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique words).
                            Defaults to 10000.
            embedding_dim (int): Dimension of word embeddings.
                               Defaults to 100.
            max_length (int): Maximum length of input sequences.
                            Defaults to 200.
            lstm_units (int): Number of LSTM units in each layer.
                            Defaults to 64.
            dropout_rate (float): Dropout rate for regularization.
                                Defaults to 0.2.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self) -> None:
        """
        Build the LSTM model architecture.
        
        Architecture:
        1. Embedding layer: Converts word indices to dense vectors
        2. First LSTM layer: Processes sequences and returns sequences
        3. Dropout layer: Prevents overfitting
        4. Second LSTM layer: Further processes sequences
        5. Dropout layer: Additional regularization
        6. Dense layer: Feature extraction
        7. Output layer: Binary classification
        
        The model uses binary crossentropy loss and Adam optimizer.
        """
        try:
            # Create sequential model
            self.model = Sequential([
                # Embedding layer converts word indices to dense vectors
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    input_length=self.max_length
                ),
                
                # First LSTM layer with return sequences for stacking
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True
                ),
                Dropout(self.dropout_rate),
                
                # Second LSTM layer
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                
                # Dense layer for feature extraction
                Dense(64, activation='relu'),
                Dropout(self.dropout_rate),
                
                # Output layer for binary classification
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model with Adam optimizer
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model built successfully")
            self.model.summary(print_fn=logger.info)
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 10,
        model_save_path: Optional[str] = None
    ) -> dict:
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            X_train (np.ndarray): Training data (padded sequences)
            y_train (np.ndarray): Training labels (binary)
            X_val (np.ndarray, optional): Validation data
            y_val (np.ndarray, optional): Validation labels
            batch_size (int): Number of samples per gradient update.
                            Defaults to 32.
            epochs (int): Number of epochs to train.
                         Defaults to 10.
            model_save_path (str, optional): Path to save the best model
        
        Returns:
            dict: Training history containing loss and accuracy metrics
        
        Note:
            - Uses early stopping to prevent overfitting
            - Saves the best model based on validation loss
            - Restores best weights after training
        """
        if self.model is None:
            self.build_model()
            
        try:
            callbacks = []
            
            # Add early stopping to prevent overfitting
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=3,  # Number of epochs with no improvement
                    restore_best_weights=True  # Restore weights from best epoch
                )
            )
            
            # Add model checkpointing to save best model
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                callbacks.append(
                    ModelCheckpoint(
                        model_save_path,
                        monitor='val_loss' if X_val is not None else 'loss',
                        save_best_only=True  # Only save when model improves
                    )
                )
            
            # Train the model
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path where the model should be saved
        
        Raises:
            ValueError: If no model exists to save
            Exception: If there's an error saving the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model
        
        Raises:
            Exception: If there's an error loading the model
        """
        try:
            self.model = load_model(path)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make sentiment predictions on new data.
        
        Args:
            X (np.ndarray): Input data (padded sequences)
        
        Returns:
            np.ndarray: Predicted probabilities of positive sentiment
        
        Raises:
            ValueError: If no model is loaded
            Exception: If there's an error during prediction
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage of the SentimentModel class
    model = SentimentModel()
    
    # Build the model architecture
    model.build_model()
    
    # Example of model training (assuming data is prepared)
    # X_train, y_train = prepare_data()  # You would need to implement this
    # history = model.train(X_train, y_train, model_save_path="models/sentiment_model.h5")
    # print("Training metrics:", history)
