import os
import logging
import numpy as np
from typing import Tuple, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentModel:
    """A class for building and training sentiment analysis models."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        max_length: int = 200,
        lstm_units: int = 64,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the sentiment model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding layer
            max_length: Maximum sequence length
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        try:
            self.model = Sequential([
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                Dense(64, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model built successfully")
            self.model.summary(print_fn=logger.info)
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
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
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs
            model_save_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        try:
            callbacks = []
            
            # Early stopping
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=3,
                    restore_best_weights=True
                )
            )
            
            # Model checkpointing
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                callbacks.append(
                    ModelCheckpoint(
                        model_save_path,
                        monitor='val_loss' if X_val is not None else 'loss',
                        save_best_only=True
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
            logger.error(f"Error training model: {e}")
            raise
            
    def save_model(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to saved model
        """
        try:
            self.model = load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
