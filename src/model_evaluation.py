import logging
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """A class for evaluating sentiment analysis models."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the model evaluator.
        
        Args:
            threshold: Classification threshold for binary predictions
        """
        self.threshold = threshold
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }
            
            # Log metrics
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
            
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Confusion matrix
        """
        try:
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            logger.info("Confusion Matrix:")
            logger.info(f"True Negatives: {cm[0,0]}")
            logger.info(f"False Positives: {cm[0,1]}")
            logger.info(f"False Negatives: {cm[1,0]}")
            logger.info(f"True Positives: {cm[1,1]}")
            
            return cm
            
        except Exception as e:
            logger.error(f"Error computing confusion matrix: {e}")
            raise
            
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (metrics dictionary, confusion matrix)
        """
        try:
            metrics = self.evaluate(y_true, y_pred_proba)
            cm = self.get_confusion_matrix(y_true, y_pred_proba)
            
            return metrics, cm
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            raise