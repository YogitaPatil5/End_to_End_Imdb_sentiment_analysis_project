import os
import logging
from data_ingestion import DataIngestion
from data_preprocessing import TextPreprocessor
from model_training import SentimentModel
from model_evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the complete pipeline."""
    try:
        # Create necessary directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)
        
        # Initialize components
        data_ingestion = DataIngestion()
        text_preprocessor = TextPreprocessor()
        model = SentimentModel()
        evaluator = ModelEvaluator()
        
        # Download and load data
        logger.info("Downloading dataset...")
        downloaded_path = data_ingestion.download_dataset()
        if not downloaded_path:
            raise Exception("Failed to download dataset")
        
        logger.info("Loading data...")
        csv_path = os.path.join(downloaded_path, "IMDB Dataset.csv")
        train_df, test_df = data_ingestion.load_data(csv_path)
        X_train, y_train, X_test, y_test = data_ingestion.prepare_data(train_df, test_df)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train_processed = text_preprocessor.fit_transform(
            X_train,
            save_path="saved_models/tokenizer.pickle"
        )
        X_test_processed = text_preprocessor.transform(X_test)
        
        # Train model
        logger.info("Training model...")
        model.build_model()
        history = model.train(
            X_train_processed,
            y_train,
            X_val=X_test_processed,
            y_val=y_test,
            model_save_path="saved_models/sentiment_model.h5"
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_pred_proba = model.predict(X_test_processed)
        metrics, cm = evaluator.evaluate_predictions(y_test, y_pred_proba)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 