# 📚 Project Explanation: End-to-End IMDB Sentiment Analysis

## 📝 Problem Statement

Build an end-to-end machine learning pipeline to classify IMDB movie reviews as positive or negative using deep learning (LSTM) and deploy it as an interactive web app.

---

## 🗂️ Project Structure & Script Roles

### 1. **Data Ingestion (`src/data_ingestion.py`)**
- **Purpose:** Download the IMDB dataset from Kaggle using `kagglehub`.
- **Key Steps:**
  - Handles Kaggle API authentication
  - Downloads and extracts the dataset
  - Loads the CSV into pandas DataFrames for further processing

### 2. **Data Preprocessing (`src/data_preprocessing.py`)**
- **Purpose:** Clean and prepare the text data for modeling.
- **Key Steps:**
  - Cleans text (removes HTML, punctuation, stopwords, etc.)
  - Tokenizes and pads sequences (maxlen=200)
  - Saves the tokenizer for use in the app

### 3. **Model Training (`src/model_training.py`)**
- **Purpose:** Build and train the LSTM model.
- **Key Steps:**
  - Defines the LSTM architecture
  - Trains the model on the preprocessed data
  - Saves the trained model (`sentiment_model.h5`)

### 4. **Model Evaluation (`src/model_evaluation.py`)**
- **Purpose:** Evaluate the trained model's performance.
- **Key Steps:**
  - Predicts on the test set
  - Calculates accuracy and other metrics
  - Optionally displays a confusion matrix

### 5. **Main Pipeline (`src/main.py`)**
- **Purpose:** Orchestrate the entire workflow.
- **Key Steps:**
  - Runs data ingestion, preprocessing, training, and evaluation in sequence.
  - Ensures all artifacts (model, tokenizer) are saved for deployment.

### 6. **Web App (`app/app.py`)**
- **Purpose:** Provide an interactive UI for sentiment prediction.
- **Key Steps:**
  - Loads the trained model and tokenizer
  - Accepts user input and predicts sentiment in real time
  - Shows confidence scores and sample reviews

### 7. **Dockerfile**
- **Purpose:** Containerize the entire project for easy deployment.
- **Key Steps:**
  - Installs all dependencies.
  - Copies project files.
  - Sets up the environment to run the Streamlit app.

---

## 🧩 Workflow Summary

1. **Download and prepare data** with `data_ingestion.py` and `data_preprocessing.py`.
2. **Train and evaluate the model** with `model_training.py` and `model_evaluation.py`.
3. **Run the main pipeline** with `main.py` to automate the above steps.
4. **Launch the web app** with `app/app.py` (locally or via Docker).
5. **(Optional) Deploy anywhere** using the Dockerfile.

---

## 🏁 Result

- A robust, reproducible, and interactive sentiment analysis solution for IMDB reviews.
- Easily extensible for other text classification tasks.

---

## 📄 Detailed Explanation: src/data_ingestion.py

This file handles downloading and loading the IMDB dataset for the project. Below is a line-by-line explanation:

```python
import os  # For file and directory operations
import logging  # For logging messages
import pandas as pd  # For data manipulation
from typing import Tuple, Optional  # For type hints
import kagglehub  # For downloading datasets from Kaggle
```
- **Imports**: Brings in all necessary libraries for file operations, logging, data handling, type hints, and Kaggle dataset download.

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```
- **Logging setup**: Configures logging to show time, module, level, and message for easier debugging and tracking.

```python
class DataIngestion:
    """A class for downloading and loading the IMDB dataset."""
```
- **Class definition**: Encapsulates all data ingestion logic.

```python
    def __init__(self, dataset_id: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"):
        ...
        self.dataset_id = dataset_id  # Store the dataset ID
```
- **Constructor**: Sets the default Kaggle dataset ID for IMDB reviews.

```python
    def download_dataset(self, dest_folder: str = "data/raw") -> str:
        ...
        os.makedirs(dest_folder, exist_ok=True)  # Ensure destination folder exists
        path = kagglehub.dataset_download(self.dataset_id)  # Download dataset
        logger.info(f"Dataset downloaded to {path}")  # Log success
        return path  # Return path to dataset
```
- **download_dataset**: Downloads the dataset using kagglehub, ensures the destination folder exists, logs the result, and returns the path to the downloaded data.

```python
    def load_data(self, data_path: str = "data/raw/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/IMDB Dataset.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...
        df = pd.read_csv(data_path)  # Load the dataset as a DataFrame
        train_df = df.sample(frac=0.8, random_state=42)  # Random 80% for training
        test_df = df.drop(train_df.index)  # Remaining 20% for testing
        logger.info(f"Loaded dataset with {len(df)} samples")
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        return train_df, test_df  # Return train and test DataFrames
```
- **load_data**: Loads the CSV file, splits it into training and test sets (80/20), logs the sizes, and returns both DataFrames.

```python
    def prepare_data(...):
        ...
        X_train = train_df[text_column]  # Extract review texts for training
        y_train = (train_df[label_column] == 'positive').astype(int)  # Convert sentiment to binary
        X_test = test_df[text_column]  # Extract review texts for testing
        y_test = (test_df[label_column] == 'positive').astype(int)  # Convert sentiment to binary
        logger.info("Data prepared successfully")
        return X_train, y_train, X_test, y_test  # Return features and labels
```
- **prepare_data**: Extracts the review text and sentiment labels, converts sentiment to binary (1 for positive, 0 for negative), and returns the features and labels for both train and test sets.

```python
# The following block is not used in the main workflow and can be removed or updated as needed.
if __name__ == "__main__":
    if download_dataset_kagglehub():
        try:
            df = load_dataset("data/raw/IMDB Dataset.csv")
        except Exception:
            pass
```
- **Main block**: This is not used in the main workflow and can be removed or updated as needed.

---

**For more details, see the [README](./README.md) or the [GitHub repo](https://github.com/YogitaPatil5/End_to_End_Imdb_sentiment_analysis_project).** 

## 📄 Detailed Code Explanations

### 1. Data Ingestion (`src/data_ingestion.py`)

```python
# Import necessary libraries
import os  # For file and directory operations
import logging  # For logging messages and debugging
import pandas as pd  # For data manipulation and CSV handling
from typing import Tuple, Optional  # For type hints to improve code readability
import kagglehub  # For downloading datasets from Kaggle

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Include timestamp, module, level, message
)
logger = logging.getLogger(__name__)  # Get logger for this module

class DataIngestion:
    """A class for downloading and loading the IMDB dataset."""
    
    def __init__(self, dataset_id: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"):
        """
        Initialize the data ingestion class.
        Args:
            dataset_id: Kaggle dataset ID (default is IMDB 50k reviews)
        """
        self.dataset_id = dataset_id
    
    def download_dataset(self, dest_folder: str = "data/raw") -> str:
        """
        Download dataset from Kaggle using kagglehub.
        Args:
            dest_folder: Directory to save the dataset
        Returns:
            Path to downloaded dataset or empty string if failed
        """
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(dest_folder, exist_ok=True)
            # Download the dataset using kagglehub
            path = kagglehub.dataset_download(self.dataset_id)
            logger.info(f"Dataset downloaded to {path}")
            return path
        except Exception as e:
            # Log error and return empty string to indicate failure
            logger.error(f"Error downloading dataset: {e}")
            return ""
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split dataset into train and test sets.
        Args:
            data_path: Path to the CSV file
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            # Read CSV file into pandas DataFrame
            df = pd.read_csv(data_path)
            # Split into 80% train, 20% test
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            # Log dataset statistics
            logger.info(f"Loaded dataset with {len(df)} samples")
            logger.info(f"Train set: {len(train_df)} samples")
            logger.info(f"Test set: {len(test_df)} samples")
            return train_df, test_df
        except Exception as e:
            # Log error and re-raise for proper error handling upstream
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    text_column: str = "review", label_column: str = "sentiment"
                    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training by extracting features and labels.
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            text_column: Name of text column
            label_column: Name of label column
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        try:
            # Extract features (review texts)
            X_train = train_df[text_column]
            X_test = test_df[text_column]
            # Convert sentiment labels to binary (1 for positive, 0 for negative)
            y_train = (train_df[label_column] == 'positive').astype(int)
            y_test = (test_df[label_column] == 'positive').astype(int)
            logger.info("Data prepared successfully")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the DataIngestion class
    ingestion = DataIngestion()
    dataset_path = ingestion.download_dataset()
    if dataset_path:
        try:
            train_df, test_df = ingestion.load_data()
            X_train, y_train, X_test, y_test = ingestion.prepare_data(train_df, test_df)
            print(f"Data loaded successfully: {len(X_train)} training samples, {len(X_test)} test samples")
        except Exception as e:
            print(f"Error processing data: {e}")
```

### 2. Model Training (`src/model_training.py`)

```python
# Import necessary libraries
import os
import logging
import numpy as np
from typing import Tuple, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class SentimentModel:
    """A class for building and training sentiment analysis models."""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100,
                 max_length: int = 200, lstm_units: int = 64, dropout_rate: float = 0.2):
        """
        Initialize model parameters.
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
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
            # Create sequential model
            self.model = Sequential([
                # Embedding layer to convert words to vectors
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                # First LSTM layer with return sequences for stacking
                LSTM(self.lstm_units, return_sequences=True),
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
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              batch_size: int = 32, epochs: int = 10,
              model_save_path: Optional[str] = None) -> dict:
        """
        Train the model with early stopping and checkpointing.
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs
            model_save_path: Path to save best model
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        try:
            callbacks = []
            
            # Add early stopping to prevent overfitting
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=3,
                    restore_best_weights=True
                )
            )
            
            # Add model checkpointing to save best model
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
                X_train, y_train,
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
```

### 3. Web Application (`app/app.py`)

```python
# Import necessary libraries
import os
import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Load model and tokenizer with caching
@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with error handling."""
    model, tokenizer = None, None
    
    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    
    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            st.success("✅ Tokenizer loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading tokenizer: {e}")
    
    return model, tokenizer

# Main content
def main():
    """Main application function."""
    st.title("🎬 IMDB Movie Review Sentiment Analyzer")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app analyzes movie review sentiment using LSTM.
        
        **Features:**
        - Real-time prediction
        - Confidence scores
        - Sample reviews
        """)
    
    # Input area
    review = st.text_area("Enter your review:", height=150)
    
    if st.button("Analyze"):
        if not review:
            st.warning("Please enter a review first!")
            return
            
        # Process and predict
        try:
            # Clean text
            cleaned_review = clean_text(review)
            # Tokenize and pad
            sequence = tokenizer.texts_to_sequences([cleaned_review])
            padded = pad_sequences(sequence, maxlen=200)
            # Predict
            prediction = model.predict(padded)[0][0]
            
            # Display results
            sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            st.subheader("Results:")
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
```

## 🔍 Key Implementation Details

1. **Error Handling**
   - All critical operations are wrapped in try-except blocks
   - Detailed error messages are logged
   - User-friendly error messages in the web app

2. **Code Organization**
   - Clear separation of concerns
   - Object-oriented design where appropriate
   - Consistent coding style and documentation

3. **Performance Considerations**
   - Caching for model and tokenizer loading
   - Efficient text preprocessing
   - Batch processing for predictions

4. **User Experience**
   - Clean, modern UI design
   - Informative feedback messages
   - Sample reviews for testing
   - Clear display of results

## 🚀 Next Steps

1. Add unit tests
2. Implement model versioning
3. Add more advanced text preprocessing
4. Expand the web app features
5. Set up CI/CD pipeline

For more details, see the [README.md](./README.md) or visit the [GitHub repository](https://github.com/YogitaPatil5/End_to_End_Imdb_sentiment_analysis_project). 