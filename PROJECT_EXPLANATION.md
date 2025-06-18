# 📚 Project Explanation: End-to-End IMDB Sentiment Analysis

## 📝 Problem Statement

Build an end-to-end machine learning pipeline to classify IMDB movie reviews as positive or negative using deep learning (LSTM) and deploy it as an interactive web app.

---

## 🗂️ Project Structure & Script Roles

### 1. **Data Ingestion (`src/data_ingestion.py`)**
- **Purpose:** Download the IMDB dataset from Kaggle using `kagglehub`.
- **Key Steps:**
  - Handles Kaggle API authentication.
  - Downloads and extracts the dataset.
  - Loads the CSV into pandas DataFrames for further processing.

### 2. **Data Preprocessing (`src/data_preprocessing.py`)**
- **Purpose:** Clean and prepare the text data for modeling.
- **Key Steps:**
  - Cleans text (removes HTML, punctuation, stopwords, etc.).
  - Tokenizes and pads sequences (maxlen=200).
  - Saves the tokenizer for use in the app.

### 3. **Model Training (`src/model_training.py`)**
- **Purpose:** Build and train the LSTM model.
- **Key Steps:**
  - Defines the LSTM architecture.
  - Trains the model on the preprocessed data.
  - Saves the trained model (`sentiment_model.h5`).

### 4. **Model Evaluation (`src/model_evaluation.py`)**
- **Purpose:** Evaluate the trained model's performance.
- **Key Steps:**
  - Predicts on the test set.
  - Calculates accuracy and other metrics.
  - Optionally displays a confusion matrix.

### 5. **Main Pipeline (`src/main.py`)**
- **Purpose:** Orchestrate the entire workflow.
- **Key Steps:**
  - Runs data ingestion, preprocessing, training, and evaluation in sequence.
  - Ensures all artifacts (model, tokenizer) are saved for deployment.

### 6. **Web App (`app/app.py`)**
- **Purpose:** Provide an interactive UI for sentiment prediction.
- **Key Steps:**
  - Loads the trained model and tokenizer.
  - Accepts user input and predicts sentiment in real time.
  - Shows confidence scores and sample reviews.

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

**For more details, see the [README](./README.md) or the [GitHub repo](https://github.com/YogitaPatil5/End_to_End_Imdb_sentiment_analysis_project).** 