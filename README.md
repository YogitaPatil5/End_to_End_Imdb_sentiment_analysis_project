# 🎬 End-to-End IMDB Sentiment Analysis Project

A complete end-to-end sentiment analysis pipeline for IMDB movie reviews using LSTM, TensorFlow, and Streamlit.

[GitHub Repository](https://github.com/YogitaPatil5/End_to_End_Imdb_sentiment_analysis_project)

---

## 🚀 Project Workflow

1. **Data Ingestion**
   - Downloads the IMDB dataset from Kaggle using `kagglehub`.
   - Handles credentials and robust error handling.

2. **Data Preprocessing**
   - Cleans and normalizes text (removes HTML, punctuation, stopwords, etc.).
   - Tokenizes and pads sequences (maxlen=200).
   - Saves the tokenizer for later use.

3. **Model Training**
   - Builds and trains an LSTM neural network for sentiment classification.
   - Saves the trained model (`sentiment_model.h5`).

4. **Model Evaluation**
   - Evaluates the model on test data.
   - Reports accuracy and other metrics.

5. **Interactive Web App**
   - Streamlit app (`app/app.py`) for real-time sentiment prediction.
   - Loads the trained model and tokenizer.
   - Provides a user-friendly interface with sample reviews and confidence scores.

---

## 🐳 Docker Support

- The included `Dockerfile` allows you to build and run the entire project in a containerized environment.
- **Why use Docker?**
  - Ensures consistent dependencies and environment across machines.
  - Makes deployment and sharing easy.

**To build and run with Docker:**
```bash
docker build -t imdb-sentiment .
docker run -p 8501:8501 imdb-sentiment
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YogitaPatil5/End_to_End_Imdb_sentiment_analysis_project.git
   cd End_to_End_Imdb_sentiment_analysis_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle credentials:**
   - Download `kaggle.json` from your Kaggle account.
   - Place it in `C:/Users/<YourUsername>/.kaggle/kaggle.json` (Windows).

4. **Run the pipeline:**
   ```bash
   python src/main.py
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```

---

## 📁 Project Structure

```
End_to_End_Imdb_sentiment_analysis_project/
│
├── app/
│   └── app.py
├── data/
│   └── raw/
├── saved_models/
│   ├── sentiment_model.h5
│   └── tokenizer.pickle
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ✨ Features

- End-to-end ML pipeline: ingestion → preprocessing → training → evaluation → deployment
- Robust error handling and logging
- Modern, interactive web UI with Streamlit
- Docker support for easy deployment

---

## 📜 License

MIT License

---

**Happy analyzing!**