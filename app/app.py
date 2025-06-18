"""
IMDB Sentiment Analysis Web Application

This Streamlit application provides a user interface for sentiment analysis of movie reviews.
Features:
1. Real-time sentiment prediction
2. Confidence score display
3. Sample reviews for testing
4. Error handling and user feedback
5. Modern, responsive UI

The application uses a trained LSTM model to classify reviews as positive or negative.
"""

import os
import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants for model and tokenizer paths
MODEL_PATH = "saved_models/sentiment_model.h5"
TOKENIZER_PATH = "saved_models/tokenizer.pickle"

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure the Streamlit page
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"  # Use wide layout for better space utilization
)

# Custom CSS for styling the application
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction box styling */
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Positive prediction styling */
    .positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    /* Negative prediction styling */
    .negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data for sentiment analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove HTML tags
    3. Remove special characters and numbers
    4. Remove extra whitespace
    
    Args:
        text (str): Raw text input
    
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove HTML tags that might be present in reviews
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace and normalize spacing
    text = ' '.join(text.split())
    
    return text

@st.cache_resource
def load_model_and_tokenizer() -> tuple:
    """
    Load the trained model and tokenizer with caching.
    
    This function is cached by Streamlit to avoid reloading on every rerun.
    It includes error handling and user feedback for both model and tokenizer loading.
    
    Returns:
        tuple: (model, tokenizer) if successful, (None, None) if loading fails
    """
    model, tokenizer = None, None
    
    # Load the trained model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    else:
        st.error(f"❌ Model file not found at {MODEL_PATH}")
    
    # Load the tokenizer
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            st.success("✅ Tokenizer loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading tokenizer: {str(e)}")
    else:
        st.error(f"❌ Tokenizer file not found at {TOKENIZER_PATH}")
    
    return model, tokenizer

# Load model and tokenizer at startup
model, tokenizer = load_model_and_tokenizer()

# Stop execution if model or tokenizer failed to load
if not model or not tokenizer:
    st.error("🚫 Cannot proceed without model and tokenizer. Please ensure both files exist and are valid.")
    st.stop()

# Create sidebar with information about the application
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app analyzes the sentiment of movie reviews using a trained LSTM model.
    
    **How to use:**
    1. Enter your movie review in the text area
    2. Click 'Analyze' to get the sentiment prediction
    3. View the confidence score and prediction
    
    **Model Information:**
    - Architecture: LSTM Neural Network
    - Training Data: IMDB Movie Reviews (50K reviews)
    - Accuracy: ~85-90% on test set
    """)

# Create main content layout with two columns
col1, col2 = st.columns([2, 1])

# Right column: Model statistics and sample reviews
with col2:
    st.subheader("📈 Model Statistics")
    if model:
        st.metric("Model Parameters", f"{model.count_params():,}")
        st.metric("Model Layers", len(model.layers))
    
    st.subheader("🧪 Sample Reviews")
    # Predefined sample reviews for testing
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",
        "Terrible movie. Boring plot, bad acting, and a complete waste of time. I want my money back.",
        "It was okay. Not great, not terrible. Just average entertainment.",
        "Amazing cinematography and brilliant performances by all actors. A masterpiece!",
        "I fell asleep halfway through. The story was confusing and the characters were unlikable."
    ]
    
    # Create buttons for sample reviews
    for i, sample in enumerate(sample_reviews, 1):
        if st.button(f"Sample {i}", key=f"sample_{i}"):
            st.session_state.review = sample
            st.experimental_rerun()

# Left column: Main review input and analysis
with col1:
    st.subheader("📝 Enter Your Movie Review")
    # Text input area for the review
    review = st.text_area(
        "Write your review here...",
        height=200,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot was engaging...",
        key="review"
    )
    
    # Analyze button with full width
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            with st.spinner("🤖 Analyzing sentiment..."):
                try:
                    # Preprocess the review text
                    cleaned_review = clean_text(review)
                    
                    # Convert text to sequence and pad
                    sequence = tokenizer.texts_to_sequences([cleaned_review])
                    padded = pad_sequences(sequence, maxlen=200, padding="post")
                    
                    # Make prediction
                    prediction = model.predict(padded, verbose=0)[0][0]
                    
                    # Determine sentiment and confidence
                    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    confidence_percent = confidence * 100
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Show sentiment with appropriate styling
                    if prediction > 0.5:
                        st.markdown(
                            f'<div class="prediction-box positive"><h3>🎉 {sentiment}</h3></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box negative"><h3>😔 {sentiment}</h3></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Show confidence metrics
                    st.metric("Confidence", f"{confidence_percent:.1f}%")
                    st.metric("Raw Score", f"{prediction:.4f}")
                    
                    # Show text statistics
                    st.info(f"📝 **Original Review:** {len(review)} characters")
                    st.info(f"🧹 **Cleaned Text:** {len(cleaned_review)} characters")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and TensorFlow</p>
    <p>Dataset: IMDB Movie Reviews | Model: LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)