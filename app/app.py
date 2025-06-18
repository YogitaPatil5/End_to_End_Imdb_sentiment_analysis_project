import os
import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

MODEL_PATH = "saved_models/sentiment_model.h5"  # Updated to match training script
TOKENIZER_PATH = "saved_models/tokenizer.pickle"

# Page config
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
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

st.markdown('<h1 class="main-header">🎬 IMDB Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Text preprocessing function (same as in training)
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

# Try loading model and tokenizer with error handling
@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching."""
    model, tokenizer = None, None
    model_load_error, tokenizer_load_error = None, None

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            model_load_error = str(e)
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}"

    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            st.success("✅ Tokenizer loaded successfully!")
        except Exception as e:
            tokenizer_load_error = str(e)
    else:
        tokenizer_load_error = f"Tokenizer file not found at {TOKENIZER_PATH}"

    if model_load_error:
        st.error(f"❌ Error loading model: {model_load_error}")
    if tokenizer_load_error:
        st.error(f"❌ Error loading tokenizer: {tokenizer_load_error}")

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

if not model or not tokenizer:
    st.error("🚫 Cannot proceed without model and tokenizer. Please run the training script first.")
    st.stop()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app analyzes the sentiment of movie reviews using a trained LSTM model.
    
    **How to use:**
    1. Enter a movie review in the text area
    2. Click 'Analyze' to get the sentiment prediction
    3. View the confidence score and prediction
    
    **Model Info:**
    - Architecture: LSTM Neural Network
    - Dataset: IMDB Movie Reviews (50K reviews)
    - Accuracy: ~85-90%
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📈 Model Statistics")
    if model:
        st.metric("Model Parameters", f"{model.count_params():,}")
        st.metric("Model Layers", len(model.layers))
    st.subheader("🧪 Sample Reviews")
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",
        "Terrible movie. Boring plot, bad acting, and a complete waste of time. I want my money back.",
        "It was okay. Not great, not terrible. Just average entertainment.",
        "Amazing cinematography and brilliant performances by all actors. A masterpiece!",
        "I fell asleep halfway through. The story was confusing and the characters were unlikable."
    ]
    for i, sample in enumerate(sample_reviews, 1):
        if st.button(f"Sample {i}", key=f"sample_{i}"):
            st.session_state.review = sample
            st.experimental_rerun()

with col1:
    st.subheader("📝 Enter Your Movie Review")
    review = st.text_area(
        "Write your review here...",
        height=200,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot was engaging...",
        key="review"
    )
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            with st.spinner("🤖 Analyzing sentiment..."):
                try:
                    cleaned_review = clean_text(review)
                    sequence = tokenizer.texts_to_sequences([cleaned_review])
                    padded = pad_sequences(sequence, maxlen=200, padding="post")
                    prediction = model.predict(padded, verbose=0)[0][0]
                    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    confidence_percent = confidence * 100
                    st.subheader("📊 Analysis Results")
                    if prediction > 0.5:
                        st.markdown(f'<div class="prediction-box positive"><h3>🎉 {sentiment}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-box negative"><h3>😔 {sentiment}</h3></div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence_percent:.1f}%")
                    st.metric("Raw Score", f"{prediction:.4f}")
                    st.info(f"📝 **Original Review:** {len(review)} characters")
                    st.info(f"🧹 **Cleaned Text:** {len(cleaned_review)} characters")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and TensorFlow</p>
    <p>Dataset: IMDB Movie Reviews | Model: LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)