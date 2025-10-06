import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# --- 1. Setup and Model Caching (The Engine Room) ---

# We use FinBERT (ProsusAI/finbert) as it's fine-tuned for financial text.
MODEL_NAME = "ProsusAI/finbert" 

# CRITICAL: Caching prevents the large model from reloading on every user interaction
@st.cache_resource
def load_finbert_pipeline():
    """Loads the FinBERT model and tokenizer into a Hugging Face pipeline."""
    try:
        # Load the specialized FinBERT model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Create the text classification pipeline
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading FinBERT model: {e}")
        return None

# Load the classifier pipeline
classifier = load_finbert_pipeline()


# --- 2. Interface and User Input (The Dashboard) ---

st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="centered")
st.title("Financial News Sentiment Analyzer 📈")
st.caption("A quick prototype using Python, Streamlit, and FinBERT for domain-specific AI analysis.")


# User Input (Default text relevant to a financial/ratings context)
default_text = "Fitch's credit outlook was positive, citing strong liquidity and debt service capabilities."
user_input = st.text_area(
    "Enter a Financial News Headline or Text for Analysis:", 
    default_text,
    height=150
)

# Button to trigger the analysis
if st.button("Analyze Sentiment", type="primary"):
    
    # Error Handling
    if classifier is None:
        st.error("Model failed to load. Please check installation dependencies.")
        st.stop()
        
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # --- 3. Analysis and Output (The Results Interpreter) ---
        
        # Run inference
        with st.spinner('Analyzing with FinBERT...'):
            # The pipeline runs the tokenization and inference steps
            results = classifier(user_input)[0]

        
        st.subheader("Analysis Result")
        
        # Determine the dominant score
        dominant_label = results['label']
        dominant_score = results['score']
        
        # Map label to a visual emoji
        emoji_map = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '🟡'}
        emoji = emoji_map.get(dominant_label, '❓')
        
        # Display the main finding
        st.markdown(f"**Overall Sentiment:** {emoji} **{dominant_label}**")
        st.markdown(f"**Confidence:** `{dominant_score:.4f}` ({dominant_score*100:.2f}%)")

        st.divider()
        
        # Display all scores in a DataFrame for a clean, professional look
        st.write("Detailed Probability Scores:")
        
        # Extract scores and format for table display (using Pandas)
        scores = {item['label']: item['score'] for item in results}
        scores_df = pd.DataFrame(scores, index=['Probability']).T
        scores_df['Probability'] = (scores_df['Probability'] * 100).round(2).astype(str) + '%'
        
        st.dataframe(scores_df, use_container_width=True)
