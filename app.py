import os
import joblib
import pandas as pd
import streamlit as st

from preprocess import clean_text

# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spam_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("About this Project")
    st.write(
        "This project uses Natural Language Processing (NLP) and Machine Learning "
        "to classify text messages as Spam or Not Spam."
    )
    
    st.header("How it works")
    st.markdown("""
    1. **Text preprocessing:** Cleans and lemmatizes the input text.
    2. **TF-IDF feature extraction:** Converts text into numerical features.
    3. **Logistic Regression classification:** Applies the trained ML model.
    4. **Probability-based prediction:** Outputs confidence scores.
    """)

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("📩 Spam Message Classifier")
st.subheader("AI-powered SMS & Email Spam Detection")

if model is None or vectorizer is None:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------
# Single Message Prediction
# ---------------------------------------------------------
st.markdown("### Test a Message")
user_input = st.text_area("Enter your message here:", height=150)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Predict")

if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess
        cleaned_text = clean_text(user_input)
        
        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        # Probabilities are ordered based on model.classes_ (usually ['ham', 'spam'])
        classes = model.classes_
        spam_idx = list(classes).index('spam')
        ham_idx = list(classes).index('ham')
        
        spam_prob = probabilities[spam_idx] * 100
        ham_prob = probabilities[ham_idx] * 100
        
        st.markdown("---")
        if prediction == 'spam':
            st.error(f"🚨 **SPAM**")
            st.write(f"**Confidence:** {spam_prob:.2f}%")
        else:
            st.success(f"✅ **NOT SPAM**")
            st.write(f"**Confidence:** {ham_prob:.2f}%")
            
        # Display probability breakdown
        st.write("### Probability Breakdown")
        st.write(f"- Spam: {spam_prob:.2f}%")
        st.write(f"- Not Spam: {ham_prob:.2f}%")

# ---------------------------------------------------------
# Examples Section
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Sample Messages")
st.write("Copy and paste these examples to test the classifier:")

with st.expander("🚨 Spam Examples"):
    st.code("Congratulations! You have won a free lottery prize. Call now to claim.")
    st.code("URGENT! You have won $5000. Click the link to receive your reward.")
    st.code("Claim your FREE mobile phone today by clicking this link.")

with st.expander("✅ Not Spam Examples"):
    st.code("Hey, are we meeting at college tomorrow?")
    st.code("Please send me the assignment when you get time.")
    st.code("Your Python class starts at 10 AM tomorrow.")

# ---------------------------------------------------------
# Optional: CSV Batch Prediction
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Batch Prediction (CSV Upload)")
st.write("Upload a CSV file containing a column named `message` to predict multiple messages at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'message' not in df.columns:
            st.error("The uploaded CSV must contain a column named 'message'.")
        else:
            st.write(f"Processing {len(df)} messages...")
            
            # Predict batch
            cleaned_messages = df['message'].apply(clean_text)
            vectorized_messages = vectorizer.transform(cleaned_messages)
            
            predictions = model.predict(vectorized_messages)
            
            spam_idx = list(model.classes_).index('spam')
            probabilities = model.predict_proba(vectorized_messages)[:, spam_idx] * 100
            
            # Create results dataframe
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['spam_confidence_%'] = probabilities.round(2)
            
            st.success("Batch prediction complete!")
            st.dataframe(results_df.head(10)) # Show preview
            
            # Allow download
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Predictions as CSV",
                data=csv,
                file_name='spam_predictions_results.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
