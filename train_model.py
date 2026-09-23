import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import our custom modules
from preprocess import clean_text
from evaluate_model import evaluate_and_save_metrics

DATA_PATH = os.path.join('data', 'spam.csv')
MODEL_DIR = 'models'

def main():
    print("Starting Training Pipeline...")

    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}.")
        print("Please download the SMS Spam Collection dataset, rename it to spam.csv, and place it in the data/ folder.")
        return

    try:
        # Some CSVs use different encodings, latin-1 is common for this dataset
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Automatically rename 'v1'/'v2' to 'label'/'message' if they exist
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Keep only required columns
    if 'label' in df.columns and 'message' in df.columns:
        df = df[['label', 'message']]
    else:
        print("Error: The dataset must contain 'label' and 'message' columns (or 'v1' and 'v2').")
        return

    # 2. Data Exploration & Cleaning
    original_count = len(df)
    df = df.dropna(subset=['label', 'message'])
    df = df.drop_duplicates(subset=['message'])
    new_count = len(df)
    
    # Map labels to standardize them just in case
    df['label'] = df['label'].str.lower().str.strip()

    print("Dataset loaded successfully.")
    print(f"Total messages: {new_count} (Removed {original_count - new_count} duplicates/missing)")
    
    spam_count = len(df[df['label'] == 'spam'])
    ham_count = len(df[df['label'] == 'ham'])
    
    print(f"Spam messages: {spam_count} ({spam_count/new_count*100:.1f}%)")
    print(f"Not Spam (Ham) messages: {ham_count} ({ham_count/new_count*100:.1f}%)")
    
    # 3. Text Preprocessing
    print("\nCleaning and preprocessing text data. This may take a moment...")
    df['cleaned_message'] = df['message'].apply(clean_text)

    # 4. Train/Test Split
    X = df['cleaned_message']
    y = df['label']

    # Use stratify=y to ensure the train and test sets have the same proportion of spam/ham
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 5. Feature Extraction (TF-IDF)
    print("\nExtracting features using TF-IDF...")
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000) # limit to 5000 most frequent words to prevent overfitting
    
    # Fit and transform the training data, only transform the test data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 6. Model Training
    print("Training Logistic Regression model...")
    # Initialize Logistic Regression
    # max_iter is increased in case it needs more time to converge
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 7. Model Evaluation
    print("Predicting on test set and evaluating...")
    y_pred = model.predict(X_test_tfidf)
    
    # Use our custom evaluation script
    evaluate_and_save_metrics(y_test, y_pred, output_dir='outputs')

    # 8. Save Models
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, 'spam_classifier.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Pipeline complete. Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
