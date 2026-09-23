# Spam Message Classifier

## Project Overview
This is a beginner-friendly, complete machine learning pipeline that classifies SMS and email messages as **Spam** or **Not Spam (Ham)**. It uses Natural Language Processing (NLP) to process text and a Logistic Regression model to make predictions.

## Features
- **Data Preprocessing**: Cleans text by removing URLs, HTML, punctuation, and stopwords, and applies lemmatization.
- **Machine Learning**: Uses TF-IDF Vectorization for feature extraction and a Logistic Regression classifier for high accuracy.
- **Interactive UI**: A Streamlit web application allows users to test the model dynamically and see confidence/probability scores.
- **Batch Prediction**: Upload a CSV of messages and get predictions for all of them at once.

## Technologies Used
- **Python 3.10+**
- **Pandas & NumPy** (Data manipulation)
- **NLTK** (Natural Language Toolkit for text processing)
- **Scikit-learn** (Machine learning algorithms and evaluation)
- **Matplotlib & Seaborn** (Data visualization)
- **Streamlit** (Web application framework)
- **Joblib** (Model serialization)

## Dataset
This project uses the official **SMS Spam Collection** dataset. 
You must download this dataset before training the model.

1. Go to the UCI Machine Learning Repository or Kaggle to find the "SMS Spam Collection Dataset".
2. Download the CSV file.
3. Rename the file to `spam.csv`.
4. Place `spam.csv` inside the `data/` directory.

The dataset should have two main columns containing the label (`v1` or `label`) and the message (`v2` or `message`). The code automatically handles `v1`/`v2` column names.

## Project Architecture
1. **Dataset**: Loaded from `data/spam.csv`.
2. **Preprocessing**: Text is cleaned and standardized.
3. **TF-IDF**: Converts text into numerical features based on word frequency and importance.
4. **Logistic Regression**: A linear model used to separate Spam from Not Spam based on the TF-IDF features.
5. **Evaluation**: Calculates accuracy, precision, recall, and F1-score. Generates a confusion matrix.
6. **Streamlit App**: The interactive frontend to use the trained model.

## Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the machine learning model, run:
```bash
python train_model.py
```
This script will:
- Load and clean the dataset.
- Train the TF-IDF vectorizer and Logistic Regression model.
- Save the trained models in the `models/` folder.
- Save the evaluation report and confusion matrix in the `outputs/` folder.

## Run Streamlit

To launch the web application, run:
```bash
streamlit run app.py
```

## Evaluation
The model is evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Out of all messages predicted as spam, how many were actually spam? (Crucial to avoid filtering important messages).
- **Recall**: Out of all actual spam messages, how many did the model find?
- **F1-score**: The harmonic mean of Precision and Recall.
- **Confusion Matrix**: A table showing True Positives, True Negatives, False Positives, and False Negatives.

## Sample Predictions
- **Input:** *"Congratulations! You have won a free lottery prize. Call now to claim."*
  **Output:** 🚨 SPAM
- **Input:** *"Hey, are we meeting at college tomorrow?"*
  **Output:** ✅ NOT SPAM

## Project Structure
- `app.py`: The Streamlit web application.
- `train_model.py`: Script to train and save the machine learning model.
- `preprocess.py`: Contains the text cleaning functions used by both training and the app.
- `evaluate_model.py`: Contains functions to generate performance metrics and graphs.
- `requirements.txt`: Python package dependencies.
- `data/`: Folder to place the `spam.csv` dataset.
- `models/`: Folder where the trained `.pkl` models are saved.
- `outputs/`: Folder where evaluation reports and graphs are saved.
- `notebooks/`: Jupyter notebooks for data exploration.

## Future Improvements
- Implement **DistilBERT** or **BERT** for contextual embeddings.
- Add **multilingual spam detection**.
- Build an **email integration** to scan real inboxes.
- Expand the dataset to include more modern spam.
- Add cloud deployment instructions (AWS/Heroku/Streamlit Cloud).
