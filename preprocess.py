import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Automatically download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def clean_text(text):
    """
    Cleans the input text by applying standard NLP preprocessing steps.
    
    Steps:
    1. Lowercase the text.
    2. Remove URLs.
    3. Remove HTML tags.
    4. Remove numbers.
    5. Remove punctuation and special characters.
    6. Tokenize text into words.
    7. Remove English stopwords (e.g., 'the', 'is', 'in').
    8. Lemmatize words to their root form (e.g., 'running' -> 'run').
    9. Join back into a single string.
    """
    # 1. Convert to lowercase
    text = str(text).lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove punctuation/special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 6. Tokenize (split into words)
    words = nltk.word_tokenize(text)
    
    # 7 & 8. Remove stopwords and Lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]
    
    # 9. Reconstruct cleaned text
    return ' '.join(cleaned_words)
