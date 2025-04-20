import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pickle
import joblib
import streamlit as st
import os

# Download necessary NLTK resources
# These are required for text preprocessing (stopwords, lemmatization, and tokenization)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize the lemmatizer to reduce words to their base form
# and load English stopwords (common words like 'the', 'a', 'an' that don't add meaning)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text data by performing the following operations:
    1. Convert to lowercase
    2. Remove HTML tags
    3. Remove special characters and numbers
    4. Tokenize text into individual words
    5. Lemmatize words (reduce to base form)
    6. Remove stopwords and short words
    7. Join tokens back into a single string
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text)
    # Lemmatize tokens and filter out stopwords and short words (length <= 2)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

def prepare_data(file_path):
    """
    Load and prepare the dataset for sentiment analysis:
    1. Read CSV file into pandas DataFrame
    2. Convert sentiment labels to binary (positive=1, negative=0)
    3. Preprocess the review text
    
    Args:
        file_path (str): Path to the CSV dataset file
        
    Returns:
        pandas.DataFrame: Prepared dataset with processed reviews
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)
    
    # Convert sentiment labels to binary values
    # 'positive' becomes 1, 'negative' becomes 0
    data['sentiment_binary'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    print("Preprocessing reviews...")
    # Apply text preprocessing to each review
    data['processed_review'] = data['review'].apply(preprocess_text)
    
    return data

def train_model(data):
    """
    Train and evaluate an XGBoost model for sentiment classification:
    1. Split data into training and testing sets
    2. Convert text to TF-IDF features
    3. Train XGBoost classifier
    4. Evaluate model performance
    5. Save model and vectorizer for later use
    
    Args:
        data (pandas.DataFrame): Prepared dataset with processed reviews
        
    Returns:
        tuple: (trained model, TF-IDF vectorizer)
    """
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_review'], 
        data['sentiment_binary'], 
        test_size=0.2, 
        random_state=42  # For reproducibility
    )
    
    print("Vectorizing text data...")
    # Convert text to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
    # Limit to 10,000 most important features to reduce dimensionality
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Save the vectorizer for later use in predictions
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    
    print("\nTraining XGBoost...")
    # Initialize XGBoost classifier with specified parameters
    # n_estimators=100: Use 100 trees in the ensemble
    # n_jobs=-1: Use all available CPU cores for parallel processing
    # use_label_encoder=False: Avoid label encoding warning
    model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    
    # Train the model on the training data
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate and display model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model to a file for later use
    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - XGBoost')
    plt.savefig('confusion_matrix.png')
    
    return model, tfidf_vectorizer

def predict_sentiment(review_text, model, vectorizer):
    """
    Predict sentiment for a new review text:
    1. Preprocess the input text
    2. Convert to TF-IDF features using the saved vectorizer
    3. Make prediction using the saved model
    
    Args:
        review_text (str): Raw review text
        model: Trained XGBoost model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (sentiment prediction, confidence probability)
    """
    # Preprocess the input text
    processed_review = preprocess_text(review_text)
    
    # Convert processed text to TF-IDF features
    review_vector = vectorizer.transform([processed_review])
    
    # Make prediction
    prediction = model.predict(review_vector)[0]
    
    # Get confidence probability for the prediction
    probability = model.predict_proba(review_vector)[0][prediction]
    
    # Return sentiment label and confidence probability
    return "Positive" if prediction == 1 else "Negative", probability

def main():
    """
    Main function to either:
    1. Load existing model and vectorizer if available
    2. Or train a new model if no saved model exists
    
    Returns:
        tuple: (model, vectorizer)
    """
    file_path = 'IMDB Dataset.csv'
    
    # Check if pre-trained model and vectorizer exist
    if os.path.exists('xgboost_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        print("Loading pre-trained model...")
        model = pickle.load(open('xgboost_model.pkl', 'rb'))
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    
    # If no pre-trained model exists, train a new one
    print("Loading and preparing data...")
    data = prepare_data(file_path)
    print(f"Dataset shape: {data.shape}")
    return train_model(data)

def app():
    """
    Streamlit web application for sentiment analysis:
    1. Set up web interface with tabs
    2. Allow users to input movie reviews
    3. Display sentiment predictions and confidence
    4. Show model performance metrics
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="wide"
    )
    
    # Set page title and description
    st.title("🎬 Movie Review Sentiment Analysis")
    st.write("Analyze movie review sentiment using XGBoost model")
    
    # Create tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Analyze", "Performance", "About"])
    
    # Analysis tab - main functionality
    with tab1:
        # Text input for movie review
        review = st.text_area("Enter your movie review:", height=150)
        
        # Button to trigger analysis
        if st.button("Analyze Sentiment", type="primary"):
            if not review:
                st.warning("Please enter a review")
            else:
                with st.spinner("Analyzing..."):
                    # Load or train model
                    if os.path.exists('xgboost_model.pkl'):
                        model = pickle.load(open('xgboost_model.pkl', 'rb'))
                        vectorizer = joblib.load('tfidf_vectorizer.pkl')
                    else:
                        model, vectorizer = main()
                    
                    # Get sentiment prediction and display results
                    sentiment, confidence = predict_sentiment(review, model, vectorizer)
                    st.metric("Sentiment", sentiment)
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Show the processed text
                    with st.expander("See processed text"):
                        st.write(preprocess_text(review))
    
    # Performance tab - model metrics
    with tab2:
        # Display confusion matrix if available
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption='XGBoost Confusion Matrix')
        
        # Display model details
        st.write("""
        ## Model Details
        - **Algorithm**: XGBoost Classifier
        - **Accuracy**: ~89-90% on test set
        - **Vectorization**: TF-IDF with 10,000 features
        - **Preprocessing**: Text cleaning, lemmatization, stopword removal
        """)
    
    # About tab - information about the app
    with tab3:
        st.write("""
        ## About This App
        Powered by XGBoost machine learning model trained on 50,000 IMDB movie reviews.
        Developed for sentiment analysis demonstration.
        """)

# Entry point for the script
if __name__ == '__main__':
    app()
