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
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text data"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(processed_tokens)

def prepare_data(file_path):
    """Load and prepare the dataset"""
    data = pd.read_csv(file_path)
    data['sentiment_binary'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    print("Preprocessing reviews...")
    data['processed_review'] = data['review'].apply(preprocess_text)
    return data

def train_model(data):
    """Train and evaluate XGBoost model"""
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_review'], 
        data['sentiment_binary'], 
        test_size=0.2, 
        random_state=42
    )
    
    print("Vectorizing text data...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    
    print("\nTraining XGBoost...")
    model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    # Confusion matrix
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
    """Predict sentiment for a new review"""
    processed_review = preprocess_text(review_text)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)[0]
    probability = model.predict_proba(review_vector)[0][prediction]
    return "Positive" if prediction == 1 else "Negative", probability

def main():
    file_path = 'IMDB Dataset.csv'
    
    if os.path.exists('xgboost_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        print("Loading pre-trained model...")
        model = pickle.load(open('xgboost_model.pkl', 'rb'))
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    
    print("Loading and preparing data...")
    data = prepare_data(file_path)
    print(f"Dataset shape: {data.shape}")
    return train_model(data)

def app():
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Review Sentiment Analysis")
    st.write("Analyze movie review sentiment using XGBoost model")
    
    tab1, tab2, tab3 = st.tabs(["Analyze", "Performance", "About"])
    
    with tab1:
        review = st.text_area("Enter your movie review:", height=150)
        if st.button("Analyze Sentiment", type="primary"):
            if not review:
                st.warning("Please enter a review")
            else:
                with st.spinner("Analyzing..."):
                    if os.path.exists('xgboost_model.pkl'):
                        model = pickle.load(open('xgboost_model.pkl', 'rb'))
                        vectorizer = joblib.load('tfidf_vectorizer.pkl')
                    else:
                        model, vectorizer = main()
                    
                    sentiment, confidence = predict_sentiment(review, model, vectorizer)
                    st.metric("Sentiment", sentiment)
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    with st.expander("See processed text"):
                        st.write(preprocess_text(review))
    
    with tab2:
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption='XGBoost Confusion Matrix')
        st.write("""
        ## Model Details
        - **Algorithm**: XGBoost Classifier
        - **Accuracy**: ~89-90% on test set
        - **Vectorization**: TF-IDF with 10,000 features
        - **Preprocessing**: Text cleaning, lemmatization, stopword removal
        """)
    
    with tab3:
        st.write("""
        ## About This App
        Powered by XGBoost machine learning model trained on 50,000 IMDB movie reviews.
        Developed for sentiment analysis demonstration.
        """)

if __name__ == '__main__':
    app()
