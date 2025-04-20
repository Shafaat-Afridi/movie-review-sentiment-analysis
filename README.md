# Movie Review Sentiment Analysis

A machine learning application that analyzes movie reviews and predicts whether the sentiment is positive or negative using XGBoost classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## 🔍 Overview

This project implements a sentiment analysis system for movie reviews using Natural Language Processing (NLP) techniques and machine learning. The application allows users to input movie reviews and receive immediate feedback on whether the sentiment is positive or negative, with a confidence score.

## ✨ Features

- Text preprocessing pipeline with lemmatization and stopword removal
- TF-IDF vectorization for feature extraction (10,000 features)
- XGBoost classifier for sentiment prediction
- Interactive web interface built with Streamlit
- Confidence scores for predictions
- Visual representation of model performance with confusion matrix

## 💻 Technology Stack

- **Python**
- **Data Processing**: pandas, NumPy
- **NLP**: NLTK (Natural Language Toolkit)
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Data Persistence**: pickle, joblib

## 🚀 Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn xgboost joblib streamlit
   ```

3. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

4. Ensure the IMDB Dataset CSV file is in the project root directory
   - The dataset should contain 'review' and 'sentiment' columns

## 🔧 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will start and open in your default web browser.

### Using the Application

1. Enter a movie review in the text area
2. Click "Analyze Sentiment"
3. View the sentiment prediction (Positive/Negative) and confidence score
4. Explore the processed text and model performance in the other tabs

## 📊 Model Performance

The XGBoost model achieves:
- **Accuracy**: 85.69%
- **Precision**: 0.87 (negative), 0.88 (positive)
- **Recall**: 0.83 (negative), 0.88 (positive)
- **F1-Score**: 0.85 (negative), 0.86 (positive)
- **Support**: 4961 (negative), 5039 (positive)

The model shows balanced performance across both positive and negative classes, with slightly better performance on positive reviews.

## 📁 Project Structure

```
project/
│
├── app.py                  # Main application with Streamlit interface and model training
├── IMDB Dataset.csv        # Dataset containing movie reviews
├── xgboost_model.pkl       # Trained XGBoost model (created after first run)
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer (created after first run)
└── confusion_matrix.png    # Model performance visualization (created after first run)
```

## 🔮 Future Improvements

- Implement word embeddings for better feature representation
- Add explainability features to highlight influential words in reviews
- Extend beyond binary classification to predict rating scores
- Add aspect-based sentiment analysis for specific movie elements
- Create a batch processing option for multiple reviews
- Implement hyperparameter tuning to further improve model performance
