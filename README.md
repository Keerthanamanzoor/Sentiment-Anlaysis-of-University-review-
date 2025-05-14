# University Review Sentiment Analysis

This project analyzes and classifies university reviews based on sentiment using various machine learning models. It includes data preprocessing, text vectorization (BoW, TF-IDF), and evaluation of several classifiers.

## 📄 Project Overview

The goal is to classify reviews into three sentiment categories:
- **Positive (1)**
- **Neutral (0)**
- **Negative (-1)**

The dataset contains **206 university reviews**, with distribution:
- Positive: 54.4%
- Neutral: 23.8%
- Negative: 21.8%

## 🧰 Libraries Used

- `pandas`, `numpy`
- `nltk`, `re`, `string`
- `matplotlib`, `seaborn`, `wordcloud`
- `scikit-learn`
- `xgboost`

## 🔍 Workflow

1. **Data Exploration and Visualization**
   - Sentiment distribution plot
   - Word clouds for overall and sentiment-specific reviews

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation, digits, special characters
   - Handling negations
   - Tokenization
   - Stopword removal
   - Lemmatization

3. **Text Representation**
   - Bag of Words (BoW)
   - TF-IDF

4. **Model Training and Evaluation**
   - Models Tested:
     - Multinomial Naive Bayes
     - Support Vector Machine (SVM)
     - Logistic Regression
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Decision Tree
   - Evaluation Metrics:
     - Accuracy
     - Precision, Recall, F1-Score
     - Confusion Matrix

5. **Prediction on New Reviews**
   - Predicts sentiment of unseen reviews using the trained model

## 📈 Results

| Model              | Accuracy (%) |
|-------------------|--------------|
| Naive Bayes        | 57.14        |
| SVM (Linear)       | 73.81        |
| Logistic Regression| 73.81        |
| Random Forest      | 76.19        |
| KNN                | 64.29        |
| Decision Tree      | 69.05        |

**Best performing model: Random Forest**

## 🔮 Example Predictions

```text
Review: Great environment and exposure  
Predicted Sentiment: Positive

Review: Terrible alumni support and career development services  
Predicted Sentiment: Negative

Review: It gets worse with new decisions  
Predicted Sentiment: Neutral
