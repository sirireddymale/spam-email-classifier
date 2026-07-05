# Spam Email Classifier

A machine learning model that classifies SMS messages as spam or ham (not spam),
built with Python and scikit-learn.

## How it works
- Dataset: SMS Spam Collection (5,572 messages)
- Text is converted to numeric features using CountVectorizer
- Model: Multinomial Naive Bayes
- Achieved 98.3% accuracy on test data

## How to run
1. Install dependencies: `pip install pandas scikit-learn`
2. Run: `python spam_classifier.py`

## Example output
'Congratulations! You won a free iPhone, click here now!' --> spam
'Hey, are we still meeting for lunch today?' --> ham
