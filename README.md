# Mental-Health-Predictor
This repository contains a Python-based project that predicts mental health challenges (e.g., depression, anxiety) using logistic regression and natural language processing (NLP). It leverages clinical notes, PHQ (Patient Health Questionnaire) scores, and GAD (Generalized Anxiety Disorder) scores to make predictions. The project is designed for reusability and can be adapted for similar datasets.
Features

Preprocesses clinical notes by cleaning and vectorizing text using TF-IDF.
Combines numerical data (PHQ and GAD scores) with text features for model training.
Trains a logistic regression model to predict mental health challenges.
Provides a reusable prediction function for new data.
Includes tools to save and reuse the trained model and vectorizer.

Files
mental_health_nlp_project.py: The main codebase, including data preprocessing, model training, evaluation, and prediction functions.
logistic_model.pkl: Saved logistic regression model for future predictions.
tfidf_vectorizer.pkl: Saved TF-IDF vectorizer for text preprocessing.

Requirements
Python 3.8+
Libraries: pandas, numpy, scikit-learn, re, joblib

Installation

Clone the repository:
git clone https://github.com/FortuneIgboanugo/mental-health-predictor.git
cd mental-health-predictor

Install required libraries:
pip install -r requirements.txt

Place your dataset in the working directory if using custom data.

Usage
Training the Model
-To train the model on a dataset:

Replace the sample dataset in mental_health_nlp_project.py with your dataset, ensuring the following columns are present:
-clinical_notes: Text data from clinical notes.
-PHQ_score: Numerical PHQ scores.
-GAD_score: Numerical GAD scores.
-mental_health_challenge: Target column (1 for challenge, 0 for no challenge).

Run the script to train the model and evaluate its performance:
python mental_health_nlp_project.py

Making Predictions
Use the predict_mental_health function to predict mental health challenges for new data:

new_data_notes = [
    "Patient reports severe anxiety and frequent panic attacks.",
    "No significant mental health concerns."
]
new_data_PHQ = [12, 3]
new_data_GAD = [10, 2]

predictions = predict_mental_health(new_data_notes, new_data_PHQ, new_data_GAD)
print("Predictions:", predictions)

Reusing the Model
The trained model and vectorizer are saved as logistic_model.pkl and tfidf_vectorizer.pkl for future use:

from joblib import load

model = load('logistic_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

Example
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Predictions for new data: [1 0]

Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.
