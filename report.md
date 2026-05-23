# AI-Based Health Symptom Checker and Disease Prediction System

## Introduction
This project builds an AI-based system that predicts possible diseases from user-reported symptoms. It demonstrates a full ML pipeline including data acquisition, preprocessing, model training, evaluation, and a simple web UI for predictions.

## Problem Statement
Given a set of user-reported symptoms, predict the most likely disease and provide top-3 predictions with confidence scores and basic precautions.

## Dataset
- Source: Kaggle - itachi9604/disease-symptom-description-dataset
- Link: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- Structure: Symptom columns (symptom_1 ... symptom_n) with a disease label column.

## Data Preprocessing
- Column normalization and removal of unnamed columns
- Convert symptom columns into a comma-separated symptom text
- Text cleanup (lowercasing, punctuation removal)
- Removal of empty rows and duplicates

## Model / Algorithm
- TF-IDF vectorization of symptom text
- Logistic Regression classifier (multiclass)

## Training & Testing
- Train/test split: 80/20 with stratification
- Model trained on TF-IDF features
- Evaluation on hold-out test set

## Results & Evaluation
- Accuracy recorded in outputs/metrics.json
- Confusion matrix saved to outputs/confusion_matrix.png
- Classification report stored in outputs/metrics.json

## User Interface
- Streamlit web app in app.py
- Inputs symptom text
- Outputs top-3 disease predictions with probabilities and precautions

## Conclusion
The system demonstrates a practical symptom-based disease prediction workflow. Future improvements can include richer data, additional models, and more detailed clinical guidance.
