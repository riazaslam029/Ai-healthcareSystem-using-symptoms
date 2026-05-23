# AI-Based Health Symptom Checker and Disease Prediction

This project builds a symptom-based disease prediction model and exposes it through a simple Streamlit web UI. The model is trained on a public symptoms-to-disease dataset and returns the top-3 predicted diseases with probabilities and recommended precautions.

## Features
- Text-based symptom input
- ML model with TF-IDF + Logistic Regression
- Accuracy and confusion matrix evaluation
- Streamlit UI with top-3 predictions and precautions

## Project Structure
- app.py: Streamlit UI
- src/train.py: Training and evaluation
- src/data_utils.py: Data loading and cleaning
- data/: Dataset metadata and precautions
- models/: Saved model artifacts
- outputs/: Metrics and plots

## Setup
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python -m src.train
```

## Run the app
```bash
streamlit run app.py
```

## Dataset
Place your CSV at data/dataset.csv. The loader supports:
- Text format: columns like `symptoms` + `disease`
- Binary format: many symptom columns (0/1) + label column like `prognosis`
- Multi-column format: `symptom_1`, `symptom_2`, ... + label column like `disease`

See data/dataset_info.md for details.
