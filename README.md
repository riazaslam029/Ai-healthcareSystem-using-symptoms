# AI-Based Health Symptom Checker and Disease Prediction

A symptom-based disease prediction system with a simple Streamlit UI. The pipeline cleans a dataset, trains an ML model, and shows top-3 predictions with precaution tips.

## Highlights
- Symptom text input with top-3 ranked predictions
- TF-IDF + Logistic Regression baseline
- Dataset cleaning and preprocessing
- Optional precaution hints in the UI

## Quick Start
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python -m src.train
```

Run the app:
```bash
streamlit run app.py
```

## Dataset Format
Place your CSV at data/dataset.csv. Supported formats:
- Text format: columns like `symptoms` + `disease`
- Binary format: many symptom columns (0/1) + label column like `prognosis`
- Multi-column format: `symptom_1`, `symptom_2`, ... + label column like `disease`

Details and source notes are in data/dataset_info.md.

## Outputs
- outputs/confusion_matrix.png
- outputs/cleaned_dataset.csv
- models/symptom_model.joblib
- models/tfidf_vectorizer.joblib

## Project Layout
- app.py: Streamlit UI
- src/train.py: Training and evaluation
- src/data_utils.py: Data loading and cleaning
- data/: Dataset files and precautions
- models/: Saved model artifacts
- outputs/: Metrics and plots

## Notes
This project is for educational use only and is not medical advice.
