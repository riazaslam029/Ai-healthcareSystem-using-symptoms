import csv
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import streamlit as st

MODEL_PATH = os.path.join("models", "symptom_model.joblib")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
PRECAUTIONS_PATH = os.path.join("data", "precautions.json")
PRECAUTIONS_CSV_PATH = os.path.join("data", "symptom_precaution.csv")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def train_model_if_missing():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return True

    if st.session_state.get("auto_trained"):
        return False

    st.session_state["auto_trained"] = True
    with st.spinner("Training model (first run)..."):
        result = subprocess.run(
            [sys.executable, "-m", "src.train"],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        st.error("Model training failed. Please check the logs.")
        if result.stderr:
            st.code(result.stderr)
        return False
    return True


def load_precautions():
    if not os.path.exists(PRECAUTIONS_PATH):
        if os.path.exists(PRECAUTIONS_CSV_PATH):
            return _load_precautions_csv(PRECAUTIONS_CSV_PATH)
        return {
            "_default": [
                "Rest and stay hydrated",
                "Monitor symptoms",
                "Seek medical advice if symptoms persist",
            ]
        }
    with open(PRECAUTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_precautions_csv(path):
    precautions = {
        "_default": [
            "Rest and stay hydrated",
            "Monitor symptoms",
            "Seek medical advice if symptoms persist",
        ]
    }
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row.get("Disease") or row.get("disease")
            if not disease:
                continue
            items = []
            for key in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
                value = row.get(key) or ""
                value = value.strip()
                if value:
                    items.append(value)
            if items:
                precautions[disease.strip()] = items
    return precautions


def format_top_k(labels, probs, k=3):
    top_idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in top_idx]


def main():
    st.set_page_config(page_title="Symptom Checker", page_icon="+", layout="centered")
    st.title("AI-Based Health Symptom Checker")
    st.write("Enter symptoms separated by commas. Example: fever, headache, fatigue")

    model, vectorizer = load_artifacts()
    if model is None:
        if not train_model_if_missing():
            st.warning(
                "Model artifacts not found. Please train the model first by running: python -m src.train"
            )
            return
        model, vectorizer = load_artifacts()
        if model is None:
            st.error("Model artifacts still missing after training.")
            return

    precautions = load_precautions()

    symptoms_input = st.text_area("Symptoms", height=120)
    if st.button("Predict"):
        if not symptoms_input.strip():
            st.error("Please enter at least one symptom.")
            return

        X = vectorizer.transform([symptoms_input])
        probs = model.predict_proba(X)[0]
        labels = model.classes_
        top3 = format_top_k(labels, probs, k=3)

        st.subheader("Top 3 Predictions")
        for disease, prob in top3:
            st.write(f"**{disease}** - {prob:.2%}")
            recs = precautions.get(disease, precautions.get("_default", []))
            if recs:
                st.caption("Precautions: " + "; ".join(recs))


if __name__ == "__main__":
    main()
