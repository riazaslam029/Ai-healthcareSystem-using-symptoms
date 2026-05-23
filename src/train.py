import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .data_utils import load_symptom_dataset

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["symptoms"],
        df["disease"],
        test_size=0.2,
        random_state=42,
        stratify=df["disease"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, vectorizer, acc, report, cm


def save_artifacts(model, vectorizer):
    joblib.dump(model, os.path.join(MODEL_DIR, "symptom_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))


def save_metrics(acc, report, cm, labels):
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train symptom checker model.")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to dataset CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
    df = (
        load_symptom_dataset(args.data, save_cleaned_path=cleaned_path)
        if args.data
        else load_symptom_dataset(save_cleaned_path=cleaned_path)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df["symptoms"],
        df["disease"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["disease"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    save_artifacts(model, vectorizer)
    save_metrics(acc, report, cm, model.classes_)

    print(f"Training complete. Accuracy: {acc:.4f}")
    print(f"Model saved to {MODEL_DIR}")
    print(f"Metrics saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
