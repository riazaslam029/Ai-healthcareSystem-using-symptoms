import os
import re

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

TEXT_COLUMNS = ["symptoms", "symptom", "text", "input", "sentence"]
LABEL_COLUMNS = ["disease", "label", "output", "target", "prognosis"]


def _clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False)]
    return df


def _infer_columns(df):
    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
    return text_col, label_col


def _clean_symptom_value(value):
    value = str(value).strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def _symptoms_from_binary(row, symptom_columns):
    active = [col.replace("_", " ") for col in symptom_columns if row.get(col, 0) == 1]
    return ", ".join(active)


def _symptoms_from_values(row, symptom_columns):
    values = []
    for col in symptom_columns:
        val = row.get(col, "")
        if pd.isna(val):
            continue
        cleaned = _clean_symptom_value(val)
        if cleaned:
            values.append(cleaned)
    return ", ".join(values)


def _is_binary_symptom_matrix(df, symptom_columns):
    sample = df[symptom_columns].stack().dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    return sample.isin(["0", "1", "0.0", "1.0"]).all()


def load_symptom_dataset(dataset_path=DEFAULT_DATASET_PATH, save_cleaned_path=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Dataset not found. Place your CSV at data/dataset.csv "
            "or pass a custom path with --data."
        )

    df = pd.read_csv(dataset_path)
    df = _normalize_columns(df)

    text_col, label_col = _infer_columns(df)
    if label_col is None:
        raise ValueError(
            f"Could not infer label column. Found columns: {list(df.columns)}"
        )

    if text_col:
        df = df[[text_col, label_col]].rename(
            columns={text_col: "symptoms", label_col: "disease"}
        )
        df["symptoms"] = df["symptoms"].astype(str).map(_clean_text)
        df["disease"] = df["disease"].astype(str).str.strip()
        df = df[df["symptoms"].str.len() > 0]
        df = df[df["disease"].str.len() > 0]
        df = df.drop_duplicates(subset=["symptoms", "disease"]).reset_index(drop=True)
        if save_cleaned_path:
            df.to_csv(save_cleaned_path, index=False)
        return df

    symptom_columns = [c for c in df.columns if c != label_col]
    if _is_binary_symptom_matrix(df, symptom_columns):
        df["symptoms"] = df.apply(
            lambda row: _symptoms_from_binary(row, symptom_columns), axis=1
        )
    else:
        df["symptoms"] = df.apply(
            lambda row: _symptoms_from_values(row, symptom_columns), axis=1
        )

    df["disease"] = df[label_col].astype(str).str.strip().str.lower()
    df = df[df["symptoms"].str.len() > 0]
    df = df[df["disease"].str.len() > 0]
    df = df.drop_duplicates(subset=["symptoms", "disease"]).reset_index(drop=True)

    cleaned = df[["symptoms", "disease"]]
    if save_cleaned_path:
        cleaned.to_csv(save_cleaned_path, index=False)
    return cleaned
