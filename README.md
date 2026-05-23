# 🩺 AI-Based Health Symptom Checker & Disease Prediction

<img src="https://img.icons8.com/fluency/96/000000/robot-2--v2.png" width="60" align="right" alt="AI Healthcare Icon">

Welcome to the **AI HealthCare System** — an intelligent application designed to predict diseases based on user-provided symptoms. Powered by machine learning, it features an intuitive [Streamlit](https://streamlit.io/) interface and real-time precaution tips.

---

## 🚀 Features

- 🤖 **AI-Powered Disease Prediction**  
  Input symptoms as text; receive the top 3 ranked disease predictions within seconds!
- 🧑‍⚕️ **Precautionary Tips**  
  Get actionable medical precautions along with predictions.
- 🧹 **Smart Data Processing**  
  Automatic dataset cleaning and preparation for best model performance.
- 🔬 **ML Pipeline**  
  Baseline pipeline using TF-IDF + Logistic Regression.

---

## 🏃 Quick Start

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**

   ```bash
   python -m src.train
   ```

3. **Launch the App**

   ```bash
   streamlit run app.py
   ```

---

## 📁 Dataset Format

- Place your CSV at `data/dataset.csv`.
- Supported formats:
  - **Text format**: columns like `symptoms`, `disease`
  - **Binary format**: many symptom columns (0/1) + a label like `prognosis`
  - **Multi-column format**: `symptom_1`, `symptom_2`, ... + a label column

ℹ️ More details in [`data/dataset_info.md`](data/dataset_info.md).

---

## 📊 Outputs

| File                                       | Description             |
|---------------------------------------------|-------------------------|
| `outputs/confusion_matrix.png`              | Model performance plot  |
| `outputs/cleaned_dataset.csv`               | Cleaned, formatted data |
| `models/symptom_model.joblib`               | Trained model           |
| `models/tfidf_vectorizer.joblib`            | Text vectorizer         |

---

## 🗂 Project Structure

```
├── app.py                  # Streamlit UI
├── src/
│   ├── train.py            # Model training and evaluation
│   └── data_utils.py       # Data loading & cleaning
├── data/                   # Dataset files & precautions
├── models/                 # Saved ML artifacts
├── outputs/                # Metrics and plots
```

---

## ⚠️ Disclaimer

This project is for **educational use only** and **not** intended as medical advice.  
For any health concerns, always consult a qualified medical professional.

---

<p align="center">
  <img src="https://img.icons8.com/fluency/48/000000/health-graph.png"/>
  <img src="https://img.icons8.com/fluency/48/000000/artificial-intelligence.png"/>
  <img src="https://img.icons8.com/fluency/48/000000/doctor-male.png"/>
</p>
