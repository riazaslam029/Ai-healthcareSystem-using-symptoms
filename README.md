# 🩺 AI-Based Health Symptom Checker & Disease Prediction

<img src="https://img.icons8.com/fluency/96/000000/robot-2--v2.png" width="60" align="right" alt="AI Healthcare Icon">

Welcome to the **AI HealthCare System** — an intelligent machine learning application that predicts diseases based on user-provided symptoms. Built with **Python**, **Scikit-learn**, and **Streamlit**, it provides real-time disease predictions along with precautionary recommendations through an intuitive web interface.

---

## 🌐 Live Demo

**🚀 Try the Application:**  
https://ai-healthcaresystem-using-symptoms-fdcjyrq9ssajzncvsvvhbg.streamlit.app/

**💻 GitHub Repository:**  
https://github.com/riazaslam029/Ai-healthcareSystem-using-symptoms

---

## 🚀 Features

- 🤖 **AI-Powered Disease Prediction**  
  Input symptoms as text and receive the **Top 3 most likely disease predictions** within seconds.

- 🧑‍⚕️ **Precautionary Recommendations**  
  Get practical health precautions associated with the predicted diseases.

- 🧹 **Smart Data Processing**  
  Automatic dataset cleaning and preprocessing for improved prediction accuracy.

- 🔬 **Machine Learning Pipeline**  
  Built using **TF-IDF Vectorization** and **Logistic Regression** for reliable text classification.

- ⚡ **Interactive Web Interface**  
  Clean and responsive Streamlit application for an enhanced user experience.

---

## 🏃 Quick Start

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python -m src.train
```

### 3️⃣ Launch the Application

```bash
streamlit run app.py
```

---

## 📁 Dataset Format

Place your dataset inside:

```
data/dataset.csv
```

Supported formats:

- **Text Format** → `symptoms`, `disease`
- **Binary Format** → Symptom columns (0/1) + `prognosis`
- **Multi-column Format** → `symptom_1`, `symptom_2`, ... + disease label

More details are available in:

```
data/dataset_info.md
```

---

## 📊 Outputs

| File | Description |
|------|-------------|
| `outputs/confusion_matrix.png` | Model performance visualization |
| `outputs/cleaned_dataset.csv` | Cleaned dataset |
| `models/symptom_model.joblib` | Trained ML model |
| `models/tfidf_vectorizer.joblib` | TF-IDF vectorizer |

---

## 🗂 Project Structure

```
├── app.py                  # Streamlit Web App
├── src/
│   ├── train.py            # Model Training
│   └── data_utils.py       # Data Processing
├── data/                   # Dataset & Precautions
├── models/                 # Saved ML Models
├── outputs/                # Evaluation Results
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This application is developed **for educational and research purposes only**.

It should **not** be considered a substitute for professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare professional for medical concerns.

---

<p align="center">
  <img src="https://img.icons8.com/fluency/48/000000/health-graph.png"/>
  <img src="https://img.icons8.com/fluency/48/000000/artificial-intelligence.png"/>
  <img src="https://img.icons8.com/fluency/48/000000/doctor-male.png"/>
</p>

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.

Your support helps others discover the project and motivates future improvements.
