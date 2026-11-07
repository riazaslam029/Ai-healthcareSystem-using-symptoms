<div align="center">
  <img src="https://img.icons8.com/fluency/96/medical-doctor.png" alt="AI Healthcare" width="84" />
  <h1>AI Health Symptom Checker</h1>
  <p>Machine-learning powered disease prediction from user-provided symptoms.</p>

  <p>
    <a href="https://ai-healthcaresystem-using-symptoms-fdcjyrq9ssajzncvsvvhbg.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live demo" /></a>
    <a href="https://github.com/riazaslam029/Ai-healthcareSystem-using-symptoms"><img src="https://img.shields.io/github/stars/riazaslam029/Ai-healthcareSystem-using-symptoms?style=for-the-badge&logo=github&label=Stars" alt="GitHub stars" /></a>
    <a href="https://github.com/riazaslam029/Ai-healthcareSystem-using-symptoms/issues"><img src="https://img.shields.io/github/issues/riazaslam029/Ai-healthcareSystem-using-symptoms?style=for-the-badge&logo=github" alt="Open issues" /></a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+" />
    <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/Status-Educational-2EA44F?style=flat-square" alt="Educational project" />
  </p>
</div>

> **Important:** This project is intended for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for health concerns.

## ✨ Overview

**AI Health Symptom Checker** is a lightweight Python application that transforms a user’s symptom description into a ranked list of likely conditions. It combines TF-IDF text features with Logistic Regression and presents the top three predictions and related precautionary guidance through an interactive Streamlit interface.

## 🚀 Features

- 🤖 **Top-three predictions** from a natural-language symptom description
- 🧠 **TF-IDF + Logistic Regression** classification pipeline
- 🧹 **Dataset normalization** for supported symptom/disease CSV formats
- 🩺 **Precautionary guidance** loaded from JSON or CSV data
- 📊 **Training evaluation** with accuracy, classification metrics, and a confusion matrix
- ⚡ **Automatic first-run training** when model artifacts are unavailable
- 🖥️ **Simple Streamlit UI** for quick local or hosted demos

## 🌐 Live Demo

Try the deployed application: **[AI Health Symptom Checker](https://ai-healthcaresystem-using-symptoms-fdcjyrq9ssajzncvsvvhbg.streamlit.app/)**

## 🛠️ Technology Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| User interface | Streamlit |
| Machine learning | scikit-learn, TF-IDF, Logistic Regression |
| Data processing | pandas, NumPy |
| Model persistence | joblib |
| Evaluation and visualization | Matplotlib, Seaborn |

## ⚙️ Getting Started

### Prerequisites

- Python 3.9 or newer
- `pip`

### Installation

```bash
git clone https://github.com/riazaslam029/Ai-healthcareSystem-using-symptoms.git
cd Ai-healthcareSystem-using-symptoms
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Train the model

```bash
python -m src.train
```

Training creates the serialized model and vectorizer in `models/`, together with cleaned data and evaluation outputs in `outputs/`.

### Launch the application

```bash
streamlit run app.py
```

If model artifacts are missing, the application attempts to train the model automatically on its first run.

## 🧪 Example Input

Enter symptoms as comma-separated text, for example:

```text
fever, headache, fatigue
```

The application returns the three highest-probability predictions and displays available precautions for each result.

## 📁 Project Structure

```text
├── app.py                         # Streamlit application
├── data/
│   ├── dataset.csv                # Training data
│   ├── precautions.json           # Precautionary guidance
│   └── dataset_info.md            # Dataset format notes
├── models/
│   ├── symptom_model.joblib       # Trained classifier
│   └── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer
├── outputs/
│   ├── cleaned_dataset.csv        # Normalized training data
│   ├── metrics.json               # Evaluation metrics
│   └── confusion_matrix.png       # Evaluation visualization
├── src/
│   ├── data_utils.py              # Dataset loading and cleaning
│   └── train.py                   # Training and evaluation pipeline
├── requirements.txt
└── README.md
```

## 📊 Supported Dataset Formats

The data loader supports common layouts including:

- `symptoms` + `disease` text columns
- Binary symptom columns with a `prognosis` label
- Multiple symptom columns such as `symptom_1`, `symptom_2`, and a disease label

See [`data/dataset_info.md`](data/dataset_info.md) for the expected structure and preprocessing notes.

## 🤝 Contributing

Contributions are welcome. Open an issue for a bug or feature request, or submit a pull request with a clear description of the change and any relevant validation results.

## 📄 License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the software, subject to the license terms.

<div align="center">
  <br />
  <strong>Built with Python, Streamlit, and scikit-learn.</strong>
  <br />
  <a href="https://github.com/riazaslam029/Ai-healthcareSystem-using-symptoms">⭐ Star the repository if you find it useful</a>
</div>
