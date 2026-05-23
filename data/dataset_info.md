# Dataset Information

Source: Kaggle - itachi9604/disease-symptom-description-dataset
Link: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

## Structure
The dataset can be in one of three formats:
- Text format: a symptom text column and a disease label column
- Binary format: multiple symptom columns (0/1) and a disease label column
- Multi-column format: symptom_1, symptom_2, ... and a disease label column
This project uses the following fields:
- Symptoms: each column is a symptom (0/1)
- prognosis: the target label

The training pipeline reads data from data/dataset.csv and writes a cleaned copy
to outputs/cleaned_dataset.csv.
