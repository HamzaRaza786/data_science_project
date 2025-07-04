
# Causal Impact of Contributions on Member Churn

This project analyzes statutory health insurance data in Germany to estimate member churn based on contribution rates and fund-level risk profiles. It combines causal inference techniques with machine learning to understand and predict member behavior.
---

## 📁 Repository Overview

This repository provides a full workflow:
- **Data Preprocessing and EDA** 
- **Causal analysis** 
- **Predictive modeling**
- **Interactive app**
---

## 🏗️ Project Structure
```
├── .gitignore # Untracked file rules
├── app.py # Streamlit dashboard for prediction interface
├── ml_churn_code.py # Core ML and feature engineering logic
├── README.md # This file
├── requirements.txt # Python dependencies
│
├── data/ # Source and processed datasets
│ ├── 230807_Survey.xlsx
│ ├── Marktanteile je Kasse.xlsx
│ ├── Morbidity_Region.xlsx
│ ├── causal_effects.csv
│ ├── Zusatzbeitrag_je Kasse je Quartal.xlsx
│ └── preprocessed_for_ml_model.csv
│
├── ml_model/ # Serialized ML model
│ └── predictive_model_with_additional_params.pkl
│
└── notebooks/ # Jupyter notebooks for each stage of analysis
├── eda_and_causal_analysis.ipynb
├── predictive_model_analysis.ipynb
└── survey_analysis.ipynb
```

## ⚙️ Installation Instructions

Follow the steps below to set up your local environment.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd data_science_project

```
## Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

## Install Dependencies
```
pip install -r requirements.txt
```
## 🚀 Run the Streamlit App
Once setup is complete, launch the app:
```
streamlit run app.py
```
Also can locate the jupyter notebooks in the ```notebooks``` folder, and check the flow. Each notebook is descriptive.