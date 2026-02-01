# 🛡️ LoanGuard AI - Intelligent Loan Default Prediction

An AI-powered risk scoring platform that helps financial institutions predict loan defaults with precision, offering clear, actionable insights for confident lending decisions.

---

## 📌 Problem Statement

Traditional credit scoring lacks transparency and explainability. Manual risk assessment is time-consuming, error-prone, and struggles to meet regulatory compliance requirements. High default rates continue to impact profitability across the lending industry.

## 💡 Our Solution

LoanGuard AI uses a **Random Forest Classifier** trained on historical lending data to estimate the probability of a borrower defaulting. The system provides:
- Real-time risk assessments in milliseconds
- Full explainability with feature importance breakdowns
- Regulatory-compliant audit trails
- Customizable thresholds for portfolio management

---

## 🏗️ Project Structure

```
Saint-Dnyaneshwar-Maharaj_PS17/
│
├── app.py                      # Main Streamlit application (Frontend + Backend Integration)
├── train_model.py              # Model training script (Random Forest & XGBoost)
├── generate_synthetic_data.py  # Script to generate synthetic loan data
├── run_data_cleaning.py        # Data preprocessing pipeline
├── verify_analysis.py          # Data verification and analysis utilities
│
├── Data_analysis.ipynb         # Exploratory Data Analysis notebook
├── Data_cleaning.ipynb         # Data cleaning and preprocessing notebook
│
├── Cleaned_LoanApproval.csv    # Processed dataset used for training
├── synthetic_loan_data.csv     # Raw synthetic loan data
│
├── loan_model_rf.pkl           # Trained Random Forest model
├── scaler.pkl                  # StandardScaler for feature normalization
├── model_columns.pkl           # Feature column names for inference
│
└── README.md                   # This file
```

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Risk Calculator** | Multi-section form to assess individual loan applications |
| **Dashboard** | Portfolio-level KPIs, risk distribution charts, and trend analysis |
| **Real-Time Inference** | Sub-second predictions using trained ML models |
| **Explainable AI** | Feature importance visualization for every decision |
| **Responsive UI** | Modern, mobile-friendly interface built with Streamlit |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit, Custom CSS, Plotly |
| **Backend** | Python 3.x |
| **Machine Learning** | Scikit-Learn (Random Forest), XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/devyanshujasud-ai/Saint-Dnyaneshwar-Maharaj_PS17.git
   cd Saint-Dnyaneshwar-Maharaj_PS17
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost plotly joblib
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   Open your browser and navigate to: `http://localhost:8501`

---

## 📊 Model Performance

| Metric | Random Forest | XGBoost |
|--------|---------------|---------|
| Accuracy | 100% | 100% |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| ROC-AUC | 1.000 | 1.000 |

> **Note:** High accuracy is due to synthetic data with clear decision boundaries. Real-world performance may vary.

---

## 📖 Usage Guide

### Risk Calculator
1. Navigate to the **Calculator** page from the top menu.
2. Fill in applicant details across three sections:
   - Personal Information (Income, Employment, etc.)
   - Loan Details (Amount, Term, Interest Rate)
   - Credit History (FICO Score, DTI, etc.)
3. Click **Assess Risk** to generate a risk score.
4. Review the detailed explanation and recommendation.

### Dashboard
- View portfolio-level statistics including total loans, risk distribution, and trends.
- Analyze risk breakdown by loan grade.

---

## 🧪 Retraining the Model

To retrain the model with new data:

```bash
python train_model.py
```

This will:
- Load `Cleaned_LoanApproval.csv`
- Train both Random Forest and XGBoost models
- Save the best model to `loan_model_rf.pkl`
- Save the scaler and feature columns

---

## 📁 Data Pipeline

```
synthetic_loan_data.csv
        ↓
run_data_cleaning.py
        ↓
Cleaned_LoanApproval.csv
        ↓
train_model.py
        ↓
loan_model_rf.pkl + scaler.pkl + model_columns.pkl
        ↓
app.py (Inference)
```

---

## 👥 Team

Developed for the **Saint Dnyaneshwar Maharaj Hackathon (PS17)**.

---

## 📄 License

This project is open-source and available under the MIT License.
