import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('Cleaned_LoanApproval.csv')
    
    # Preprocessing
    print("Preprocessing...")
    
    # Encode target 'loan_status' (Fully Paid=1, Charged Off=0) - or vice versa. 
    # Usually we want to predict Default (Charged Off). 
    # Let's see: original notebook likely treated 'Charged Off' as the positive class or negative.
    # In Step 158 output: "Class 0.0 (Charged Off) has lower precision..." 
    # This implies Fully Paid is 1.0.
    
    # Let's map explicitly
    # df['loan_status'] is 'Fully Paid' or 'Charged Off'
    
    # We want to predict Default Risk. Let's make Charged Off = 1 (Risk), Fully Paid = 0 (No Risk).
    # Or follow the notebook's convention if it helps.
    # Notebook output: "Class 0.0 (Charged Off)"
    # So the notebook used Charged Off = 0, Fully Paid = 1.
    
    le = LabelEncoder()
    df['loan_status_num'] = le.fit_transform(df['loan_status'])
    # Check mapping
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Target Mapping: {mapping}")
    
    # Drop original target and non-feature columns
    X = df.drop(columns=['loan_status', 'loan_status_num', 'id', 'credit_score_bucket']) # bucket is derived from fico
    y = df['loan_status_num']
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Align columns (ensure no issues with future data)
    # For now, just proceed
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf_clf.predict(X_test_scaled)
    print("Random Forest Performance:")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC AUC: {roc_auc_score(y_test, rf_clf.predict_proba(X_test_scaled)[:, 1]):.3f}")
    
    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_clf.fit(X_train_scaled, y_train)
    
    y_pred_xgb = xgb_clf.predict(X_test_scaled)
    print("XGBoost Performance:")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC AUC: {roc_auc_score(y_test, xgb_clf.predict_proba(X_test_scaled)[:, 1]):.3f}")
    
    # Save best model (let's pick RF for simplicity if scores are close, or XGB if better)
    # Saving RF for now as it's often more robust out-of-the-box
    
    print("\nSaving Random Forest model and scaler...")
    joblib.dump(rf_clf, 'loan_model_rf.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Save column names to ensure input consistency in Streamlit
    joblib.dump(list(X.columns), 'model_columns.pkl')
    
    print("Models saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
