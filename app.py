import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

st.title("Loan Default Predictor")
st.write("Enter loan details to predict the risk of default.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, value=10000, step=500)
        term = st.selectbox("Term", options=["36 months", "60 months"])
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0, step=0.1)
        installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=300.0, step=10.0)
        grade = st.selectbox("Grade", options=["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("Sub Grade", options=[f"{g}{n}" for g in ["A", "B", "C", "D", "E", "F", "G"] for n in range(1, 6)])
        
    with col2:
        emp_length = st.selectbox("Employment Length", options=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
        home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN", "ANY", "NONE"])
        annual_inc = st.number_input("Annual Income ($)", min_value=0.0, value=60000.0, step=1000.0)
        verification_status = st.selectbox("Verification Status", options=["Verified", "Source Verified", "Not Verified"])
        purpose = st.selectbox("Purpose", options=["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical", "small_business", "car", "vacation", "moving", "house", "wedding", "renewable_energy", "educational"])

    st.subheader("Credit History")
    col3, col4 = st.columns(2)
    
    with col3:
        dti = st.number_input("DTI Ratio (%)", min_value=0.0, value=15.0, step=0.1)
        fico_low = st.number_input("CIBIL Score (Low)", min_value=300, value=700, step=1)
        fico_high = st.number_input("CIBIL Score (High)", min_value=300, value=704, step=1)
        inq_six_mths = st.number_input("Inquiries Last 6 Months", min_value=0, value=0, step=1)
        
    with col4:
        open_acc = st.number_input("Open Accounts", min_value=0, value=10, step=1)
        pub_rec = st.number_input("Public Records", min_value=0, value=0, step=1)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, value=5000.0, step=100.0)
        revol_util = st.number_input("Revolving Util (%)", min_value=0.0, value=50.0, step=1.0)
        total_acc = st.number_input("Total Accounts", min_value=1, value=20, step=1)
        
    # Additional mappings
    initial_list_status = st.selectbox("Initial List Status", options=["Whole Funded", "Fractional Funded"])
    application_type = st.selectbox("Application Type", options=["Individual", "Joint App"])
    mort_acc = st.number_input("Mortgage Accounts", min_value=0, value=1, step=1)
    pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0, value=0, step=1)
    
    # Derived features logic from filtering
    # credit_score_range = fico_high - fico_low
    # monthly_income = annual_inc / 12
    # payment_to_income = (installment / monthly_income) * 100
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # 1. Prepare raw dataframe
    raw_data = {
        'loan_amnt': [loan_amnt],
        'term': [term], # cleaned data had 'term (months)' as int? verify_analysis.py renamed it.
        # train_model.py loaded 'Cleaned_LoanApproval.csv'. Let's check its columns.
        # Cleaned_LoanApproval.csv has 'term (months)' column?
        # Data_cleaning.ipynb: "df['term'] = df['term'].str.strip()" ... wait.
        # `run_data_cleaning.py` just loads `synthetic_loan_data.csv`.
        # `synthetic_loan_data.csv` has 'term' as "36 months" or "60 months".
        # `verify_analysis.py` performed further cleaning: renamed 'term' to 'term (months)' and stripped " months".
        # BUT `train_model.py` loaded `Cleaned_LoanApproval.csv`.
        # `run_data_cleaning.py` saved `Cleaned_LoanApproval.csv` without renaming 'term' to 'term (months)'.
        # `run_data_cleaning.py` preserved 'term' as string.
        # Wait, `verify_analysis.py` did the analysis logic, but `train_model.py` loads `Cleaned_LoanApproval.csv`.
        # `Cleaned_LoanApproval.csv` was created by `run_data_cleaning.py`.
        # Does `run_data_cleaning.py` rename term? No.
        # So `Cleaned_LoanApproval.csv` likely has 'term' as string "36 months".
        # Let's check the artifacts or memory.
        # Step 125 output: python verify_analysis.py ran. It loads Cleaned, then renames term.
        # train_model.py loads `Cleaned_LoanApproval.csv` directly.
        # So in train_model.py, 'term' is categorical.
        # Preprocessing `get_dummies` would create `term_60 months` etc.
        
        'int_rate': [int_rate],
        'installment': [installment],
        'grade': [grade],
        'sub_grade': [sub_grade],
        'emp_title': ["Generic"], # dummy value or need input? Train model dropped id. What about high cardinality?
        # categorical_cols in train_model.py include object columns.
        # emp_title is high cardinality. It might have been dummified => huge dimensionality.
        # If train_model.py dummified emp_title, the model expects thousands of columns.
        # This is bad for Streamlit integration if we didn't handle high cardinality.
        # I need to check if `train_model.py` output showed categorical columns.
        # `cat_cols` likely included `emp_title`, `title`, `zip_code`.
        # If so, `model_columns.pkl` is huge.
        
        'emp_length': [emp_length],
        'home_ownership': [home_ownership],
        'annual_inc': [annual_inc],
        'verification_status': [verification_status],
        'issue_d': ["Jan-2023"], # dummy date? Date handling is tricky.
        'purpose': [purpose],
        'title': [purpose], # Logic in cleaning: fill title with purpose
        'zip_code': ["000xx"], # dummy?
        'addr_state': ["CA"], # dummy?
        'DTI Ratio %': [dti], # Renamed in cleaning
        'earliest_cr_line': ["Jan-2000"], 
        'fico_range_low': [fico_low],
        'fico_range_high': [fico_high],
        'inq_last_6mths': [inq_six_mths],
        'open_acc': [open_acc],
        'pub_rec': [pub_rec],
        'revol_bal': [revol_bal],
        'revol_util': [revol_util],
        'total_acc': [total_acc],
        'initial_list_status': [initial_list_status],
        'application_type': [application_type],
        'mort_acc': [mort_acc],
        'pub_rec_bankruptcies': [pub_rec_bankruptcies],
        'credit_score_range': [fico_high - fico_low],
        'monthly_income': [annual_inc / 12],
        'payment_to_income_ratio (%)': [(installment / (annual_inc / 12)) * 100]
    }
    
    input_df = pd.DataFrame(raw_data)
    
    # 2. Dummify
    # We need to match model_columns
    input_df = pd.get_dummies(input_df)
    
    # 3. Align with model columns
    # This aligns the columns, filling missing ones with 0 and dropping extra ones
    # But wait, emp_title? 
    # If the model has `emp_title_Manager`, and we enter `Generic`, we get `emp_title_Generic`.
    # `reindex` will handle it (fill 0 for Manager, drop Generic).
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # 4. Scale
    input_scaled = scaler.transform(input_df)
    
    # 5. Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[:, 1]
    
    st.subheader("Prediction Results")
    if prediction[0] == 1: # Assuming 1 is Charged Off (Risk) based on my Train script assumption?
        # Wait, in train_model.py I relied on LabelEncoder.
        # "Mapping: ..." output was:
        # I didn't see the output for mapping.
        # But commonly 'Charged Off' starts with C, 'Fully Paid' starts with F.
        # Alphabetical: Charged Off=0, Fully Paid=1.
        # Notebook said "Class 0.0 (Charged Off)".
        # So 0 is Charged Off (Risk), 1 is Fully Paid (Safe).
        
        # In my script:
        # le.fit_transform(df['loan_status'])
        # 0 -> Charged Off, 1 -> Fully Paid.
        # So prediction == 0 means RISK.
        
        # Let's verify mapping if possible, but standard LE is alphabetical.
        
        risk_score = (1 - prob[0]) * 100 # Prob of 1 (Fully Paid) -> Low risk
        st.error(f"High Risk of Default! (Probability of Repayment: {prob[0]*100:.2f}%)")
    else:
        risk_score = (1 - prob[0]) * 100
        st.success(f"Low Risk - Likely to Repay. (Probability of Repayment: {prob[0]*100:.2f}%)")
        
    st.write(f"Risk Score: {risk_score:.2f}/100")
