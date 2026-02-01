import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the synthetic dataset
df = pd.read_csv('synthetic_loan_data.csv')
print("Loaded synthetic data.")

# Rename dti to DTI Ratio %
if 'dti' in df.columns:
    df.rename(columns={'dti': 'DTI Ratio %'}, inplace=True)

# Ensure integer columns are integers
int_columns = ['fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc']
for col in int_columns:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Mapping initial_list_status
status_mapping = {'w': 'Whole Funded', 'f': 'Fractional Funded'}
df['initial_list_status'] = df['initial_list_status'].replace(status_mapping)

# Credit Score Buckets
score_bins = [0, 579, 669, 739, 799, float('inf')]
score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
df['credit_score_bucket'] = pd.cut(df['fico_range_low'], bins=score_bins, labels=score_labels)

# Additional Features derived in original notebook
df['credit_score_range'] = df['fico_range_high'] - df['fico_range_low']
df['monthly_income'] = df['annual_inc'] / 12
df['payment_to_income_ratio (%)'] = (df['installment'] / df['monthly_income']) * 100

# Fill title with purpose if empty (logic from original)
df['title'] = df['title'].fillna(df['purpose'])

# Save to CSV
df.to_csv('Cleaned_LoanApproval.csv', index=False)
print("Cleaned data saved to 'Cleaned_LoanApproval.csv'")
