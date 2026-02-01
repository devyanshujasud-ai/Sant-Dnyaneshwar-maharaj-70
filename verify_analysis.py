import pandas as pd
import numpy as np

# Load Cleaned Data
try:
    df = pd.read_csv('Cleaned_LoanApproval.csv')
    print("Loaded Cleaned_LoanApproval.csv")
    
    # Logic from Data_analysis.ipynb
    
    # Strip term
    if 'term' in df.columns:
        df['term'] = df['term'].str.strip()
    
    # Filter loan_status
    df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]
    
    # Replace infinite values
    if 'DTI Ratio %' in df.columns:
        # Check for infinite
        max_val = df.loc[~df['DTI Ratio %'].isin([np.inf, -np.inf]), 'DTI Ratio %'].max()
        df['DTI Ratio %'].replace([np.inf, -np.inf], max_val, inplace=True)
        
    if 'payment_to_income_ratio (%)' in df.columns:
        max_val_pim = df.loc[~df['payment_to_income_ratio (%)'].isin([np.inf, -np.inf]), 'payment_to_income_ratio (%)'].max()
        df['payment_to_income_ratio (%)'].replace([np.inf, -np.inf], max_val_pim, inplace=True)
        
    # Rename term
    df.rename(columns={'term': 'term (months)'}, inplace=True)
    if 'term (months)' in df.columns and df['term (months)'].dtype == object:
        df['term (months)'] = df['term (months)'].str.replace(' months', '')
        df['term (months)'] = df['term (months)'].astype(int)
        
    print("Analysis logic executed successfully.")
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")
