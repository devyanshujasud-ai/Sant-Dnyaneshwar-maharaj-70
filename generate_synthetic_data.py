import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=10000, output_file='synthetic_loan_data.csv'):
    np.random.seed(42)
    random.seed(42)

    data = {}

    # IDs
    data['id'] = np.arange(1000000, 1000000 + num_samples)

    # Loan Amount
    data['loan_amnt'] = np.random.randint(1000, 40000, size=num_samples)
    
    # Term
    data['term'] = np.random.choice([' 36 months', ' 60 months'], size=num_samples, p=[0.7, 0.3])

    # Interest Rate (will be correlated with grade later)
    # Base interest rate
    base_int_rate = np.random.uniform(5, 25, size=num_samples)
    data['int_rate'] = np.round(base_int_rate, 2)

    # Installment (approximation based on loan amount and interest rate)
    # Simple amortization calculation just for plausible values: (r * P) / (1 - (1+r)^-n)
    # r = monthly rate, n = months
    def calc_installment(row):
        P = row['loan_amnt']
        r = row['int_rate'] / 100 / 12
        n = 36 if '36' in row['term'] else 60
        if r == 0: return P / n
        return (r * P) / (1 - (1 + r)**(-n))

    # Grade and Subgrade
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    subgrades = [f"{g}{i}" for g in grades for i in range(1, 6)]
    
    # Assign grades based on interest rate (higher rate -> worse grade)
    # This is a simplification
    def assign_grade(int_rate):
        if int_rate < 8: return 'A'
        elif int_rate < 11: return 'B'
        elif int_rate < 14: return 'C'
        elif int_rate < 18: return 'D'
        elif int_rate < 22: return 'E'
        elif int_rate < 26: return 'F'
        else: return 'G'
    
    data['grade'] = [assign_grade(x) for x in data['int_rate']]
    
    # Randomly assign subgrade within grade
    data['sub_grade'] = [f"{g}{random.randint(1, 5)}" for g in data['grade']]

    # Employment Title
    emp_titles = ['Manager', 'Teacher', 'Supervisor', 'RN', 'Project Manager', 'Director', 'Engineer', 'Sales', 'Driver']
    data['emp_title'] = np.random.choice(emp_titles + [np.nan], size=num_samples, p=[0.1]*9 + [0.1])

    # Employment Length
    emp_lengths = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
    data['emp_length'] = np.random.choice(emp_lengths + [np.nan], size=num_samples)

    # Home Ownership
    home_ownerships = ['MORTGAGE', 'RENT', 'OWN', 'ANY']
    data['home_ownership'] = np.random.choice(home_ownerships, size=num_samples, p=[0.5, 0.4, 0.09, 0.01])

    # Annual Income (log-normal distribution)
    data['annual_inc'] = np.random.lognormal(mean=np.log(60000), sigma=0.6, size=num_samples)
    
    # Verification Status
    verification_statuses = ['Verified', 'Source Verified', 'Not Verified']
    data['verification_status'] = np.random.choice(verification_statuses, size=num_samples)

    # Issue Date (random dates between 2015 and 2018)
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2018, 12, 31)
    days_diff = (end_date - start_date).days
    random_days = np.random.randint(0, days_diff, size=num_samples)
    data['issue_d'] = [(start_date + timedelta(days=int(d))).strftime('%b-%Y') for d in random_days]

    # Purpose
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business', 'car', 'vacation', 'moving', 'house', 'wedding', 'renewable_energy', 'educational']
    weights = [0.55, 0.20, 0.06, 0.06, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002]
    # Adjust weights to sum to 1
    weights = np.array(weights) / np.sum(weights)
    data['purpose'] = np.random.choice(purposes, size=num_samples, p=weights)

    # Title (often similar to purpose or NaN)
    data['title'] = data['purpose'] # Simplification
    
    # Zip Code (partial)
    data['zip_code'] = [f"{random.randint(100, 999)}xx" for _ in range(num_samples)]

    # Addr State
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 'NC', 'MI', 'MA', 'MD', 'AZ', 'WA', 'CO', 'MN', 'TN', 'MO', 'CT', 'IN', 'NV', 'OR', 'WI', 'AL', 'SC', 'LA', 'KY', 'OK', 'UT', 'KS', 'AR', 'HI', 'NM', 'MS', 'NH', 'RI', 'WV', 'ID', 'DE', 'MT', 'AK', 'DC', 'SD', 'WY', 'VT', 'ME', 'ND', 'NE']
    # Approximate population weights could be better, but uniform random for now
    data['addr_state'] = np.random.choice(states, size=num_samples)

    # DTI (Debt to Income)
    data['dti'] = np.random.uniform(0, 40, size=num_samples)

    # Earliest Credit Line (random dates well before issue date)
    start_cr_date = datetime(1990, 1, 1)
    end_cr_date = datetime(2010, 12, 31)
    days_cr_diff = (end_cr_date - start_cr_date).days
    random_cr_days = np.random.randint(0, days_cr_diff, size=num_samples)
    data['earliest_cr_line'] = [(start_cr_date + timedelta(days=int(d))).strftime('%b-%Y') for d in random_cr_days]

    # FICO Score
    # Higher income somewhat correlated with higher FICO
    data['fico_range_low'] = np.random.randint(660, 850, size=num_samples)
    data['fico_range_high'] = data['fico_range_low'] + 4

    # Inquiries in last 6 months
    data['inq_last_6mths'] = np.random.choice(range(6), size=num_samples, p=[0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

    # Open Accounts
    data['open_acc'] = np.random.poisson(10, size=num_samples)

    # Public Records
    data['pub_rec'] = np.random.choice([0, 1, 2], size=num_samples, p=[0.85, 0.12, 0.03])

    # Revolving Balance
    data['revol_bal'] = np.random.randint(0, 50000, size=num_samples)

    # Revolving Utilization
    data['revol_util'] = np.random.uniform(0, 100, size=num_samples)

    # Total Accounts
    data['total_acc'] = data['open_acc'] + np.random.randint(0, 20, size=num_samples)

    # Initial List Status
    data['initial_list_status'] = np.random.choice(['w', 'f'], size=num_samples)
    
    # Application Type
    data['application_type'] = np.random.choice(['Individual', 'Joint App'], size=num_samples, p=[0.9, 0.1])
    
    # Mortgage Accounts
    data['mort_acc'] = np.random.choice(range(10), size=num_samples)
    
    # Public Record Bankruptcies
    data['pub_rec_bankruptcies'] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])

    df = pd.DataFrame(data)
    
    # Calculate Installment properly now that columns exist
    df['installment'] = df.apply(calc_installment, axis=1).round(2)

    # Target Variable Generation
    # Probability of default increases with:
    # - Lower FICO
    # - Higher DTI
    # - Higher Interest Rate
    # - Lower Annual Income
    
    # Normalize features for weighted sum
    norm_fico = (850 - df['fico_range_low']) / (850 - 660) # 0 (good) to 1 (bad)
    norm_dti = df['dti'] / 40
    norm_int_rate = (df['int_rate'] - 5) / 20
    
    # Weighted score for risk (higher is riskier)
    risk_score = (0.4 * norm_fico) + (0.3 * norm_int_rate) + (0.3 * norm_dti) + np.random.normal(0, 0.1, size=num_samples)
    
    # Threshold for default (top 20% risky)
    threshold = np.percentile(risk_score, 80)
    
    df['default_risk'] = (risk_score > threshold).astype(int)
    df['loan_status'] = df['default_risk'].apply(lambda x: 'Charged Off' if x == 1 else 'Fully Paid')

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {num_samples} records and saved to {output_file}")
    print(df['loan_status'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data()
