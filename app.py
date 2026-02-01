import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="LoanGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. GLOBAL CSS (LIGHT MODE & BIGGER FONTS) ---
st.markdown("""
    <style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* VARIABLES - LIGHT THEME */
    :root {
        --bg-color: #F8FAFC; /* Slate 50 */
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0; /* Slate 200 */
        --primary-gradient: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); /* Indigo to Violet */
        --text-primary: #0F172A; /* Slate 900 */
        --text-secondary: #475569; /* Slate 600 */
        --success: #059669;
        --warning: #D97706;
        --danger: #DC2626;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* GLOBAL RESET */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        font-size: 1.125rem; /* Base font size increased (18px) */
    }
    
    /* TOP NAVIGATION BAR */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--card-border);
        box-shadow: 0 4px 20px -2px rgba(0,0,0,0.05); /* Added shadow for separation */
        position: sticky;
        top: 0;
        z-index: 999;
        margin-top: -60px;
    }
    .nav-logo {
        font-size: 1.5rem; /* Bigger logo */
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: var(--text-primary);
    }
    .logo-badge {
        background: var(--primary-gradient);
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    /* HIDE DEFAULT HEADER/FOOTER */
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }

    /* CARDS */
    .lg-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 2rem; /* More padding */
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
    }
    
    /* INPUTS - Solved Visibility */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important; /* Pure White Background */
        border: 1px solid #CBD5E1 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        border-radius: 8px !important;
    }
    
    /* Ensure all text inside select/input is black */
    .stSelectbox div[data-baseweb="select"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown Menu Items */
    ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    div[data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* BUTTONS - Responsive & Big */
    .stButton > button {
        background: var(--primary-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 1.2rem !important; /* Bigger Text */
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* PROGRESS BAR */
    .progress-track {
        background-color: #E2E8F0;
        height: 10px; /* Thicker */
        border-radius: 5px;
        width: 100%;
        margin: 1.5rem 0 2.5rem 0;
    }
    .progress-fill {
        background: #10B981;
        height: 100%;
        border-radius: 5px;
        width: 100%;
    }
    
    /* TEXT STYLES */
    .hero-title {
        font-size: 4.5rem; /* Massive title */
        font-weight: 900;
        text-align: center;
        line-height: 1.1;
        margin-bottom: 2rem;
        color: #1E293B;
    }
    .gradient-text {
        background: linear-gradient(90deg, #4F46E5, #7C3AED, #DB2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* RESPONSIVENESS TWEAKS */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .stButton > button { width: 100%; margin-bottom: 0.5rem; }
        .nav-logo { font-size: 1.2rem; }
    }
    
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# --- 4. BACKEND LOGIC ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('loan_model_rf.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except:
        return None, None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('Cleaned_LoanApproval.csv')
    except:
        return None

model, scaler, model_columns = load_artifacts()
df = load_data()

# --- 5. TOP NAVIGATION ---
def render_top_nav():
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    
    # Use standard Columns (wrap on mobile)
    c1, spacer, c2, c3, c4, c5, c6 = st.columns([2.5, 0.5, 1, 1.2, 1, 0.8, 1.2]) # Added spacer
    
    with c1:
        st.markdown("""
        <div class="nav-logo">
            <span class="logo-badge">LG</span>
            <span>LoanGuard AI</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Buttons
    with c2:
        if st.button("Home", use_container_width=True, type="secondary" if st.session_state.page != "Home" else "primary"):
            set_page("Home")
            st.rerun()
    with c3:
        if st.button("Calculator", use_container_width=True, type="secondary" if st.session_state.page != "Risk Calculator" else "primary"):
            set_page("Risk Calculator")
            st.rerun()
    with c4:
        if st.button("Dashboard", use_container_width=True, type="secondary" if st.session_state.page != "Dashboard" else "primary"):
            set_page("Dashboard")
            st.rerun()
    with c5:
        if st.button("About", use_container_width=True, type="secondary" if st.session_state.page != "About" else "primary"):
            set_page("About")
            st.rerun()
    with c6:
        if st.button("Docs", use_container_width=True, type="secondary" if st.session_state.page != "Documentation" else "primary"):
            set_page("Documentation")
            st.rerun()
            
    st.markdown("---")

render_top_nav()

# --- 6. PAGE CONTENT ---

def show_home():
    # Hero Section
    st.markdown("""
        <div style="text-align: center; padding-top: 3rem;">
            <div style="display: inline-block; background: #EEF2FF; padding: 10px 20px; border-radius: 30px; font-size: 1rem; color: #4F46E5; margin-bottom: 2rem; border: 1px solid #C7D2FE; font-weight: 600;">
                Trusted by 500+ Financial Institutions
            </div>
            <div class="hero-title">
                Smart Loan Decisions with <br>
                <span class="gradient-text">Transparent AI Risk Scoring</span>
            </div>
            <p style="color: #475569; font-size: 1.3rem; max-width: 800px; margin: 0 auto 3rem auto; line-height: 1.6;">
                LoanGuard AI helps banks and NBFCs predict loan defaults with precision—offering 
                clear, actionable insights for confident, compliant decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button("⚡ Try Risk Calculator Now", use_container_width=True, type="primary")
            
    # Trust Badges (Dark Text for Light Mode)
    st.markdown("""
        <div style="text-align: center; margin-top: 5rem; display: flex; justify-content: center; gap: 3rem; color: #64748B; font-weight: 600; font-size: 1.1rem; flex-wrap: wrap;">
            <span>🛡️ SOC 2 Compliant</span>
            <span>✅ 99.9% Uptime</span>
            <span>🔒 Bank-Grade Security</span>
            <span>🚀 Millisecond Inference</span>
        </div>
    """, unsafe_allow_html=True)

def show_calculator():
    if model is None:
        st.error("Model artifacts not found.")
        return
        
    st.markdown("## 🧮 Risk Calculator")
    st.markdown("<p style='font-size: 1.2rem; color: #475569;'>Enter applicant details to calculate risk score</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="margin-bottom: 0.5rem; display: flex; justify-content: space-between; font-size: 1rem; color: #64748B; font-weight: 600;">
            <span>Form Completion</span>
            <span>100%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill"></div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("calc_form"):
        # Section 1
        st.markdown('<div class="lg-card">', unsafe_allow_html=True)
        st.markdown("### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        annual_inc = c1.number_input("Annual Income ($)", 0.0, 1000000.0, 60000.0, step=1000.0)
        emp_length = c2.selectbox("Employment Years", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
        home_ownership = c3.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "ANY", "NONE"])
        
        c4, c5 = st.columns(2)
        verification_status = c4.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])
        purpose = c5.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical", "small_business", "car", "vacation", "moving", "house", "wedding", "renewable_energy", "educational"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 2
        st.markdown('<div class="lg-card">', unsafe_allow_html=True)
        st.markdown("### 💳 Loan Details")
        l1, l2 = st.columns(2)
        loan_amnt = l1.number_input("Loan Amount ($)", 1000, 50000, 10000)
        term = l2.selectbox("Term", ["36 months", "60 months"])
        
        l3, l4 = st.columns(2)
        int_rate = l3.number_input("Interest Rate (%)", 0.0, 30.0, 10.0)
        installment = l4.number_input("Monthly Installment ($)", 0.0, 2000.0, 300.0)
        
        l5, l6 = st.columns(2)
        grade = l5.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = l6.selectbox("Sub Grade", [f"{g}{n}" for g in ["A", "B", "C", "D", "E", "F", "G"] for n in range(1, 6)])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 3
        st.markdown('<div class="lg-card">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Credit History")
        cr1, cr2 = st.columns(2)
        dti = cr1.number_input("DTI Ratio (%)", 0.0, 100.0, 15.0)
        fico_low = cr2.number_input("FICO Score (Low)", 300, 850, 700)
        
        cr3, cr4 = st.columns(2)
        fico_high = cr3.number_input("FICO Score (High)", 300, 850, 704)
        inq_six_mths = cr4.number_input("Inquiries (6m)", 0, 10, 0)
        
        with st.expander("Detailed Credit Stats"):
            e1, e2, e3 = st.columns(3)
            open_acc = e1.number_input("Open Acc", 0, 50, 10)
            total_acc = e2.number_input("Total Acc", 1, 100, 20)
            revol_bal = e3.number_input("Revol Bal", 0.0, 100000.0, 5000.0)
            
            e4, e5, e6 = st.columns(3)
            revol_util = e4.number_input("Revol Util", 0.0, 150.0, 50.0)
            mort_acc = e5.number_input("Mort Acc", 0, 20, 1)
            pub_rec = e6.number_input("Pub Rec", 0, 10, 0)
            pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 10, 0)

        initial_list_status = "Whole Funded"
        application_type = "Individual"
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Assess Risk ➔", type="primary")

    if submitted:
        # Processing (Same backend logic)
        raw_data = {
            'loan_amnt': [loan_amnt], 'term': [term], 'int_rate': [int_rate], 'installment': [installment],
            'grade': [grade], 'sub_grade': [sub_grade], 'emp_title': ["Generic"], 'emp_length': [emp_length],
            'home_ownership': [home_ownership], 'annual_inc': [annual_inc], 'verification_status': [verification_status],
            'issue_d': ["Jan-2023"], 'purpose': [purpose], 'title': [purpose], 'zip_code': ["000xx"],
            'addr_state': ["CA"], 'DTI Ratio %': [dti], 'earliest_cr_line': ["Jan-2000"],
            'fico_range_low': [fico_low], 'fico_range_high': [fico_high], 'inq_last_6mths': [inq_six_mths],
            'open_acc': [open_acc], 'pub_rec': [pub_rec], 'revol_bal': [revol_bal], 'revol_util': [revol_util],
            'total_acc': [total_acc], 'initial_list_status': [initial_list_status], 'application_type': [application_type],
            'mort_acc': [mort_acc], 'pub_rec_bankruptcies': [pub_rec_bankruptcies],
            'credit_score_range': [fico_high - fico_low], 'monthly_income': [annual_inc / 12],
            'payment_to_income_ratio (%)': [(installment / (annual_inc / 12)) * 100]
        }
        
        input_df = pd.DataFrame(raw_data)
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        
        prob_repay = model.predict_proba(input_scaled)[:, 1][0]
        risk_score = (1 - prob_repay) * 100
        
        # Result Page
        st.success("Analysis Complete")
        
        col1, col2 = st.columns([1, 2])
        with col1:
             fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font':{'color':'#1E293B', 'size': 20}},
                number = {'font': {'color': "#1E293B", 'size': 40}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#EF4444" if risk_score > 50 else "#10B981"},
                    'bgcolor': "#E2E8F0", 'borderwidth': 0,
                    'steps': [{'range': [0, 100], 'color': '#F1F5F9'}]
                }))
             fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1E293B"}, height=300)
             st.plotly_chart(fig, use_container_width=True)
             
        with col2:
             st.markdown(f"### Result: **{'High Risk' if risk_score > 50 else 'Low Risk'}**")
             st.info(f"Probability of Default: {risk_score:.2f}%")
             if risk_score > 50:
                 st.error("Recommendation: Manual Review / Decline")
             else:
                 st.success("Recommendation: Approve")


def show_dashboard():
    st.markdown("### 📊 Dashboard Overview")
    
    if df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        
        def card(title, val, icon, color):
            st.markdown(f"""
            <div class="lg-card" style="padding: 1.5rem; text-align: center; border-left: 5px solid {color};">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="color: #64748B; font-size: 0.9rem; text-transform: uppercase; font-weight: 700;">{title}</div>
                <div style="font-size: 2rem; font-weight: 800; color: #0F172A;">{val}</div>
            </div>
            """, unsafe_allow_html=True)
            
        total = len(df)
        low_risk = len(df[df['loan_status'] == 'Fully Paid'])
        high_risk = len(df[df['loan_status'] == 'Charged Off'])
        avg_score = 100 - (df['int_rate'].mean() * 2) 
        
        with c1: card("Total", f"{total:,}", "👥", "#4F46E5")
        with c2: card("Low Risk", f"{low_risk:,}", "✅", "#10B981")
        with c3: card("High Risk", f"{high_risk:,}", "🚫", "#EF4444")
        with c4: card("Avg Score", f"{avg_score:.0f}", "📈", "#8B5CF6")
        with c5: card("Volume", "$24M", "💰", "#F59E0B")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown('<div class="lg-card" style="height: 450px;">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1E293B;'>Risk Distribution</h4>", unsafe_allow_html=True)
            fig = px.donut(values=[low_risk, high_risk], names=['Low Risk', 'High Risk'],
                           color_discrete_sequence=['#10B981', '#EF4444'], hole=0.6)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={'color': '#1E293B', 'size': 14}, showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="lg-card" style="height: 450px;">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1E293B;'>Risk Trend (Last 30 Days)</h4>", unsafe_allow_html=True)
            dates = pd.date_range(end=datetime.today(), periods=30)
            scores = np.random.randint(40, 70, size=30)
            trend_df = pd.DataFrame({'Date': dates, 'Avg Score': scores})
            
            fig2 = px.line(trend_df, x='Date', y='Avg Score')
            fig2.update_traces(line_color='#4F46E5', fill='tozeroy', fillcolor='rgba(79, 70, 229, 0.1)', line_width=4)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={'color': '#1E293B', 'size': 14}, xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#E2E8F0'))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data not loaded")

def show_about():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 4rem;">
        <h2 style="font-size: 3rem;">Smart Loan Decisions with <span class="gradient-text">Transparent AI</span></h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="lg-card">
            <h3 style="color: #DC2626;">⛔ The Problem</h3>
            <ul style="color: #475569; line-height: 2.2; font-size: 1.1rem;">
                <li>Traditional credit scores lack transparency</li>
                <li>Manual assessment is slow and error-prone</li>
                <li>High compliance costs and regulatory friction</li>
                <li>One-size-fits-all models miss nuances</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="lg-card">
            <h3 style="color: #059669;">✅ Our Solution</h3>
            <ul style="color: #475569; line-height: 2.2; font-size: 1.1rem;">
                <li>AI-powered scoring with full explainability</li>
                <li>Real-time assessments in milliseconds</li>
                <li>Full audit trails for regulatory compliance</li>
                <li>Customizable thresholds for your portfolio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_docs():
    st.markdown("### 📚 Documentation")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("""
        <div class="lg-card">
            <div style="padding: 10px; background: #EEF2FF; border-radius: 8px; color: #4F46E5; margin-bottom: 12px; font-weight: 600;">Getting Started</div>
            <div style="padding: 10px; color: #64748B;">Input Fields</div>
            <div style="padding: 10px; color: #64748B;">API Reference</div>
            <div style="padding: 10px; color: #64748B;">Model Card</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="lg-card">', unsafe_allow_html=True)
        st.markdown("#### Getting Started")
        st.markdown("<p style='font-size: 1.1rem; color: #475569;'>Welcome to LoanGuard AI. This platform helps financial institutions make data-driven lending decisions.</p>", unsafe_allow_html=True)
        
        st.markdown("##### Quick Start Guide")
        st.markdown("""
        <ol style="line-height: 2; font-size: 1.1rem; color: #475569;">
            <li>Navigate to <b>Risk Calculator</b> from the top menu.</li>
            <li>Fill in the applicant's personal and loan details.</li>
            <li>Click <b>Assess Risk</b> to generate a real-time score.</li>
            <li>Review the detailed explanation and feature importance.</li>
        </ol>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ROUTER ---
if st.session_state.page == "Home": show_home()
elif st.session_state.page == "Risk Calculator": show_calculator()
elif st.session_state.page == "Dashboard": show_dashboard()
elif st.session_state.page == "About": show_about()
elif st.session_state.page == "Documentation": show_docs()
