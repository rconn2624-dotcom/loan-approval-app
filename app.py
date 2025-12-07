import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained model from pkl
with open("loan_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Loan Approval Predictor")

st.write("Enter applicant information and the model will estimate the probability of approval.")

# 2. Collect user inputs
fico = st.number_input("FICO Score", min_value=300, max_value=900, value=680)
income = st.number_input("Monthly Gross Income ($)", min_value=0, max_value=50000, value=5000)
housing = st.number_input("Monthly Housing Payment ($)", min_value=0, max_value=50000, value=1500)
requested = st.number_input("Requested Loan Amount ($)", min_value=0, max_value=2000000, value=20000)
granted = st.number_input("Granted Loan Amount ($)", min_value=0, max_value=2000000, value=20000)

employment_status = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])
employment_sector = st.selectbox(
    "Employment Sector",
    [
        "financials", "information_technology", "health_care", "industrials",
        "real_estate", "materials", "utilities", "energy",
        "consumer_staples", "communication_services",
        "consumer_discretionary", "Unknown"
    ],
)
reason = st.selectbox(
    "Loan Reason",
    [
        "credit_card_refinancing", "home_improvement", "major_purchase",
        "cover_an_unexpected_cost", "debt_consolidation", "other"
    ],
)
lender = st.selectbox("Lender", ["A", "B", "C"])
ever_bk = st.selectbox("Ever Bankrupt or Foreclose?", ["0", "1"])

# 3. Turn inputs into a DataFrame with same column names as training data
input_dict = {
    "FICO_score": fico,
    "Monthly_Gross_Income": income,
    "Monthly_Housing_Payment": housing,
    "Requested_Loan_Amount": requested,
    "Granted_Loan_Amount": granted,
    "Employment_Status": employment_status,
    "Employment_Sector": employment_sector,
    "Reason": reason,
    "Lender": lender,
    "Ever_Bankrupt_or_Foreclose": int(ever_bk),
}

input_df = pd.DataFrame([input_dict])

# 4. Predict when user clicks button
if st.button("Predict Approval"):
    # If your saved object is a pipeline (preprocessing + model), this will work directly:
    prob_approved = model.predict_proba(input_df)[0, 1]
    pred_class = model.predict(input_df)[0]

    st.write(f"**Predicted probability of approval:** {prob_approved:.2f}")
    if pred_class == 1:
        st.success("Model prediction: APPROVED")
    else:
        st.error("Model prediction: DENIED")
