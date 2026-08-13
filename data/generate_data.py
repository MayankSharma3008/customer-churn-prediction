"""
Generates a realistic synthetic telecom customer churn dataset.
Mirrors the structure/feature set of the well-known Telco Customer Churn
dataset, with churn probability driven by realistic business logic
(contract type, tenure, charges, support tickets, etc.) so that EDA and
modeling produce meaningful, explainable patterns.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

customer_id = [f"CUST-{10000+i}" for i in range(N)]
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

tenure = np.random.gamma(shape=2.0, scale=15, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)

def dep_service(base_p_yes=0.4):
    out = []
    for isv in internet_service:
        if isv == "No":
            out.append("No internet service")
        else:
            out.append(np.random.choice(["Yes", "No"], p=[base_p_yes, 1 - base_p_yes]))
    return np.array(out)

online_security = dep_service(0.35)
online_backup = dep_service(0.40)
device_protection = dep_service(0.40)
tech_support = dep_service(0.35)
streaming_tv = dep_service(0.45)
streaming_movies = dep_service(0.45)

# Monthly charges depend on services subscribed
base_charge = np.where(internet_service == "Fiber optic", 70,
              np.where(internet_service == "DSL", 45, 20))
addon_count = (
    (online_security == "Yes").astype(int) + (online_backup == "Yes").astype(int) +
    (device_protection == "Yes").astype(int) + (tech_support == "Yes").astype(int) +
    (streaming_tv == "Yes").astype(int) + (streaming_movies == "Yes").astype(int)
)
monthly_charges = base_charge + addon_count * 5.5 + np.random.normal(0, 5, N)
monthly_charges = np.clip(monthly_charges, 18, 120).round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).clip(min=0).round(2)

# Support tickets in last 6 months (proxy for satisfaction) - not in original dataset,
# added as a realistic engineered signal
support_tickets = np.random.poisson(lam=np.where(contract == "Month-to-month", 1.4, 0.5))

# ---- Churn probability model (business logic) ----
logit = (
    -3.0
    + 1.9 * (contract == "Month-to-month")
    + 0.5 * (contract == "One year")
    - 0.02 * tenure
    + 0.012 * monthly_charges
    + 0.9 * (internet_service == "Fiber optic")
    - 0.5 * (online_security == "Yes")
    - 0.5 * (tech_support == "Yes")
    + 0.25 * support_tickets
    + 0.35 * (payment_method == "Electronic check")
    + 0.2 * (paperless_billing == "Yes")
    - 0.3 * (partner == "Yes")
    - 0.2 * (dependents == "Yes")
    + 0.15 * senior_citizen
    + np.random.normal(0, 0.6, N)  # noise
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = np.where(np.random.rand(N) < prob_churn, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SupportTickets6mo": support_tickets,
    "Churn": churn,
})

# introduce a small amount of realistic missingness in TotalCharges (common in the real dataset)
missing_idx = df[df.tenure == 0].index
df.loc[missing_idx, "TotalCharges"] = np.nan

df.to_csv("/home/claude/churn_project/data/telecom_churn.csv", index=False)
print("Generated:", df.shape)
print(df["Churn"].value_counts(normalize=True))
