"""
Customer Churn Prediction — Flask Web App
Run: python app.py, then open http://127.0.0.1:5000
"""
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")
model = joblib.load(MODEL_PATH)

FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "SupportTickets6mo"
]

OPTIONS = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    if request.method == "POST":
        data = {}
        for field in FIELDS:
            val = request.form.get(field)
            if field in ["tenure", "MonthlyCharges", "TotalCharges", "SupportTickets6mo"]:
                val = float(val)
            data[field] = val
        row = pd.DataFrame([data])
        proba = model.predict_proba(row)[0, 1]
        prediction = "High Risk of Churn" if proba >= 0.5 else "Likely to Stay"
        probability = round(proba * 100, 1)

    return render_template("index.html", fields=FIELDS, options=OPTIONS,
                            prediction=prediction, probability=probability)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
