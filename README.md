# Customer Churn Prediction and Retention Analysis using Machine Learning

An end-to-end machine learning project that predicts which telecom customers are
likely to churn and surfaces the key drivers behind churn — built as a solo project
for a Data Science internship portfolio.

## Problem Statement
Customer acquisition costs far more than retention. This project builds a model that
flags at-risk customers *before* they leave, so a business can act (targeted offers,
support outreach) instead of reacting after the fact.

## Dataset
Telecom customer dataset (6,000 records, 22 features) covering demographics, account
information (contract type, tenure, billing), subscribed services, and charges.
Overall churn rate: **28.4%** — realistic and imbalanced, similar to real telecom data.

## Project Workflow
1. **Data Cleaning** — handled missing values in `TotalCharges`, type conversions
2. **Exploratory Data Analysis** — churn patterns by contract, tenure, charges, internet service, and payment method
3. **Feature Engineering** — one-hot encoding for categoricals, scaling for numerics, engineered support-ticket signal
4. **Handling Class Imbalance** — used `class_weight='balanced'` across models
5. **Model Training & Comparison** — Logistic Regression, Random Forest, Gradient Boosting
6. **Evaluation** — ROC-AUC, precision/recall/F1 (accuracy alone is misleading on imbalanced data)
7. **Explainability** — feature importance to identify top churn drivers
8. **Deployment** — Flask web app for live predictions

## Key Insights
| Segment | Churn Rate |
|---|---|
| Overall | 28.4% |
| Month-to-month contracts | 40.4% |
| Two-year contracts | 10.6% |
| Fiber optic internet | 39.0% |
| Electronic check payment | 31.5% |

Customers on **month-to-month contracts** churn at **~4x the rate** of two-year
contract customers — the single strongest retention lever a business has. Fiber optic
subscribers and electronic-check payers also churn above average, likely tied to price
sensitivity and payment friction.

## Model Performance
| Model | ROC-AUC | F1 (Churn) | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.760** | **0.562** | 0.474 | 0.691 |
| Random Forest | 0.758 | 0.538 | 0.494 | 0.591 |
| Gradient Boosting | 0.751 | 0.435 | 0.601 | 0.341 |

Logistic Regression was selected as the final model — it had the best ROC-AUC and,
importantly, the highest recall on churners. In a churn problem, missing an at-risk
customer (false negative) is costlier than a false alarm, so recall was prioritized
over raw accuracy.

## Top Churn Drivers (Feature Importance)
1. Two-year contract (strong negative driver — reduces churn)
2. Monthly charges
3. Total charges
4. Tenure
5. One-year contract
6. Support tickets (last 6 months)
7. Fiber optic internet service

## Tech Stack
- **Python**: pandas, numpy
- **ML**: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **Visualization**: matplotlib, seaborn
- **Deployment**: Flask
- **Model persistence**: joblib

## Project Structure
```
churn_project/
├── data/
│   └── telecom_churn.csv
├── notebooks/
│   └── churn_analysis.py       # full EDA + modeling pipeline
├── models/
│   ├── churn_model.pkl         # trained pipeline (preprocessing + model)
│   └── summary.json            # metrics + insights
├── visuals/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── model_evaluation.png
│   └── feature_importance.png
├── app/
│   ├── app.py                  # Flask app
│   └── templates/index.html
└── README.md
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Regenerate the dataset (or use the provided CSV)
python data/generate_data.py

# 3. Run the full analysis + train the model
python notebooks/churn_analysis.py

# 4. Launch the web app
cd app
python app.py
# open http://127.0.0.1:5000
```

## Business Recommendations
- Offer incentives for month-to-month customers to move to annual contracts
- Proactively reach out to customers with 2+ support tickets in the last 6 months
- Review fiber optic pricing/value proposition — churn is disproportionately high there
- Simplify or subsidize electronic check billing friction

## Future Improvements
- Add SHAP for per-customer explainability
- Try XGBoost / LightGBM and SMOTE-based resampling
- Build a CLV (Customer Lifetime Value) layer to prioritize retention spend
- Deploy to a live host (Render/Railway) with a public demo link

---
*Built as a self-driven internship project — BTech CSE (Data Science)*
