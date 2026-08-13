🔍 Project: Customer Churn Prediction and Retention Analysis using Machine Learning

I built an end-to-end ML project to predict which telecom customers are likely to
churn — and more importantly, WHY they churn — so a business can act before losing
them, not after.

📊 What I did:
• Cleaned and explored a 6,000-customer dataset (28% churn rate)
• Found month-to-month customers churn at ~4x the rate of two-year contract customers
• Engineered features and handled class imbalance
• Trained and compared 3 models (Logistic Regression, Random Forest, Gradient Boosting)
• Selected the best model based on ROC-AUC and recall — not just accuracy, since
  accuracy alone is misleading on imbalanced churn data
• Built a Flask web app so anyone can input customer details and get a live churn
  risk prediction

📈 Result: Logistic Regression model with 0.76 ROC-AUC, correctly flagging ~69% of
customers who actually churned.

💡 Key business insight: Contract type is the single strongest churn driver — nudging
month-to-month customers toward annual contracts could meaningfully cut churn.

🛠️ Tech stack: Python, Pandas, Scikit-learn, Matplotlib/Seaborn, Flask

This was a self-driven project as part of my Data Science internship prep — full
code, EDA visuals, and write-up on GitHub: [add your repo link here]

#DataScience #MachineLearning #Python #ChurnPrediction #BTech #InternshipProject
