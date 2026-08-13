"""
Customer Churn Prediction and Retention Analysis using Machine Learning
-------------------------------------------------------------------------
End-to-end pipeline: load -> clean -> EDA -> feature engineering ->
handle class imbalance -> train/compare models -> evaluate -> explain ->
save the best model for deployment.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

VIS_DIR = "/home/claude/churn_project/visuals"
MODEL_DIR = "/home/claude/churn_project/models"

# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/churn_project/data/telecom_churn.csv")

# TotalCharges has some missing values (new customers, tenure=0) -> fill with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("Churn rate: {:.1%}".format(df["ChurnFlag"].mean()))

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Churn distribution
sns.countplot(data=df, x="Churn", palette=["#2E7D32", "#C62828"], ax=axes[0, 0])
axes[0, 0].set_title("Overall Churn Distribution")

# Churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn", palette=["#2E7D32", "#C62828"], ax=axes[0, 1])
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].tick_params(axis='x', rotation=15)

# Tenure distribution by churn
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, palette=["#2E7D32", "#C62828"], ax=axes[0, 2])
axes[0, 2].set_title("Tenure Distribution by Churn")

# Monthly charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#2E7D32", "#C62828"], ax=axes[1, 0])
axes[1, 0].set_title("Monthly Charges vs Churn")

# Churn by internet service
sns.countplot(data=df, x="InternetService", hue="Churn", palette=["#2E7D32", "#C62828"], ax=axes[1, 1])
axes[1, 1].set_title("Churn by Internet Service")

# Churn by payment method
sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette=["#2E7D32", "#C62828"], ax=axes[1, 2])
axes[1, 2].set_title("Churn by Payment Method")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/eda_overview.png", bbox_inches="tight")
plt.close()

# Correlation heatmap for numeric features
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SupportTickets6mo", "ChurnFlag"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# key business insight numbers
insights = {}
insights["churn_rate_overall"] = round(df["ChurnFlag"].mean(), 3)
insights["churn_rate_month_to_month"] = round(
    df[df.Contract == "Month-to-month"]["ChurnFlag"].mean(), 3)
insights["churn_rate_two_year"] = round(
    df[df.Contract == "Two year"]["ChurnFlag"].mean(), 3)
insights["churn_rate_fiber"] = round(
    df[df.InternetService == "Fiber optic"]["ChurnFlag"].mean(), 3)
insights["churn_rate_electronic_check"] = round(
    df[df.PaymentMethod == "Electronic check"]["ChurnFlag"].mean(), 3)
insights["avg_tenure_churned"] = round(df[df.Churn == "Yes"]["tenure"].mean(), 1)
insights["avg_tenure_retained"] = round(df[df.Churn == "No"]["tenure"].mean(), 1)

print(json.dumps(insights, indent=2))

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING + TRAIN/TEST SPLIT
# ---------------------------------------------------------------
target = "ChurnFlag"
drop_cols = ["customerID", "Churn", "ChurnFlag"]
X = df.drop(columns=drop_cols)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols_feat = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols_feat),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

# ---------------------------------------------------------------
# 4. MODEL TRAINING (class_weight='balanced' handles imbalance
#    since imblearn/SMOTE isn't available in this environment)
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    results[name] = {
        "roc_auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn": round(report["1"]["recall"], 4),
    }
    fitted_pipelines[name] = pipe
    print(f"\n{name}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("ROC-AUC:", round(auc, 4))

print("\n=== MODEL COMPARISON ===")
for name, r in results.items():
    print(name, r)

# pick best model by ROC-AUC
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best_pipe = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 5. EVALUATION VISUALS FOR BEST MODEL
# ---------------------------------------------------------------
proba_best = best_pipe.predict_proba(X_test)[:, 1]
preds_best = best_pipe.predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Confusion matrix
cm = confusion_matrix(y_test, preds_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix — {best_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba_best)
axes[1].plot(fpr, tpr, label=f"AUC = {results[best_name]['roc_auc']:.3f}", color="#1565C0", linewidth=2)
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Model comparison bar chart
comp_df = pd.DataFrame(results).T
comp_df[["roc_auc", "f1_score"]].plot(kind="bar", ax=axes[2], color=["#1565C0", "#EF6C00"])
axes[2].set_title("Model Comparison")
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/model_evaluation.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 6. FEATURE IMPORTANCE (best tree-based model, fallback to RF)
# ---------------------------------------------------------------
importance_source = best_pipe if hasattr(best_pipe.named_steps["clf"], "feature_importances_") else fitted_pipelines["Random Forest"]
feat_names = importance_source.named_steps["prep"].get_feature_names_out()
importances = importance_source.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
plt.title("Top 15 Feature Importances (Churn Drivers)")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/feature_importance.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 7. SAVE ARTIFACTS
# ---------------------------------------------------------------
joblib.dump(best_pipe, f"{MODEL_DIR}/churn_model.pkl")

summary = {
    "best_model": best_name,
    "results": results,
    "insights": insights,
    "top_features": imp_df.to_dict(orient="records"),
}
with open(f"{MODEL_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved model + summary. Done.")
