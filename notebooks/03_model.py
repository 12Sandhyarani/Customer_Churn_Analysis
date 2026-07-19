import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, RocCurveDisplay)
import matplotlib.pyplot as plt

df = pd.read_csv("../data/cleaned_churn.csv")
df = df.drop(columns=['tenure_bucket'], errors='ignore')

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                              eval_metric='logloss', random_state=42)
}

results = {}
fig, ax = plt.subplots(figsize=(7,6))

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'auc': auc,
        'precision_churn': report['1']['precision'],
        'recall_churn': report['1']['recall'],
        'f1_churn': report['1']['f1-score']
    }
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)

ax.plot([0,1],[0,1], linestyle='--', color='gray')
ax.set_title('ROC Curve Comparison')
plt.tight_layout()
plt.savefig('../images/roc_comparison.png', bbox_inches='tight')
print("Saved roc_comparison.png\n")

results_df = pd.DataFrame(results).T.round(3)
print(results_df)

# Feature importance from best model (Random Forest / XGBoost)
best_model = models["Random Forest"]
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
importances.sort_values().plot(kind='barh', color='#2E86AB')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('../images/feature_importance.png', bbox_inches='tight')
print("\nSaved feature_importance.png")
print("\nTop features:\n", importances)

results_df.to_csv("../outputs/model_results.csv")
