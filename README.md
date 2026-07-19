# Customer Churn Analysis & Prediction

Predicting customer churn for a telecom provider using EDA and machine learning, with actionable retention recommendations.

## Business Problem
Customer acquisition costs 5-25x more than retention. This project identifies which customers
are at risk of churning and what factors drive that risk, so the business can act before losing them.

## Dataset
- Source: Telco Customer Churn dataset (7,043 customers, 21 features)
- Fields: demographics, account tenure, contract/billing details, subscribed services, churn label

## Approach
1. **Data Cleaning** — fixed `TotalCharges` type issue, handled 11 missing values (new customers, tenure=0)
2. **Exploratory Analysis** — churn rate by contract type, tenure, internet service, payment method
3. **Modeling** — Logistic Regression, Random Forest, and XGBoost, evaluated on AUC/precision/recall (class-imbalance aware)
4. **Feature Importance** — identified the strongest predictors of churn

## Key Findings
- Overall churn rate: **26.5%**
- **Contract type is the #1 driver**: month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
- **Fiber optic customers churn at ~42%**, more than double DSL customers — signals a pricing or service quality issue
- **Electronic check payers churn at 45.3%**, far above autopay methods (~15-17%)
- Churned customers have an average tenure of 18 months vs 37.6 for retained customers — **risk is highest early in the customer lifecycle**

## Model Performance

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.840 | 0.505 | 0.797 | 0.618 |
| Random Forest | 0.841 | 0.541 | 0.762 | 0.633 |
| XGBoost | 0.819 | 0.550 | 0.679 | 0.608 |

Random Forest was selected for the best AUC/F1 balance. Logistic Regression is a strong
alternative when maximizing recall (catching more true churners) matters more than precision.

## Business Recommendations
1. Incentivize month-to-month customers to switch to annual contracts (e.g. discount for 1-year lock-in)
2. Investigate fiber optic service/pricing complaints — this segment churns at nearly 2x the average
3. Encourage autopay adoption among electronic check users to reduce friction-driven churn
4. Focus retention outreach on customers in their first 12-18 months — this is the highest-risk window

## Tech Stack
Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn

## Project Structure
```
churn_project/
├── data/               # raw and cleaned datasets
├── notebooks/          # analysis scripts (cleaning, EDA, modeling)
├── images/             # generated charts
├── outputs/            # model results
└── README.md
```
