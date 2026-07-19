import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv("../data/cleaned_churn.csv")

# 1. Churn by Contract type
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

contract_churn = df.groupby('Contract')['Churn'].mean().sort_values() * 100
contract_churn.plot(kind='barh', ax=axes[0], color='#2E86AB')
axes[0].set_title('Churn Rate by Contract Type')
axes[0].set_xlabel('Churn Rate (%)')

# 2. Churn by tenure buckets
df['tenure_bucket'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72],
                              labels=['0-12','13-24','25-36','37-48','49-60','61-72'])
tenure_churn = df.groupby('tenure_bucket', observed=True)['Churn'].mean() * 100
tenure_churn.plot(kind='bar', ax=axes[1], color='#A23B72')
axes[1].set_title('Churn Rate by Tenure (months)')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].tick_params(axis='x', rotation=45)

# 3. Churn by Internet Service
internet_churn = df.groupby('InternetService')['Churn'].mean().sort_values() * 100
internet_churn.plot(kind='barh', ax=axes[2], color='#F18F01')
axes[2].set_title('Churn Rate by Internet Service')
axes[2].set_xlabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('../images/churn_drivers_1.png', bbox_inches='tight')
print("Saved churn_drivers_1.png")

# 4. Monthly charges distribution by churn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[0], palette=['#2E86AB','#C73E1D'])
axes[0].set_xticklabels(['Stayed', 'Churned'])
axes[0].set_title('Monthly Charges: Stayed vs Churned')

payment_churn = df.groupby('PaymentMethod')['Churn'].mean().sort_values() * 100
payment_churn.plot(kind='barh', ax=axes[1], color='#3B1F2B')
axes[1].set_title('Churn Rate by Payment Method')
axes[1].set_xlabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('../images/churn_drivers_2.png', bbox_inches='tight')
print("Saved churn_drivers_2.png")

# Print summary stats for the README/insights
print("\n--- KEY INSIGHTS ---")
print("\nChurn by Contract:\n", contract_churn.round(1))
print("\nChurn by Internet Service:\n", internet_churn.round(1))
print("\nChurn by Payment Method:\n", payment_churn.round(1))
print("\nAvg tenure - churned:", df[df['Churn']==1]['tenure'].mean().round(1))
print("Avg tenure - stayed:", df[df['Churn']==0]['tenure'].mean().round(1))
