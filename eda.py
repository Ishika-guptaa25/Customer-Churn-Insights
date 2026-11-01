
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/telco_customer_churn.csv")

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Basic dataset information
print("Dataset Overview:")
print(df.shape)

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 2. Tenure vs Churn
plt.figure(figsize=(7, 4))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="stack"
)
plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 3. Monthly Charges vs Churn
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# 4. Contract Type vs Churn
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="ChurnPercentage")
)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=contract_churn,
    x="Contract",
    y="ChurnPercentage"
)
plt.title("Contract Type vs Churn Percentage")
plt.xlabel("Contract Type")
plt.ylabel("Churn Percentage")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully!")
