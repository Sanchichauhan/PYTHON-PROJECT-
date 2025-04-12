import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Load the dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\online+shoppers+purchasing+intention+dataset (1)\online_shoppers_intention.csv")

# 1. Shape of the dataset
print(" Shape of the dataset:", df.shape)

# 2. Column names
print("\n Column names:")
print(df.columns.tolist())

# 3. Data types of each column
print("\n Data types:")
print(df.dtypes)

# 4. Check for missing values
print("\n Missing values per column:")
print(df.isnull().sum())

# 5. Show first 5 rows as sample
print("\n First 5 rows of the dataset:")
print(df.head())

# 6. Summary statistics for numeric columns
print("\n Summary statistics:")
print(df.describe())


# 7. Count of unique values per column
print("\n Unique value counts per column:")
print(df.nunique())

# 8. Count of each class in target variable (Revenue)
print("\n Target variable class distribution:")
print(df['Revenue'].value_counts())

sns.set(style="whitegrid")

sns.set(style="whitegrid")

sns.set(style="whitegrid")

# 🔥 Fig 0: Correlation Heatmap for Numeric Features
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include='number')  # Select only numeric columns
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Fig 0: Heatmap of Feature Correlations')
plt.tight_layout()
plt.show()

# 🔹 Fig 1: Countplot - Number of Sessions per Month by Revenue
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Month', hue='Revenue', palette='Set2', order=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Fig 1: Sessions per Month by Revenue')
plt.ylabel('Number of Sessions')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔹 Fig 2: Box Plot of PageValues by Revenue
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Revenue', y='PageValues', width=0.5, fliersize=3)
plt.title('Fig 2: PageValues by Revenue (with Means)')
plt.ylabel('Page Values')
plt.xlabel('Revenue')
plt.tight_layout()
plt.show()

# 🔹 Fig 3: Histogram of Administrative Duration
plt.figure(figsize=(10, 6))
sns.histplot(df['Administrative_Duration'], bins=30, kde=False, color='orange')
plt.title('Fig 3: Distribution of Administrative Duration')
plt.xlabel('Administrative Duration')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 🔹 Fig 4: Scatter Plot of PageValues vs ExitRates
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PageValues', y='ExitRates', hue='Revenue', palette='coolwarm', alpha=0.6)
plt.title('Fig 4: PageValues vs ExitRates')
plt.xlabel('Page Values')
plt.ylabel('Exit Rates')
plt.tight_layout()
plt.show()


# 🔷 Pie Chart: Revenue Distribution
revenue_counts = df['Revenue'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(revenue_counts, labels=revenue_counts.index, autopct='%1.1f%%', startangle=140,
        colors=['#66b3ff', '#ff9999'])
plt.title('Pie Chart: Revenue Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 🔷 Box Plot: ProductRelated_Duration vs Revenue
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Revenue', y='ProductRelated_Duration', hue='Revenue',
            palette='Set2', width=0.5, fliersize=3, dodge=False)
plt.title('Box Plot: ProductRelated_Duration vs Revenue')
plt.xlabel('Revenue')
plt.ylabel('Product Related Duration')
plt.tight_layout()
plt.show()
print("\n-------------------")
print("🔎 Hypothesis Testing")
print("-------------------")

# Separate BounceRates for Revenue True and False
revenue_true = df[df['Revenue'] == True]['BounceRates']
revenue_false = df[df['Revenue'] == False]['BounceRates']

# ------------------- Z-Test -------------------
# Assumes large sample size and known variance (approximate use case)
z_stat, z_pval = stats.ttest_ind(revenue_true, revenue_false)

print(f"\n📊 Z-Test (BounceRates for Revenue vs No Revenue)")
print(f"Z-Statistic: {z_stat:.2f}")
print(f"P-Value: {z_pval:.2f}")
if z_pval < 0.05:
    print("Result: Significant difference ✅")
else:
    print("Result: No significant difference ❌")

# ------------------- T-Test -------------------
# Standard t-test for independent samples
t_stat, t_pval = stats.ttest_ind(revenue_true, revenue_false, equal_var=False)

print(f"\n📊 T-Test (BounceRates for Revenue vs No Revenue)")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {t_pval:.2f}")
if t_pval < 0.05:
    print("Result: Significant difference ✅")
else:
    print("Result: No significant difference ❌")




