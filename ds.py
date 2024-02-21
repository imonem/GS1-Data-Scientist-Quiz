import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Import the data
df = pd.read_csv('retail_sales_data.csv')

# Step 2: Explore the dataset
print(df.info())
print(df.head())

# Step 3: Clean the data
# Handling missing values
df.dropna(inplace=True)

# Handling outliers (assuming Revenue cannot be negative)
df = df[df['Revenue'] >= 0]

# Step 4: Calculate total revenue per month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_revenue = df.groupby('Month')['Revenue'].sum()

# Step 5: Create visualizations
monthly_revenue_df = pd.DataFrame({'Month': monthly_revenue.index.astype(str), 'Revenue': monthly_revenue.values})

plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_revenue_df, x='Month', y='Revenue', marker='o')
plt.title('Monthly Revenue Trends')
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()