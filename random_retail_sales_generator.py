import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
data = {
    'Date': np.random.choice(date_rng, size=100),
    'Product_ID': np.random.randint(1, 5, size=100),
    'Quantity': np.random.randint(1, 10, size=100),
    'Revenue': np.random.uniform(0, 100, size=100)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('retail_sales_data.csv', index=False)