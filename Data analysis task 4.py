#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('D:/New folder/house-prices.csv')

# Rename columns for consistency
data.rename(columns={'SqFt': 'Size', 'Bedrooms': 'Number of Rooms'}, inplace=True)

# Convert categorical variables into numerical
data = pd.get_dummies(data, columns=['Brick', 'Neighborhood'], drop_first=True)

# Check dataset info
print("Dataset Information:")
print(data.info())

# Handle missing values
data['Size'].fillna(data['Size'].median(), inplace=True)
data['Number of Rooms'].fillna(data['Number of Rooms'].median(), inplace=True)

# Handle outliers
upper_limit = data['Price'].quantile(0.95)
data['Price'] = np.where(data['Price'] > upper_limit, upper_limit, data['Price'])

# Feature scaling
scaler = MinMaxScaler()
data[['Size', 'Number of Rooms']] = scaler.fit_transform(data[['Size', 'Number of Rooms']])

# Define X and y
x = data.drop('Price', axis=1)
y = data['Price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(x_train, y_train)

# Model evaluation
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# Residual distribution
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()


# In[ ]:




