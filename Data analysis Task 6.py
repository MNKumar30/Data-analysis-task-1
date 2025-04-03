#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

# ============================
# Task 1: Time Series Analysis (Sales Forecasting)
# ============================

# Generate Sample Sales Dataset
np.random.seed(42)
date_rng = pd.date_range(start='1/1/2020', periods=100, freq='D')
sales_data = pd.DataFrame({'Date': date_rng, 'Sales': np.random.randint(50, 200, size=(100,))})
sales_data.to_csv("sales_data.csv", index=False)  # Save dataset

# Load sales dataset
df_sales = pd.read_csv("sales_data.csv", parse_dates=["Date"], index_col="Date")
df_sales = df_sales.asfreq('D')  # Ensure consistent date frequency

# Visualize Sales Trends
plt.figure(figsize=(10,5))
plt.plot(df_sales, marker='o', linestyle='-', label="Sales Data")
plt.title("Sales Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Train ARIMA model
train_size = int(len(df_sales) * 0.8)
train_sales, test_sales = df_sales.iloc[:train_size], df_sales.iloc[train_size:]

arima_model = ARIMA(train_sales, order=(5,1,0))
arima_fit = arima_model.fit()

# Forecast sales
forecast_sales = arima_fit.forecast(steps=len(test_sales))

# Evaluate Model
rmse_sales = np.sqrt(mean_squared_error(test_sales, forecast_sales))
print(f"Sales Forecasting RMSE: {rmse_sales:.2f}")

# Plot Forecast
plt.figure(figsize=(10,5))
plt.plot(train_sales.index, train_sales, label="Training Data")
plt.plot(test_sales.index, test_sales, label="Actual Sales", color="green")
plt.plot(test_sales.index, forecast_sales, label="Forecast", color="red", linestyle="dashed")
plt.legend()
plt.title(f"Sales Forecasting (RMSE: {rmse_sales:.2f})")
plt.show()

print("Sales Forecasting Completed!\n")

# ============================
# Task 2: Heart Disease Prediction (Logistic Regression)
# ============================

# Generate Sample Heart Disease Dataset
np.random.seed(42)
heart_data = pd.DataFrame({
    'Age': np.random.randint(30, 80, size=200),
    'Gender': np.random.choice(['Male', 'Female'], size=200),
    'Cholesterol': np.random.randint(150, 300, size=200),
    'Blood Pressure': np.random.randint(90, 180, size=200),
    'Heart Disease': np.random.choice([0, 1], size=200)  # 0 = No, 1 = Yes
})
heart_data.to_csv("heart_disease.csv", index=False)  # Save dataset

# Load heart disease dataset
df_heart = pd.read_csv("heart_disease.csv")

# Convert categorical variable 'Gender' to numeric (Male=1, Female=0)
df_heart['Gender'] = df_heart['Gender'].map({'Male': 1, 'Female': 0})

# Check for missing values
print("Missing Values:\n", df_heart.isnull().sum())

# Split data into features and target variable
X = df_heart.drop(columns=['Heart Disease'])
y = df_heart['Heart Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
heart_model = LogisticRegression()
heart_model.fit(X_train, y_train)

# Predict on test data
y_pred = heart_model.predict(X_test)

# Evaluate model performance
accuracy_heart = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Heart Disease Prediction Accuracy: {accuracy_heart:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Heart Disease Prediction")
plt.show()

print("Heart Disease Prediction Completed!")


# In[ ]:




