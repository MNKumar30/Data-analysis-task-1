#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import zipfile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
dataset_path = "student-mat.csv"

# Downloading the dataset
urllib.request.urlretrieve(url, "student.zip")

# Extracting the dataset
with zipfile.ZipFile("student.zip", "r") as zip_ref:
    zip_ref.extractall(".")
    
# Loading the dataset
data = pd.read_csv("student-mat.csv", sep=";")
print("Data loaded successfully!")

# Displaying first few rows
print(data.head())

# Dataset Information
print("\nDataset Info:")
print(data.info())

# Checking for Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Dropping duplicates
data = data.drop_duplicates()

# Calculating average score
average_score = data['G3'].mean()
print(f"\nAverage Math Score (G3): {average_score:.2f}")

# Counting students scoring above 15
students_above_15 = len(data[data['G3'] > 15])
print(f"\nNumber of students scoring above 15: {students_above_15}")

# Correlation between study time and final grade
correlation = data['studytime'].corr(data['G3'])
print(f"Correlation between study time and final grade: {correlation:.2f}")

# Average grade by gender
average_grade_by_gender = data.groupby('sex')['G3'].mean()
print("\nAverage Final Grade by Gender:")
print(average_grade_by_gender)

# Plotting Histogram of Final Grades
plt.figure(figsize=(8, 5))
plt.hist(data['G3'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()

# Plotting Scatter Plot of Study Time vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='studytime', y='G3', hue='sex')
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time (hours)")
plt.ylabel("Final Grade")
plt.legend(title="Gender")
plt.show()

# Plotting Average Grade by Gender
plt.figure(figsize=(8, 5))
average_grade_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title("Average Final Grade by Gender")
plt.ylabel("Average Final Grade")
plt.xlabel("Gender")
plt.xticks(rotation=0)
plt.show()


# In[ ]:




