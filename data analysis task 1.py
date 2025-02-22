#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import zipfile

# Corrected URL assignment
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
dataset_path = "student-mat.csv"

# Download and extract dataset
urllib.request.urlretrieve(url, "student.zip")
with zipfile.ZipFile("student.zip", "r") as zip_ref:
    zip_ref.extractall(".")

# Load dataset
data = pd.read_csv(dataset_path, sep=";")
print("Data loaded successfully!")

# Display first few rows
print(data.head())

# Dataset info
print("\nDataset info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove duplicate entries
data = data.drop_duplicates()

# Calculate and print average final grade
average_score = data['G3'].mean()
print(f"\nAverage Math Score (G3): {average_score:.2f}")

# Count students scoring above 15
student_above_15 = len(data[data['G3'] > 15])
print(f"Number of students scoring above 15: {student_above_15}")

# Calculate correlation between study time and final grade
correlation = data['studytime'].corr(data['G3'])
print(f"Correlation between study time and final grade: {correlation:.2f}")

# Calculate average final grade by gender
average_grade_by_gender = data.groupby('sex')['G3'].mean()
print("\nAverage Final Grade by Gender:")
print(average_grade_by_gender)

# Plot histogram of final grades
plt.figure(figsize=(8, 5))
plt.hist(data['G3'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Study time vs Final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='studytime', y='G3', hue='sex')
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time (hours)")
plt.ylabel("Final Grade")
plt.legend(title="Gender")
plt.show()

# Bar plot: Average final grade by gender
plt.figure(figsize=(8, 5))
average_grade_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title("Average Final Grade by Gender")
plt.ylabel("Average Final Grade")
plt.xlabel("Gender")
plt.xticks(rotation=0)
plt.show()


# In[ ]:





# In[ ]:




