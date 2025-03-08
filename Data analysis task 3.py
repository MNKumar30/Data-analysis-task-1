#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(dataset_url)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Selecting features for clustering
features = df[['sepal_length', 'sepal_width', 'petal_length']]

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nFirst 5 rows of scaled features:")
print(scaled_features[:5])

# Elbow Method to find optimal k
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Fixed 'n_init' warning
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()

# Choosing k=3 based on the elbow method
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Fixed 'n_init' warning
cluster_labels = kmeans.fit_predict(scaled_features)

# Assigning cluster labels to dataset
df['Cluster'] = cluster_labels

print("\nFirst 5 rows with cluster labels:")
print(df.head())

# Scatter Plot with Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=cluster_labels, palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()
plt.show()


# In[ ]:




