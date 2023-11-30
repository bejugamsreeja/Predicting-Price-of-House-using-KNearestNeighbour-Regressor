#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Predicting Price of House using KNearestNeighbour Regressor
#Use KNearestNeighbourRegressor to Predict Price of House.
#Dataset Link:https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/house_rental_data.csv.txt
#Q1.Use pandas to get some insights into the data

import pandas as pd

url = "https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt"
df = pd.read_csv(url)

# Explore the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics of the dataset
print(df.describe())

# Count unique values in each column
print(df.nunique())

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)


# In[4]:


#Q2.Show some interesting visualization of the data


import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of the distribution of prices
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of Price vs. Sqft
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sqft', y='Price', data=df)
plt.title('Price vs. Sqft')
plt.xlabel('Sqft')
plt.ylabel('Price')
plt.show()


# In[7]:


#Q3.Manage data for training & testing


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X = df[['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Living.Room', 'Bathroom']]
y = df['Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


#Q4.Finding a better value of k


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


knn_regressor = KNeighborsRegressor()

# Define a range of K values to try
k_values = range(1, 21)

# Cross-validation to find the best K value
mae_scores = []
for k in k_values:
    knn_regressor.n_neighbors = k
    scores = cross_val_score(knn_regressor, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    mae_scores.append(-scores.mean())

# Plot the K values against the mean absolute error
plt.figure(figsize=(8, 6))
plt.plot(k_values, mae_scores, marker='o')
plt.title('KNN Regression: K vs. Mean Absolute Error')
plt.xlabel('K')
plt.ylabel('Mean Absolute Error')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Find the best K value
best_k = k_values[mae_scores.index(min(mae_scores))]
print(f"Best K value: {best_k}")


# In[ ]:




