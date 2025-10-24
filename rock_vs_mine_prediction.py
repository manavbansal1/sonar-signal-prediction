""" Rock vs Mine Prediction Model """

"""Importing the Dependencies"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing"""

# loading the dataset to a pandas DataFrame
sonar_data = pd.read_csv('sonar_data.csv', header=None)  # No header in the CSV file

# Checking the first 5 rows of the dataset
# print(sonar_data.head())

# number of rows and columns in the dataset
#print(sonar_data.shape)  

# statistical measures about the data
#print(sonar_data.describe())

# checking the distribution of Target Variable
#print(sonar_data[60].value_counts())    # Mines -> 111  Rocks -> 97

# Grouping the data based on Target Variable
#print(sonar_data.groupby(60).mean())

# Seperating data and labels
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

# print(X)
# print(Y)

"""Splitting the data into Training data and Test Data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)

"""Model Training - Logistic Regression Model"""
model = LogisticRegression()

# Training the Logistic Regression Model
model.fit(X_train, Y_train)
print("Model Training Completed.")

"""Model Evaluation"""
# accuracy on training data
