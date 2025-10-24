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
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

"""Making a Predictive System"""
# input data for a mine
input_data = (0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115)
# input_data for a mine

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')

    