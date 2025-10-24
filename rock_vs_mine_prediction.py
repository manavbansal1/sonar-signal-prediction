""" Rock vs Mine Prediction Model """

"""Importing the Dependencies"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

"""Data Collection and Data Processing"""

try:
    # loading the dataset to a pandas DataFrame
    sonar_data = pd.read_csv('sonar_data.csv', header=None)  # No header in the CSV file
    print("Dataset loaded successfully.")
    print(f"Dataset Shape: {sonar_data.shape}")
except FileNotFoundError:
    print("Error: The dataset file was not found.")
    exit()


# Checking the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(sonar_data.head())

# The number of missing values in the dataset
print(f"\nMissing values: {sonar_data.isnull().sum().sum()}") 

# statistical measures about the data
print("\nStatistical summary of the dataset:")
print(sonar_data.describe())

# checking the distribution of Target Variable
print("\nDistribution of Target Variable:")
#print(sonar_data[60].value_counts())    # Mines -> 111  Rocks -> 97
print(f"M (Mine): {(sonar_data[60] == 'M').sum()}")
print(f"R (Rock): {(sonar_data[60] == 'R').sum()} \n")

# Grouping the data based on Target Variable
print("Mean values of features grouped by Target Variable:")
print(sonar_data.groupby(60).mean())

# Seperating data and labels
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

# print(X)
# print(Y)

"""Splitting the data into Training data and Test Data"""
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
    )
print(f"\nOriginal dataset: {X.shape}")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Standardizing the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Feature scaling completed!")

"""Model Training --> Logistic Regression"""

model = LogisticRegression(max_iter=1000)

# Training the Logistic Regression model with training data
model.fit(X_train_scaled, Y_train)
print("\n✓ Model training completed!")

"""Model Evaluation"""

# Accuracy on training data
X_train_prediction = model.predict(X_train_scaled)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print(f'\n{"="*50}')
print("TRAINING DATA EVALUATION")
print(f'{"="*50}')
print(f'Accuracy on training data: {training_data_accuracy:.4f} ({training_data_accuracy*100:.2f}%)')

# Accuracy on test data
X_test_prediction = model.predict(X_test_scaled)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print(f'\n{"="*50}')
print("TEST DATA EVALUATION")
print(f'{"="*50}')
print(f'Accuracy on test data: {test_data_accuracy:.4f} ({test_data_accuracy*100:.2f}%)')

# Detailed classification report
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction, target_names=['M (Mine)', 'R (Rock)']))

# Reliable accuracy checking through cross-validation
# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=5)
print(f'\n{"="*50}')
print("CROSS-VALIDATION RESULTS")
print(f'{"="*50}')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')


"""Saving the trained model to a file using pickle for future use"""
# Save the trained model
with open('sonar_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\n✓ Model and scaler saved to successfully.")

"""Making a Predictive System"""

'''
    Predcts whether the object is a Rock or a Mine
    input data: tuple of 60 sonar feature values

    returns: "Rock" or "Mine"
    and the confidence of prediction 
'''
def predict_sonar(input_data, model, scaler):
    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the numpy array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardizing the input data
    input_data_scaled = scaler.transform(input_data_reshaped)

    # prediction
    prediction = model.predict(input_data_scaled)
    prediction_probability = model.predict_proba(input_data_scaled)

    return prediction[0], prediction_probability[0]

"""
    Interactive system for making predictions
"""
def interactive_prediction():
    
    print(f'\n{"="*50}')
    print("INTERACTIVE PREDICTION SYSTEM")
    print(f'{"="*50}')
    print("\nOptions:")
    print("1. Use sample data")
    print("2. Enter custom data (60 values)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        # Use the predefined sample data
        pred, probs = predict_sonar(input_data, model, scaler)
        print(f"\nPrediction: {'Rock' if pred == 'R' else 'Mine'}")
        print(f"Confidence: {max(probs)*100:.2f}%")
    
    elif choice == '2':
        print("\nEnter 60 feature values separated by commas:")
        try:
            user_input = input()
            features = tuple(map(float, user_input.split(',')))
            
            if len(features) != 60:
                print(f"Error: Expected 60 values, got {len(features)}")
            else:
                pred, probs = predict_sonar(features, model, scaler)
                print(f"\nPrediction: {'Rock' if pred == 'R' else 'Mine'}")
                print(f"Confidence: {max(probs)*100:.2f}%")
        except ValueError:
            print("Error: Invalid input format!")
    
    elif choice == '3':
        print("\nExiting...")
    else:
        print("\nInvalid choice!") 


# input data for a mine
input_data = (0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115)

# example usage of the predictive system
print(f'\n{"="*50}')
print("PREDICTION EXAMPLE")
print(f'{"="*50}')

prediction, probabilities = predict_sonar(input_data, model, scaler)

print(f"\nPrediction: {prediction}")
print(f"Confidence - Mine: {probabilities[0]*100:.2f}%")
print(f"Confidence - Rock: {probabilities[1]*100:.2f}%")


# Using the interactive prediction system
interactive_prediction()

print(f'\n{"="*50}')
print("ANALYSIS COMPLETE!")
print(f'{"="*50}')