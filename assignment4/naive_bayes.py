# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: niave_bayes.py
# SPECIFICATION: Implementation of Naive Bayes classifier with hyperparameter tuning
# FOR: CS 4440- Assignment #4
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [
    0.1,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
]

# reading the training data
# --> add your Python code here
training_data = pd.read_csv("weather_training.csv")
X_train = training_data.drop(["Formatted Date", "Temperature (C)"], axis=1)
y_train = training_data["Temperature (C)"].values

# update the training class values according to the discretization (11 values only)
# --> add your Python code here
y_training = []
for temp in y_train:
    # map to closest value in classes
    closest_class = min(classes, key=lambda x: abs(x - temp))
    y_training.append(closest_class)
y_training = np.array(y_training)

# reading the test data
# --> add your Python code here
test_data = pd.read_csv("weather_test.csv")
X_test = test_data.drop(["Formatted Date", "Temperature (C)"], axis=1)
y_test = test_data["Temperature (C)"].values

# update the test class values according to the discretization (11 values only)
# --> add your Python code here
y_testing = []
for temp in y_test:
    # map to closest value in classes
    closest_class = min(classes, key=lambda x: abs(x - temp))
    y_testing.append(closest_class)
y_testing = np.array(y_testing)

# loop over the hyperparameter value (s)
# --> add your Python code here
highest_accuracy = 0
best_s = 0

for s in s_values:
    # fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_train, y_training)

    # make the naive_bayes prediction for each test sample and start computing its accuracy
    # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    # to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    # --> add your Python code here
    predictions = clf.predict(X_test)
    correct_predictions = 0
    for i in range(len(y_testing)):
        percent_difference = (
            100 * abs(predictions[i] - y_testing[i]) / abs(y_testing[i])
        )
        if percent_difference <= 15:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_testing)

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_s = s
        print(
            f"Highest Naive Bayes accuracy so far: {highest_accuracy}, Parameters: s={best_s}"
        )
