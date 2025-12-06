# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: decision_tree.py
# SPECIFICATION: This program builds and evaluates a decision tree classifier using given datasets.
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ["cheat_training_1.csv", "cheat_training_2.csv", "cheat_training_3.csv"]

for ds in dataSets:
    X = []
    Y = []

    df = pd.read_csv(
        ds, sep=",", header=0
    )  # reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[
        :, 1:
    ]  # creating a training matrix without the id (NumPy library)

    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    # X =
    for record in data_training:
        features = []

        # Refund
        features.append(1 if record[0] == "Yes" else 0)

        # Marital Status
        features.append(1 if record[1] == "Single" else 0)
        features.append(1 if record[1] == "Divorced" else 0)
        features.append(1 if record[1] == "Married" else 0)

        # Taxable Income
        features.append(float(record[2].replace("k", "")))

        X.append(features)

        # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        # --> add your Python code here
        # Y =
        Y.append(1 if record[3] == "Yes" else 0)

    # Store accuracy for each run
    accuracies = []

    # loop your training and test tasks 10 times here
    for i in range(10):
        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(
            clf,
            feature_names=["Refund", "Single", "Divorced", "Married", "Taxable Income"],
            class_names=["Yes", "No"],
            filled=True,
            rounded=True,
        )
        # plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        # data_test =
        df_test = pd.read_csv(
            "cheat_test.csv", sep=",", header=0
        )  # reading the test dataset eliminating the header
        data_test = np.array(df_test.values)[
            :, 1:
        ]  # creating a test matrix without the id

        correct_predictions = 0

        for data in data_test:
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            test_features = []

            # Refund
            test_features.append(1 if data[0] == "Yes" else 0)

            # Marital Status
            test_features.append(1 if data[1] == "Single" else 0)
            test_features.append(1 if data[1] == "Divorced" else 0)
            test_features.append(1 if data[1] == "Married" else 0)

            # Taxable Income
            test_features.append(float(data[2].replace("k", "")))

            # Use the decision tree to make the class prediction
            class_predicted = clf.predict([test_features])[0]

            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            # --> add your Python code here
            true_label = 1 if data[3] == "Yes" else 0
            if class_predicted == true_label:
                correct_predictions += 1

        # find the average accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        run_accuracy = correct_predictions / len(data_test)
        accuracies.append(run_accuracy)

    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    # --> add your Python code here
    final_avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"final accuracy when training on {ds}: {final_avg_accuracy}")
