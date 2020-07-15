# Used after getting the .csv file of test.txt

# Load libraries
from csv import reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Read in data
train_path = "./train.csv"
test_path = "./test.csv"

raw_train = pd.read_csv(train_path, header=0)
raw_test = pd.read_csv(test_path, header=0)

train = raw_train.drop(["sentence"], axis=1)
test = raw_test.drop(["sentence"], axis=1)

# Encode the data to get dummy values
enc = LabelEncoder()  # Encodes labels into categorical variables
combined_data = pd.concat(
    [train, test], axis=0
)  # Concatenate to encode string columns - can't have more than one encoder
combined_data["final_punct"] = enc.fit_transform(combined_data["final_punct"])
combined_data["final_word"] = combined_data["final_word"].factorize()[0]

# Split combined_data into train and test sets again
dummy_train = combined_data.dropna(subset=["end_of_par"])
dummy_test = combined_data[combined_data.end_of_par.isnull()]

###########################################################################

# Train
X = dummy_train.drop("end_of_par", axis=1)
Y = dummy_train["end_of_par"]
x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.20)

# SVM
svclassifier = LinearSVC(random_state=0, tol=1e-5, dual=False)
svclassifier.fit(x_train, x_test)
y_pred = svclassifier.predict(y_train)
svc_accuracy = accuracy_score(y_test, y_pred)

# Decision Tree
dtclassifier = DecisionTreeClassifier()
dtclassifier.fit(x_train, x_test)
y_pred = dtclassifier.predict(y_train)
dtree_accuracy = accuracy_score(y_test, y_pred)

# Logistic Regression
lrclassifier = LogisticRegression(C=8.5, solver="liblinear")
lrclassifier.fit(x_train, x_test)
y_pred = lrclassifier.predict(y_train)
lr_accuracy = accuracy_score(y_test, y_pred)

###########################################################################

# Test
print("SVM Accuracy Score: " + str(svc_accuracy))
print("Decision Tree Accuracy Score: " + str(dtree_accuracy))
print("Logistic Regression Accuracy Score: " + str(lr_accuracy))
models_dict = {"svm": svc_accuracy, "dtree": dtree_accuracy, "lr": lr_accuracy}
best_model = max(models_dict, key=models_dict.get)

if best_model == "svm":
    print("Using SVM model")
    test_results = svclassifier.predict(dummy_test.drop("end_of_par", axis=1))
elif best_model == "dtree":
    print("Using Decision Tree model")
    test_results = dtclassifier.predict(dummy_test.drop("end_of_par", axis=1))
else:
    print("Using Logistic Regression model")
    test_results = lrclassifier.predict(dummy_test.drop("end_of_par", axis=1))

raw_test["end_of_par"] = test_results.astype(int)

# Export
raw_test.head()
raw_test.to_csv("results.csv", index=False)

my_text = ""
for i, r in raw_test.iterrows():
    curr_sent = r["sentence"]
    my_text += curr_sent
    if raw_test.at[i, "end_of_par"] == 1:
        my_text += "\n\n"

txt_file = open("results.txt", "w", encoding="utf8")
txt_file.write(my_text)
txt_file.close()

