# Load libraries
from csv import reader 
import scipy
import sys

import matplotlib 
import pandas
import sklearn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import csv as csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_path = ".\\train.json"
test_path = ".\\test.json"

# Read in data
train = pd.read_json(train_path)
test = pd.read_json(test_path)

# Separate training and testing data
recipe_tr = [" ,".join(item) for item in train['ingredients']]
recipe_ts = [" ,".join(item) for item in test['ingredients']]

train['recipe'] = recipe_tr
test['recipe'] = recipe_ts

x_train, x_test, y_train, y_test = model_selection.train_test_split(train['recipe'], train['cuisine'],test_size=0.3)

labeler = LabelEncoder()

# Encode data (like get_dummies) so it can be processed
num_recipe_tr = labeler.fit_transform(y_train)
num_recipe_ts = labeler.fit_transform(y_test)
tfidf = TfidfVectorizer()   # Maps text to numerical value that represents how frequently the text appears in data - like mapping Category values
tfidf.fit(recipe_tr)

# Train using logistic regression model
np.random.seed(42)
matrixtfidf = tfidf.transform(x_train)
test_sheet = tfidf.transform(x_test)

LR = LogisticRegression(C=8.5, solver = 'liblinear')
LR.fit(matrixtfidf, num_recipe_tr)    # Model is learning the relationship between the descriptors and the cuisine int values - training
predictions = LR.predict(test_sheet)    # Use info from the training (line above) to predict cuisines for x_test - still part of training
print (accuracy_score(num_recipe_ts, predictions))

# Test
final_tfidf = tfidf.transform(recipe_ts)
test_results = LR.predict(final_tfidf)    # Test for real! Predict cuisines for test.json
cuisine = labeler.inverse_transform(test_results)

# Export
submission = pd.DataFrame({'id': test['id'], 'cuisine': cuisine}, columns=['id', 'cuisine'])
submission.head()
submission.to_csv("submission.csv",index=False)
