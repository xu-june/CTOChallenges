# Load libraries
from csv import reader 
import scipy
import sys

import matplotlib 
import pandas
import sklearn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import csv as csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_path = ".\\train.json"
test_path = ".\\test.json"

# Read in data
train = pd.read_json(train_path)
test = pd.read_json(test_path)

recipe = [" ,".join(item) for item in train['ingredients']]
culture = [" ,".join(item) for item in test['ingredients']]

train['recipe'] = recipe
train['ingredients'] = recipe

test['recipe'] = culture

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train['recipe'],train['cuisine'],test_size=0.3)

labeler = LabelEncoder()

num_recipe = labeler.fit_transform(Train_Y)
num_culture = labeler.fit_transform(Test_Y)
tfidf = TfidfVectorizer()
tfidf.fit(recipe)

# Train
np.random.seed(42)
matrixtfidf = tfidf.transform(Train_X)
test_sheet = tfidf.transform(Test_X)
SVM = SVC(C=3, kernel="linear")
SVM.fit(matrixtfidf, num_recipe)
prediction = SVM.predict(test_sheet)
print("Accuracy: ",accuracy_score(prediction, num_culture)*100)

# Test
label = labeler.fit_transform(train['cuisine'])
cook = tfidf.transform(train['recipe'])
final_svm = SVC(C=2, kernel='linear')
final_svm.fit(cook, label)

final_tfidf = tfidf.transform(culture)
solution = final_svm.predict(final_tfidf)
cuisine = labeler.inverse_transform(solution)

# Export
submission = pd.DataFrame({'id': test['id'], 'cuisine': cuisine}, columns=['id', 'cuisine'])
submission.head()
submission.to_csv("submission.csv",index=False)
