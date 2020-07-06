# Load libraries
from csv import reader 
import scipy
import sys

import matplotlib 
import pandas
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import csv as csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read in data
train_path = ".\\contest-train.csv"
test_path = ".\\contest-test.csv"

train_base = pd.read_csv(train_path, encoding='latin1', index_col=0, header=0)
test_base = pd.read_csv(test_path, encoding='latin1', index_col=0, header=0)      # bases are DataFrames

# Drop unrelated columns - chosen based on previous testing? on what descriptors are the best predictors for the Category?
cols_to_drop = ['ID', 'Course Name', 'Course URL / Code', 'Certification URL','Consolidated Course Name', 'Assigned To', 'Request Status', 'Start Date', 'End Date', 'Start Mo/Yr', 'End Mo/Yr', 'Start FY', 'End FY', 'Individual Travel Hours', 'Rqst Tot Labor Hrs', 'Airfare', 'Hotel', 'Per Diem', 'Other', 'Estimated Individual Travel', 'Misc Expenses', 'Catering', 'Facility Rental', 'Direct Other Expenses', 'Describe Other Expenses', 'Direct Expense Impact', 'Rqst NPR Alloc', 'Rqst NPR OH', 'Cancel No Response', 'Created', 'Retroactive Start Date', 'Duplicates', 'Reporting Status']
# Identfiers used to help identify the categories
cols_to_keep = ['Training Source', 'Home Office/Metro Area', 'Organization Number', 'Organization', 'Capability', 'Function 2', 'Career Level', 'Function', 'Function Name', 'Title','Training Type', 'Training Provider', 'Training Delivery Type', 'Training Location', 'Vendor Name', 'Conference Name', 'Course or Event Name', 'Certification Type', 'Certification Name', 'Is there a course with this certification?', 'Activity', 'Support Group', 'Business Justification', 'What % of the conference is business development?', 'Travel Required']

train_base = train_base.drop(cols_to_drop, axis=1)
out_base = test_base.drop(cols_to_drop, axis=1)

# Combine data together to 'dummify' it all at once?
combined_base = pd.concat([train_base, out_base], axis=0)
combined_dummies = pd.get_dummies(data = combined_base, columns = cols_to_keep, drop_first = True)     # Dummy values allow training algorithm to process the values - can't process strings

# Separate the training rows from the test rows
combined_train = combined_dummies.dropna(subset=['Category'])       # Train rows have a Category, so ignore the ones that don't
combined_test = combined_dummies[combined_dummies.Category.isnull()].drop(['Category'], axis=1)   # Test rows are where Category is blank, so just take those

#create a map for Category values (again, so they can be processed as ints)
categories = combined_train['Category'].unique()
category_map = dict()
i = 0
for cat in categories:
    category_map[cat] = i
    i  = i+1

#split the training portion - setting very low for final pass, use .2 for reasonable accuracy test
train, test =  train_test_split(combined_train, test_size = .02)

#separate Category
x_train = train.drop(['Category'], axis = 1)        # x is everything but Category column, y is only the categories
y_train = pd.DataFrame(data=train, columns = ['Category'])

x_test = test.drop(['Category'], axis = 1)
y_test = pd.DataFrame(data=test, columns = ['Category'])

# Apply the map to Category to get their int values
y_train_preprocessed = pd.DataFrame(data = y_train['Category'].map(category_map))
y_test_preprocessed = pd.DataFrame(data = y_test['Category'].map(category_map))

LR = LogisticRegression(C=8.5, solver='liblinear')  # Using liblinear to avoid ConvergenceWarning
LR.fit(x_train, y_train_preprocessed.values.ravel())    # Model is learning the relationship between the descriptors and the Category int values - training
predictions = LR.predict(x_test)    # Use info from the training (line above) to predict Category column for x_test - still part of training
print (accuracy_score(y_test_preprocessed.values.ravel(), predictions))

predictions_test = LR.predict(combined_test)    # Test for real! Predict Category column for test csv

# Changing dummy values back into categories/strings
i = 1
for row in predictions_test:
    cat_val = list(category_map.keys())[list(category_map.values()).index(row)]
    test_base.loc[i,'Category'] = cat_val
    i = i+1
    
test_base.to_csv('results.csv')   

# Used this site for understanding: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
