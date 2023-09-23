# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:06:31 2023

@author: aravi
"""

import pandas as pd
import sweetviz

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import pandas as pd
import warnings
import random

from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# import dabl

# Ignore all warnings (not recommended)
warnings.filterwarnings("ignore")
# sns.set_style("darkgrid")


df = pd.read_csv(r'H:\NIIT\Python\Py_prac\MGP\Proj1_RTA\RTA Dataset.csv')

df.describe()
df.dtypes

df2 = df.drop(['Educational_level','Time','Owner_of_vehicle','Work_of_casuality'],axis=1) # dropping unnecessary columns


# Identify categorical columns
categorical_columns = df2.select_dtypes(include=['object']).columns

# Initialize label encoders and encode categorical columns
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df2[column] = le.fit_transform(df2[column])
    label_encoders[column] = le

missing_values = df2.isnull().sum()
percentage_missing = (missing_values / len(df2)) * 100
missing_data_summary = pd.DataFrame({'Missing Values': missing_values, 'Percentage Missing': percentage_missing})

print(missing_data_summary)

# df3 = df2.copy(deep=True)


# knn_imputer = KNNImputer(n_neighbors=3, weights="uniform")

# df_imputed = pd.DataFrame(knn_imputer.fit_transform(df4), columns=df4.columns)

X = df2.drop('Accident_severity', axis=1)  # Features (all columns except the target)
y = df2['Accident_severity']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# feature selection
# Initialize SelectKBest with the desired number of features (e.g., k=5)
k_best = SelectKBest(score_func=f_classif, k=5)

# Fit and transform the feature selection on the training data
X_train_k_best = k_best.fit_transform(X_train, y_train)

# Transform the test data using the same feature selection
X_test_k_best = k_best.transform(X_test)

# class_weights = {2: 1.0, 1: 7.0, 3: 80.0}
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

# clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_k_best, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_k_best)

# Evaluate the model on the test data
accuracy = clf.score(X_test_k_best, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Get the indices of selected features
selected_feature_indices = k_best.get_support(indices=True)

# Get the column names of the selected features
selected_feature_names = X.columns[selected_feature_indices]

# Print the selected feature names
print("Selected Feature Names:", selected_feature_names)

# Calculate Precision
precision = precision_score(y_test, y_pred, average=None)

# Calculate Recall
recall = recall_score(y_test, y_pred, average=None)
