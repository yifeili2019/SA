#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:55:48 2020

@author: emelaktas
"""

# data standardisation
from sklearn import preprocessing

# data splitting
from sklearn.model_selection import train_test_split

# keras for nn building
from keras.models import Sequential

# a dense network
from keras.layers import Dense

# import the numpy library
import numpy as np

# to plot error
import matplotlib.pyplot as plt

# pandas for data analysis
import pandas as pd

# confusion matrix from scikit learn
from sklearn.metrics import confusion_matrix

# for visualisation
import seaborn as sns

# for roc-curve
from sklearn.metrics import roc_curve

# preparing the data
# read data
df_diabetes = pd.read_csv('diabetes.csv')

# first five observations in the dataset
df_diabetes.head()

# are there any missing values
print(df_diabetes.isnull().any())

# descriptives
df_diabetes.describe()

# some variables should never be 0 (eg blood pressure or BMI)
print("Number of rows with 0 values for each variable")
for col in df_diabetes.columns:
    missing_rows = df_diabetes.loc[df_diabetes[col]==0].shape[0]
    print(col + ": " + str(missing_rows))
    
    
# repace zeros with nan s
df_diabetes['Glucose'] = df_diabetes['Glucose'].replace(0, np.nan)
df_diabetes['BloodPressure'] = df_diabetes['BloodPressure'].replace(0, np.nan)
df_diabetes['SkinThickness'] = df_diabetes['SkinThickness'].replace(0, np.nan)
df_diabetes['Insulin'] = df_diabetes['Insulin'].replace(0, np.nan)
df_diabetes['BMI'] = df_diabetes['BMI'].replace(0, np.nan)


# check zeros again
print("Number of rows with 0 values for each variable")
for col in df_diabetes.columns:
    missing_rows = df_diabetes.loc[df_diabetes[col]==0].shape[0]
    print(col + ": " + str(missing_rows))


# replace na with mean
df_diabetes['Glucose'] = df_diabetes['Glucose'].fillna(df_diabetes['Glucose'].mean())
df_diabetes['BloodPressure'] = df_diabetes['BloodPressure'].fillna(df_diabetes['BloodPressure'].mean())
df_diabetes['SkinThickness'] = df_diabetes['SkinThickness'].fillna(df_diabetes['SkinThickness'].mean())
df_diabetes['Insulin'] = df_diabetes['Insulin'].fillna(df_diabetes['Insulin'].mean())
df_diabetes['BMI'] = df_diabetes['BMI'].fillna(df_diabetes['BMI'].mean())
    
# scale data
df_scaled = preprocessing.scale(df_diabetes)

# convert back to data frame
df_scaled = pd.DataFrame(df_scaled, columns=df_diabetes.columns)

# check that scaling worked
df_scaled.describe()

# outcome to be binary
df_scaled['Outcome'] = df_diabetes['Outcome']
df_diabetes2 = df_scaled


# assign features and target
X = df_diabetes2.loc[:, df_diabetes2.columns != 'Outcome']
y = df_diabetes2.loc[:, 'Outcome']

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# split train and validate
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# building the nn
model = Sequential()


# Add the first hidden layer
model.add(Dense(32, activation='relu', input_dim=8))

# Add the second hidden layer
model.add(Dense(16, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model for 200 epochs
model.fit(X_train, y_train, epochs=200)

# testing accuracy
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# confusion matrix
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix,  annot=True, fmt="d",
                 xticklabels=['No Diabetes', 'Diabetes'],
                 yticklabels=['No Diabetes', 'Diabetes'], 
                 cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.savefig('test.png')



y_test_pred_probs = model.predict(X_test)

FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)

plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc.png')
