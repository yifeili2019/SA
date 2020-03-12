# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:13:56 2020

@author: Emel Aktas
"""

# import the numpy package
import numpy as np

# import seaborn for graphs
import seaborn as sn
sn.set(color_codes = True)

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import pandas for data analysis
import pandas as pd

# define the dataset file name
dataset_filename = "affinity_dataset.txt"

# load the dataset
X = np.loadtxt(dataset_filename)

# check the dimensions of the loaded data file
X.shape

# check the first five observations
X[:5]

# assign number of observations and number of variables
n_samples, n_features = X.shape

# names of variables (features)
features = ["bread", "milk", "cheese", "apples", "bananas"]

# third row of the X as an example sample
sample = X[2]

# take a look at what's in sample and compare with X[:5]
sample

# the fifth element of sample (corresponds to bananas)
sample[4]

# support and confidence for the rule: "if a person buys bananas, 
# they also buy X"

 # create a default dictionary to capture valid and invalid rules
from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences = defaultdict(int)


# check the entire dataset for each feature as a premise and 
# check the conclusion.
# when the premise is true, if the conclusion is also true, the rule is valid.

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0:
            continue
        # Record that the premise was bought in another transaction
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:
                # It makes little sense to measure if X -> X.
                continue
            if sample[conclusion] == 1:
                # This person also bought the conclusion item
                valid_rules[(premise, conclusion)] += 1

# how many times each product is bought
num_occurences

# how many times bananas were bought together with other products
valid_rules


# support of the rule
support = valid_rules

# confidence calculation
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = valid_rules[rule] / num_occurences [premise]
    
# confidence of the rule. percentage of times the rule applies 
# when the premise applies
confidence    

# In English
for premise, conclusion in confidence:
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".
          format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))
    print(" - Support: {0}".format(support [(premise, conclusion)]))
    print("")
    
from operator import itemgetter

# Let us sort rules by support
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)    

sorted_support

# Let us write in English
for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_support[index][0]
    print("Rule: If a person buys {0} they will also buy {1}".
          format(features[premise], features[conclusion]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")

# Let us sort rules by confidence
sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
sorted_confidence

for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    print("Rule: If a person buys {0} they will also buy {1}".
          format(features[premise], features[conclusion]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))
    print(" - Support: {0}".format(support [(premise, conclusion)]))
    print("")


from matplotlib import pyplot as plt
plt.plot([confidence[rule[0]] for rule in sorted_confidence])
plt.ylabel('Confidence')
plt.xlabel('Rule') # possibly use the first five rules


# let us read the famous iris data set
iris = pd.read_csv("iris.csv")

# First five observations of the data set
iris.head()

# Shape of the data set
iris.shape

#information on the data set
iris.info()

#Descriptive statistics
iris.describe()

# box and whisker plots
#iris.plot(kind='box', sharex=False, sharey=False)

# histograms
#iris.hist(edgecolor='white', linewidth=1.2)

# boxplot on each feature split out by species
#iris.boxplot(by="species",figsize=(10,10))

# violinplots on petal-length for each species 
# https://seaborn.pydata.org/generated/seaborn.violinplot.html
#sn.violinplot(data=iris,x="species", y="petal_length")


from pandas.plotting import scatter_matrix
# scatter plot matrix
#scatter_matrix(iris,figsize=(10,10))
#plt.show()


# Using seaborn pairplot to see the bivariate relation between each 
# pair of features
#sn.pairplot(iris, hue="species", diag_kind = "hist")

# updating the diagonal elements in a pairplot to show a kde
#sn.pairplot(iris, hue="species",diag_kind="kde")
#plt.show()

# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Seperating the data into dependent and independent variables
X = iris.iloc[:, :-1].values
print("x:",X)
print('X.type:',type(X))
print("x.shape:",X.shape)
y = iris.iloc[:, -1].values
print("Y:",y)
print('y.type:',type(y))
print('y.shape:',y.shape)
print(y[0])
print(len(y))

z = np.array([1,2])
print("z:",z)
print("z.shape:",z.shape)
print("z.type:",type(z))

a = np.array([[1,2]])
print("a:",a)
print("a.shape:",a.shape)
print("a.type:",type(a))

b = [1,2,3]
print("b.shape:",b.shape)
'''
# Splitting the dataset into a training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# Support Vector Machine's
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))
'''