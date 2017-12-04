#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:56:51 2017

@author: gengzhen
"""

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

reader = csv.reader(open("data.csv", "rt"), delimiter=",")
x = list(reader)


data = pd.DataFrame(x)
X1 = data.iloc[1:, 1]
Y = pd.DataFrame({'diagnosis': X1})
X = data.iloc[1:, 2:32]


print(Y.head())
print('-------------------')


X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)









