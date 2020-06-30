# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 02:26:59 2020

@author: esteb
"""


from culp.classifier import culp
import numpy
from sklearn.model_selection import train_test_split
import pandas

wine = numpy.array(pandas.read_csv('culp/data/wine.txt', header=None))
data = wine[:, 1:].astype(float)
labels = wine[:, 0]
for i, j in enumerate([1, 2, 3]): # labels should be between 0 and C-1
    labels[labels == j] = i

def preprocess(train_data, test_data):
    '''preprocessing by normalization'''
    M = numpy.mean(train_data, axis=0)
    S = numpy.std(train_data, axis=0)
    S[S == 0] = M[S == 0] + 10e-10  # Controling devision by zero
    return (train_data - M) / S, (test_data - M) / S
    
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
X_train, X_test = preprocess(X_train, X_test)
for lp in ('CN','AA','RA','CS'):
    prediction = culp(X_train, y_train, X_test, link_predictor=lp, similarity='manhattan', k=12)
    print("Prediction Accuracy for Wine Dataset (λ={0}) = {1}%".format(lp, round(100 * float(numpy.mean(prediction == y_test)), 2)))
