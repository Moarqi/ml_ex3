#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import plot_2d_decisionboundary


if __name__ == "__main__":
    # Load data
    data = np.load('data_3_logreg_a.npz')
    X, y = data['X'], data['y']
    print(X.shape)
    print(y.shape)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.33)
    model = LogisticRegression(multi_class='ovr', solver='liblinear')
    model.fit(Xtrain, ytrain)
    #print("Test-Accuracy: {0}".format(accuracy_score(ytest, model.predict(Xtest))))
    print("Test-Accuracy: {0}".format(model.score(Xtest, ytest)))
    print("Train-Accuracy: {0}".format(model.score(Xtrain, ytrain)))
    plot_2d_decisionboundary(model, X, y)
    
