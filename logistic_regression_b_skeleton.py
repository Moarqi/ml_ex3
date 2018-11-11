#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import plot_classification_dataset, plot_2d_decisionboundary


if __name__ == "__main__":
    # Load data
    data = np.load('data_3_logreg_b.npz')
    X, y = data['X'], data['y']
    print(X.shape)
    print(y.shape)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.33)

    model = LogisticRegression(multi_class='ovr', solver='liblinear')    
    # Fit and evaluate (compute test error) logistic regression on this 1d data set
    model.fit(Xtrain, ytrain)    
    print("Test-Accuracy: {0}".format(model.score(Xtest, ytest)))
    print("Train-Accuracy: {0}".format(model.score(Xtrain, ytrain)))    
    # DOC: Accuracy is pretty bad (around 0.5) as the 1 dimensional data with points of class 0
    # surrounding the points of class 1 is not possible to split in half!    
    

    # Inspect the data set
    plot_classification_dataset(X, y)

    # Feature transformation (1d -> 2d)
    X2 = np.copy(X)
    X2.resize((X.shape[0], 2))
    print(X2.shape)
    for i in range(X2.shape[0]):
        X2[i][1] = X2[i][0] * X2[i][0]

    # split new 2d data set
    X2train, X2test, y2train, y2test = train_test_split(X2,y,test_size=0.33)

    # Fit logistic regression to  new 2d data set    
    # Evaluate the model (compute test error)
    model2 = LogisticRegression(multi_class='ovr', solver='liblinear')    
    # Fit and evaluate (compute test error) logistic regression on this 1d data set
    model2.fit(X2train, y2train)    
    print("Test-Accuracy: {0}".format(model2.score(X2test, y2test)))
    print("Train-Accuracy: {0}".format(model2.score(X2train, y2train)))

    # Visualize the decision boundary of the final model
    plot_2d_decisionboundary(model2, X2, y)

    #DOC: Performance for the 2-dim data is significantly better as the data can easier be split
    # by a linear curve as they are positioned in 2-dim space