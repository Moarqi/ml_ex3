#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from utils import plot_classification_dataset, plot_2d_decisionboundary


# My own implementation of the knn classifier
class MyKnnClassifier(ClassifierMixin):
    def __init__(self, k=3):
        self.X = None
        self.y = None
        self.k = k

    # ATTENTION: We assume that the data is stored row wise!
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Note: The current data point can be accessed by X[i,:]
            distArray = np.empty((self.X.shape[0],2))                       
            for j in range(self.X.shape[0]):
                vec = X[i,:] - self.X[j,:]
                dist = np.sqrt(vec.dot(vec))
                distArray[j] = (j, dist)                    
            # sort array by second column:
            sortedArray = distArray[distArray[:,1].argsort()]            
            kNearest = sortedArray[0:self.k, :]
            summedClass = 0
            for j in range(self.k):
                summedClass = summedClass + self.y[kNearest[j][0]]

            if (summedClass > self.k//2):
                y_pred[i] = 1
            else:
                y_pred[i] = 0                        

        return y_pred


# TEST
if __name__ == "__main__":
    # Load data
    data = np.load('data_3_myknn.npz')
    X, y = data['X'], data['y']
    print(X.shape)
    print(y.shape)

    # Plot data
    #plot_classification_dataset(X, y)  # Uncomment this line if you want to plot the data set

    # Splitting data into a training and testset
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.33)

    # Fit model
    #from sklearn.neighbors import KNeighborsClassifier
    #model = KNeighborsClassifier(n_neighbors=5)    # Uncomment this line to use the scikit-learn implementation
    model = MyKnnClassifier(k=5)
    model.fit(Xtrain, ytrain)

    print("Accuracy: {0}".format(accuracy_score(ytest, model.predict(Xtest))))

    # Plot decision boundary of the model
    plot_2d_decisionboundary(model, X, y)
