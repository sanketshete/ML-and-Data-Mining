# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:55:29 2018

@author: sanket
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split


class irisDataSet:
    
    def __init__(self,X,y,irisDatset):
        self.X=X #Store feature matrix
        self.y=y #store response vector
        self.irisDatset=irisDatset
        
    def traindataModel(self):        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, 
                                 random_state=42)# split the data into train and test


        x1_train = X_train[:,0]
        x2_train = X_train[:,1]
        x3_train = X_train[:,2]
        x4_train = X_train[:,3]
        m = len(x4_train)
        x0 = np.ones(m)
        X = np.array([x0, x1_train, x2_train, x3_train, x4_train]).T
        # Initial Coefficients
        coef = np.array([0, 0, 0, 0, 0])
        Y = np.array(y_train)
        alpha = 0.0001
        newCoef = irisDataSet.grad_desc(X, Y, coef, alpha, 100000)
        # New Values of coef
        print("New Coefficients")
        print(newCoef)
        x5_test = X_test[:,0]
        x6_test = X_test[:,1]
        x7_test = X_test[:,2]
        x8_test = X_test[:,3]
        m = len(x5_test)
        x9 = np.ones(m)
        X5 = np.array([x9, x5_test,x6_test,x7_test,x8_test]).T
        
        Y_pred = X5.dot(newCoef)
        print("Root mean squre")
        print(irisDataSet.rootMeanSqur(y_test, Y_pred))
        light_color = ListedColormap(['#FF0000', '#62A9FF', '#99FF66'])
        plt.scatter(y_test,Y_pred,s=60,c=y_test,cmap=light_color,edgecolor='k')                              
        plt.show()
    def grad_desc(X, Y, coef, alpha, iterations):
        m = len(Y)

        for iteration in range(iterations):
            # Hypothesis Values
            h_value = X.dot(coef)
            loss = h_value - Y
            gradient = X.T.dot(loss) / m  # Gradient Calculation
            coef = coef - alpha * gradient
            # New Cost Value        
        return coef
    
    def rootMeanSqur(Y_test, Y_pred):
        rootmeansq = np.sqrt(sum((Y_test - Y_pred) ** 2) / len(Y_test))
        return rootmeansq

                            
                                      
def main():
    irisDatsetVar=datasets.load_iris()  #Load the dataset
    IrisDatasetObj=irisDataSet(irisDatsetVar.data,irisDatsetVar.target,
                               irisDatsetVar) 
    IrisDatasetObj.traindataModel() 
    
if __name__ == "__main__":
     main()
     
