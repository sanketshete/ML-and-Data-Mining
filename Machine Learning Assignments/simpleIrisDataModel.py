# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:44:36 2018

@author: sanket
"""
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


class irisDataSet:
    
    def __init__(self,X,y,irisDatset):
        self.X=X #Store feature matrix
        self.y=y #store response vector
        self.irisDatset=irisDatset
        
    def traindataModel(self):
        global Knn
        Knn=KNeighborsClassifier()
        Knn.fit(self.X,self.y) #Train the datamodel

    def plotGraph(self):
        bold_color = ListedColormap(['#000066', '#5757FF', '#99FF00'])
        light_color = ListedColormap(['#0000FF', '#62A9FF', '#99FF66'])                            
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx_mesh, yy_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
        Z = Knn.predict(np.c_[xx_mesh.ravel(), yy_mesh.ravel()])
        Z = Z.reshape(xx_mesh.shape)
        plt.figure()
        plt.pcolormesh(xx_mesh, yy_mesh, Z, cmap=light_color)
        plt.scatter(self.irisDatset.data[:,2],self.irisDatset.data[:,3],s=20,c
                    =self.y,cmap=bold_color,edgecolor='k') #plot a scatter graph
        plt.xlabel("petalLength")
        plt.ylabel("petalWidth")
        plt.show()


def main():
    irisDatsetVar=datasets.load_iris()  #Load the dataset
    IrisDatasetObj=irisDataSet(irisDatsetVar.data[:,2:],irisDatsetVar.target,
                               irisDatsetVar) 
    IrisDatasetObj.traindataModel() 
    IrisDatasetObj.plotGraph()
    
if __name__ == "__main__":
     main()