import numpy as np
from utils import *
from run_knn import *
from nn4 import *
from sklearn.decomposition import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.grid_search import GridSearchCV

test_inputs = LoadTest()

for i in range(1):
    
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()  

   
    n = 150
    pca2 = PCA(n_components=n, whiten=True).fit(train_inputs)   
    eigenfaces = pca2.components_.reshape((n, 32, 32))    
    train_pre_inputs2 = pca2.transform(train_inputs)
    valid_pre_inputs2 = pca2.transform(valid_inputs)
    test_pre_inputs2 = pca2.transform(test_inputs)  
    
    print("clf1: ")
    param_grid = {'C': [0.1, 0.5, 1, 1.5, 2, 5],
                  'verbose': [1,3,5,9,12], }
    clf1 = GridSearchCV(LogisticRegression(random_state=1, solver='newton-cg',dual=False), param_grid)
       
    c1 = clf1.fit(train_pre_inputs2, train_targets.T[0])
    print c1.score(valid_pre_inputs2, valid_targets)
    pred1 = c1.predict(test_pre_inputs2)
    pred1 = np.reshape(pred1,(pred1.shape[0],1))
    np.savetxt("3lrpre1.csv", pred1, delimiter=",", fmt="%10.5f")  
    
   
    