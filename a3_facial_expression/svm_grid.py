import numpy as np
import random as rd
from utils import *
import scipy.io as sio
import os.path
from sklearn import svm
from sklearn import datasets
from sklearn.decomposition import *
from sklearn.grid_search import GridSearchCV


accuracy = {'clf1':[],'clf2':[]}

test_inputs = LoadTest()
unlabeled = LoadTest("U")

for i in range(1):
    print i
    train_inputs, train_labels, valid_inputs, valid_labels = LoadData()   
    
    #N, dim = train_inputs.shape
    
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150
    
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, unlabeled.shape[0]))
    
    pca = PCA(n_components=n_components, whiten=True).fit(test_inputs)   
    eigenfaces = pca.components_.reshape((n_components, 32, 32))
    
    train_pre_inputs = pca.transform(train_inputs)
    valid_pre_inputs = pca.transform(valid_inputs)
    test_pre_inputs = pca.transform(test_inputs)
    
    ## Uncomment out if want to use original data:
    #train_pre_inputs = train_inputs
    #valid_pre_inputs = valid_inputs
    #test_pre_inputs = test_inputs   
    
    # fit SVM
    print("clf1: ")
    param_grid = {'C': [2, 1e3, 5e3, 1e4],
                  'gamma': [0.01, 0.04, 0.1], }
    clf1 = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    clf1 = clf1.fit(train_pre_inputs, train_labels.T[0])
    
    # Predict valid_set:
    valid_prediction = clf1.predict(valid_pre_inputs)
    score1 = clf1.score(valid_pre_inputs, valid_labels.T[0])
    accuracy["clf1"].append(score1)
    print score1
    
    # Predict test_set:
    test_prediction = clf1.predict(test_pre_inputs)
    test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],1)) 
    
    ## fit SVM
    #print("clf2: ")    
    #clf2 = clf2.fit(train_pre_inputs, train_labels.T[0])
    
    ## Predict valid_set:
    #valid_prediction = clf2.predict(valid_pre_inputs)
    
    ## Predict test_set:
    #test_prediction = clf2.predict(test_pre_inputs)
    #test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],1))  
    #score2 = clf2.score(valid_pre_inputs, valid_labels.T[0])
    #print score2
    #accuracy["clf2"].append(score2)


