import numpy as np
from utils import *
from nn4 import *
from sklearn.decomposition import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from classRate import *



if __name__ == '__main__':   

    test_inputs = LoadTest()
    
    for i in range(1):
        print("clf1: ")
  

        train_inputs, train_targets, valid_inputs, valid_targets = LoadData()  
        
        print "Training Regression"        
        param_grid = {'C': [0.1, 0.5, 1, 1.5, 2, 5],
                      'verbose': [1,3,5,9,12], }
        clf_lr = GridSearchCV(LogisticRegression(random_state=1, solver='newton-cg',dual=False), param_grid)              
        clf_lr = clf_lr.fit(train_inputs, train_targets.T[0])
        lr_proba = clf_lr.predict_proba(test_inputs)
        lr_pred = clf_lr.predict(test_inputs)
        
        print "Training svm"
        n = 150
        clf_pca = PCA(n_components=n, whiten=False).fit(test_inputs)   
        eigenfaces = clf_pca.components_.reshape((n, 32, 32))        
        train_pre_inputs = clf_pca.transform(train_inputs)
        valid_pre_inputs = clf_pca.transform(valid_inputs)
        test_pre_inputs = clf_pca.transform(test_inputs)
        
        param_grid = {'C': [2, 1e3, 5e3, 1e4],
                          'gamma': [0.01, 0.04, 0.1], }
        clf_svm = GridSearchCV(svm.SVC(kernel='rbf',probability=True), param_grid)
        clf_svm = clf_svm.fit(train_pre_inputs, train_targets.T[0])
        svm_proba = clf_svm.predict_proba(test_pre_inputs)
        svm_pred = clf_svm.predict(test_pre_inputs)
        
        print "Training NN"
        num_hiddens = 15
        eps = 0.1
        momentum = 0.5
        num_epochs = 10000 
        W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn_nb, valid_nn_nb = TrainNN(train_inputs, train_targets, valid_inputs, valid_targets, num_hiddens, eps, momentum, num_epochs)  
        nn_proba = Predict(test_inputs.T, W1, W2, b1, b2).T
        nn_pred = Prediction(test_inputs.T, W1, W2, b1, b2)

        # Sum up the probability of three models
        pred_proba = lr_proba + svm_proba + nn_proba
        a= np.argmax(pred_proba, axis=1) + 1
        prediction = np.reshape(a, (a.shape[0],1))
        np.savetxt("6vote1.csv", prediction, delimiter=",", fmt="%10.5f") 
        
        # Adjusted:
        svm_pred_v = clf_svm.predict(valid_pre_inputs)
        nn_pred_v = Prediction(valid_inputs.T, W1, W2, b1, b2)
        lr_pred_v = clf_lr.predict(valid_inputs)
        
        print "svm"
        svm_weights = np.array(classificationRate(valid_targets, svm_pred_v) / 0.7)
        print "nn"
        nn_weights = np.array(classificationRate(valid_targets, nn_pred_v) / 0.7)
        print "lr"
        lr_weights = np.array(classificationRate(valid_targets, lr_pred_v) / 0.7)
        pred_proba2 = lr_proba * lr_weights + svm_proba * svm_weights + nn_proba * nn_weights
        a2= np.argmax(pred_proba2, axis=1) + 1
        prediction2 = np.reshape(a2, (a2.shape[0],1))
        np.savetxt("7vote2.csv", prediction2, delimiter=",", fmt="%10.5f") 
        
        # mce on validation
        svm_proba_v = clf_svm.predict_proba(valid_pre_inputs)
        nn_proba_v = Predict(valid_inputs.T, W1, W2, b1, b2).T
        lr_proba_v = clf_lr.predict_proba(valid_inputs)
        pred_proba2_v = lr_proba_v * lr_weights + svm_proba_v * svm_weights + nn_proba_v * nn_weights
        a2= np.argmax(pred_proba2_v, axis=1) + 1
        prediction2_v = np.reshape(a2, (a2.shape[0],1))
        print (prediction2_v == valid_targets).mean()
        