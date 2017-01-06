import numpy as np
from utils import *
from run_knn import *
from nb import *
from nn4 import *
from autoencoder import *
#from logistic_reg import *

   
def demo():
    models = {'knn':True, 'logistic':False, 'pca':False, 'nn':True, 'nb':False}
    pre = {'pca':True, 'nb':True, 'auto':False}
    
    p = 30 # % of information used
    
    print "Loading data ..."    
    # Load Training and Validation data sets:
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
    
    # Load Testing data set:
    #train_inputs, train_targets, valid_inputs, valid_targets = LoadTest('P')
    
    print "Preprocessing ..."
    #if pre['nb']:
        #v = nbayes(train_inputs, train_targets,0, (100-p)) 
        #train_pre_inputs_nb = train_inputs[:,v]
        #valid_pre_inputs_nb = valid_inputs[:,v]       
    
    #if pre['pca']:    
        #V_train = pca(train_inputs, int(train_inputs.shape[1]*p/100))
        #train_pre_inputs_pca = np.dot(train_inputs, V_train.T)
        #valid_pre_inputs_pca = np.dot(valid_inputs, V_train.T)  
        
    if pre['auto']:
        num_hiddens = 2000
        eps = 0.1 # leanrning rate
        momentum = 0.5
        W1, W2, b1, b2, train_pre_inputs_auto, valid_pre_inputs_auto = AutoEncoder(train_inputs, valid_inputs, num_hiddens, eps, momentum, 200)
        train_pre_inputs_auto = train_pre_inputs_auto.T
        valid_pre_inputs_auto = valid_pre_inputs_auto.T
        
       
    
    ### Model 1: knn 
    if models['knn']:
        # nb
        print "KNN nb ..."
        pred_valid = run_knn(5, train_pre_inputs_nb, train_targets, valid_pre_inputs_nb)
        valid_knn_nb =  knn_MCE(pred_valid, valid_targets)
        
        # pca
        print "KNN pca ..."
        pred_valid = run_knn(5, train_pre_inputs_pca, train_targets, valid_pre_inputs_pca)
        valid_knn_pca = knn_MCE(pred_valid, valid_targets)        
          
    ## Model 2: logistic
    if models['logistic']:
        learning_rates = [0.01]  
        #learning_rates = [0.001,0.01,0.1,1.0] 
        for item in learning_rates:  
            run_logistic_regression(item)
            
    
    ## Model 3: Naive Bayes 
    if models['nb']:
        nb = NaiveBayesClassifier()
        nb.trainNB(train_inputs, train_targets)
        valid_prediction = nb.predict(valid_inputs)        
        nb_valid_accuracy = nb.compute_accuracy(valid_inputs, valid_targets)
        print ('Naive Bayes MCE: ', nb_valid_accuracy)
        
        
        #np.savetxt("nb_mean.txt", nb.mean * 255, delimiter=",", fmt="%10.5f") 
        #np.savetxt("nb_var.txt", nb.var * 255, delimiter=",", fmt="%10.5f") 
    
    
    # Model 4: NN
    if models['nn']:  
        num_hiddens = 15
        eps = 0.1
        momentum = 0.5
        num_epochs = 7000
        
        # nb
        print "NN nb ..."
        W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn_nb, valid_nn_nb = TrainNN(train_pre_inputs_nb, train_targets, valid_pre_inputs_nb, valid_targets, num_hiddens, eps, momentum, num_epochs)
        
        ## auto encoder
        #print "NN auto ..."
        #W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn_auto, valid_nn_auto = TrainNN(train_pre_inputs_auto, train_targets, valid_pre_inputs_auto, valid_targets, num_hiddens, eps, momentum, num_epochs)         
        
        num_hiddens = 15
        eps = 0.1
        momentum = 0.5
        num_epochs = 15000          
              
        # pca
        print "NN pca ... "
        W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn_pca,  valid_nn_pca = TrainNN(train_pre_inputs_pca, train_targets, valid_pre_inputs_pca, valid_targets, num_hiddens, eps, momentum, num_epochs)
        
        # none
        print "NN none ... "
        W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn, valid_nn = TrainNN(train_inputs, train_targets, valid_inputs, valid_targets, num_hiddens, eps, momentum, num_epochs)
    
    train_nn_auto = 0
    valid_nn_auto = 0
    
    return train_nn, train_nn_nb, train_nn_pca, valid_nn, valid_knn_nb, valid_knn_pca, valid_nn_nb, valid_nn_pca, train_nn_auto, valid_nn_auto
    
    ##np.savetxt("pca.txt", pca[:100] * 255, delimiter=",", fmt="%10.5f")   
    ###plot_digits(train_inputs)
    ###plt.savefig("train_inputs.png")
    
if __name__ == '__main__':   
    
    mce_train_nn = []
    mce_train_nn_nb = []
    mce_train_nn_pca = [] 
    mce_train_nn_auto = []    
    mce_valid_knn_nb = [] 
    mce_valid_knn_pca = []
    mce_valid_nn = []
    mce_valid_nn_nb = []
    mce_valid_nn_pca = []
    mce_valid_nn_auto = []
    for i in range(1):
        print "\n","Iteration #", i, ":"
        train_nn, train_nn_nb, train_nn_pca, valid_nn, valid_knn_nb, valid_knn_pca, valid_nn_nb, valid_nn_pca, train_nn_auto, valid_nn_auto = demo()
        
        mce_train_nn.append(train_nn)
        mce_train_nn_nb.append(train_nn_nb)
        mce_train_nn_pca.append(train_nn_pca)
        mce_train_nn_auto.append(train_nn_auto)
        mce_valid_knn_nb.append(valid_knn_nb)
        mce_valid_knn_pca.append(valid_knn_pca)
        mce_valid_nn.append(valid_nn)
        mce_valid_nn_nb.append(valid_nn_nb)
        mce_valid_nn_pca.append(valid_nn_pca)
        mce_valid_nn_auto.append(valid_nn_auto)
        
    print "=== Training ==="
    print "nn none: ", np.array(mce_train_nn).mean(), mce_train_nn
    print "nn nb: ", np.array(mce_train_nn_nb).mean(), mce_train_nn_nb
    print "nn pca: ", np.array(mce_train_nn_pca).mean(), mce_train_nn_pca
    print "nn auto: ", np.array(mce_train_nn_auto).mean(), mce_train_nn_auto
    print "=== Validation ==="
    print "knn nb: ", np.array(mce_valid_knn_nb).mean(), mce_valid_knn_nb
    print "knn pca: ", np.array(mce_valid_knn_pca).mean(), mce_valid_knn_pca
    print "nn none: ", np.array(mce_valid_nn).mean(), mce_valid_nn
    print "nn nb: ", np.array(mce_valid_nn_nb).mean(), mce_valid_nn_nb
    print "nn pca: ", np.array(mce_valid_nn_pca).mean(), mce_valid_nn_pca
    print "nn auto: ", np.array(mce_valid_nn_auto).mean(), mce_valid_nn_auto