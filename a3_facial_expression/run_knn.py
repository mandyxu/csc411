import numpy as np
from utils import *
from l2_distance import l2_distance
import matplotlib.pyplot as mp
from plot_digits import *
from matplotlib.legend_handler import HandlerLine2D
from sklearn.decomposition import *
    
def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]
    results = np.zeros([valid_labels.shape[0],1], int)
    
    for i in range(valid_labels.shape[0]):
        count = np.bincount(valid_labels[i])
        max_count = np.argmax(count)
        results[i] = max_count 

    return results

def knn_MCE(predictions, targets):
    return (targets == predictions).mean()


if __name__ == '__main__':   
    mce = []
    for i in range(50):
        train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
        
        #n_components = 150    
        #pca = PCA(n_components=n_components, whiten=True).fit(train_inputs)   
        #eigenfaces = pca.components_.reshape((n_components, 32, 32))
        #train_pre_inputs = pca.transform(train_inputs)
        #valid_pre_inputs = pca.transform(valid_inputs)    
        ##test_pre_inputs = pca.transform(test_inputs)    
        
        #pred_valid = run_knn(5, train_pre_inputs, train_targets, valid_pre_inputs)
        #valid_knn =  knn_MCE(pred_valid, valid_targets)
        #mce.append(valid_knn)
        #print valid_knn
        
        
        pred_valid = run_knn(5, train_inputs, train_targets, valid_inputs)
        valid_knn =  knn_MCE(pred_valid, valid_targets)
        mce.append(valid_knn)
        print valid_knn    
    print np.mean(mce)