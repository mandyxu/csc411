import numpy as np
from util import *
from l2_distance import l2_distance
import matplotlib.pyplot as mp
from plot_digits import *
from matplotlib.legend_handler import HandlerLine2D


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

    dist = l2_distance(valid_data, train_data)
   
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels


if __name__ == '__main__':

    train_inputs, valid_inputs, test_inputs, train_targets, valid_targets, test_targets = LoadData('digits.npz')
    
    k_list = [1,3,5,7,9]
    frac_correct_valid = []
    frac_correct_test = []
   
    for item in k_list:
        # output: predict the labels using k nearest neighbours, where (k = item).
        pred_valid = run_knn(item, train_inputs, train_targets, valid_inputs)
        pred_test = run_knn(item, train_inputs, train_targets, test_inputs)
                
        # Calculate the classification rate
        frac_correct_valid.append((valid_targets == pred_valid.reshape(-1)).mean())
        frac_correct_test.append((test_targets == pred_test.reshape(-1)).mean())
        
    # Plot Classification Rate of Validation set.
    print "cr_Validation: \n", frac_correct_valid
    print "cr_Test: \n", frac_correct_test 
    
    mp.axis([1,max(k_list),0.95,1])
    valid_line = mp.plot(k_list, frac_correct_valid, label="validation", linestyle="--")
    test_line = mp.plot(k_list, frac_correct_test, label="test", linewidth=4)
    
    mp.title("Classification Rates")
    mp.xlabel("k")
    mp.ylabel("classification rates")
    mp.savefig("2.5_knn_cr.png")
    mp.clf()
       

        