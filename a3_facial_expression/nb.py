"""
The Naive Bayes classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
#from plot_digits import *
from utils import *

_SMALL_CONSTANT = 1e-10

class NaiveBayesClassifier(object):
    """
    A simple naive Bayes classifier implementation for binary classification.
    All conditional distributions are univariate Gaussians.
    """
    def __init__(self):
        self.model_learned = False

    def trainNB(self, training_data, training_label):
        """
        Train the naive Bayes classifier on the given training set.

        training_data: n_examples x n_dimensions data matrix.
        training_label: n_examples x 1 dimensional label vector: {1, ..., 7}
        """
        n_examples, n_dims = training_data.shape
        training_label = training_label.squeeze()

        K = 7
        prior = np.empty(K, dtype=np.float)
        mean = np.empty((K, n_dims), dtype=np.float)
        var = np.empty((K, n_dims), dtype=np.float)

        for k in range(K):
            prior[k] = (training_label == (k+1)).mean()
            mean[k, :] = training_data[training_label == (k+1), :].mean(axis=0)
            var[k, :] = training_data[training_label == (k+1), :].var(axis=0)

        self.log_prior = np.log(prior + _SMALL_CONSTANT)
        self.mean = mean
        self.var = var + _SMALL_CONSTANT
        self.model_learned = True

    def predict(self, test_data):
        """
        Generate predictions using the learned model on test data.

        test_data: n_examples x n_dimensions data matrix.

        Return: n_examples x 1 dimensional binary label vector, which is the
        predictions for the test data.
        """
        if not self.model_learned:
            raise Exception('You should learn a model first!')

        K = self.log_prior.size
        n_examples = test_data.shape[0]

        log_prob = np.zeros((n_examples, K), dtype=np.float)

        for k in range(K):
            log_prob[:, k] = (-0.5 * ((test_data - self.mean[k, :][np.newaxis, :])**2 / \
                                      self.var[k, :][np.newaxis, :]) - \
                              0.5 * np.log(self.var[k, :][np.newaxis, :])).sum(axis=1) + \
                                    self.log_prior[k]

        return log_prob.argmax(axis=1)[:, np.newaxis] + 1

    def compute_accuracy(self, test_data, test_label):
        return (self.predict(test_data) == test_label).mean()

def main():
    """
    Learn a Naive Bayes classifier on the digit dataset, evaluate its
    performance on training and test sets, then visualize the mean and variance
    for each class.
    """
    
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    # add your code here (it should be less than 10 lines)
    self = NaiveBayesClassifier()
    self.train(train_inputs, train_targets)
    test_prediction = self.predict(test_inputs)
    valid_accuracy = self.compute_accuracy(valid_inputs, valid_targets)
    test_accuracy = self.compute_accuracy(test_inputs, test_targets)
    print "valid accuracy: ", valid_accuracy, "\n test accuracy: ", test_accuracy
    
    plot_digits(self.mean)
    plt.savefig("2.4 mean.png")
    plt.clf()    
    plot_digits(self.var)
    plt.savefig("2.4 variance.png")
    plt.clf()    
    
if __name__ == '__main__':
    main()
