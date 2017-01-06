import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
import matplotlib.pyplot as mp

def run_logistic_regression(weight_regularization):
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 1, 
                    'weight_regularization': weight_regularization, # penalized
                    'num_iterations': 20
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.transpose([np.random.normal(0, hyperparameters['weight_regularization'], M+1)])

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    
    ce_train = []
    ce_test = []
    fc_train = []
    fc_test = []

    # Begin learning with gradient descent
    for t in xrange(hyperparameters['num_iterations']):
        

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
       
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
        stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
        print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100),
                              float(cross_entropy_valid),
                              float(frac_correct_valid*100))
        ce_train = np.append(ce_train, cross_entropy_train)
        ce_test = np.append(ce_test, cross_entropy_valid)
        fc_train = np.append(fc_train, frac_correct_train)
        fc_test = np.append(fc_test, frac_correct_valid)
    
    mp.plot(ce_train)
    mp.plot(ce_test)
    mp.savefig("ce" + str(weight_regularization) + ".png")
    mp.clf()
    mp.plot(fc_train)
    mp.plot(fc_test)
    mp.savefig("fc" + str(weight_regularization) + ".png")
    mp.clf()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    penalizations = [0.001, 0.01, 0.1, 1.0]
    for item in penalizations:
            run_logistic_regression(item)
