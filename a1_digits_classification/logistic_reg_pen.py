import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
import matplotlib.pyplot as mp

def run_logistic_pen_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.01, 
                    'weight_regularization': [0.001,0.01,0.1,1.0], 
                    'num_iterations': 80
                 }
    
    ce_train = []
    ce_valid = []
    ce_test = []
    fe_train = []
    fe_valid = []   
    fe_test = []
    
    for item in hyperparameters['weight_regularization']:
        
        hyperparameter = {
                            'learning_rate': hyperparameters['learning_rate'], 
                            'weight_regularization': item,
                            'num_iterations': hyperparameters['num_iterations']
                         }    
        
        ce_train_sub = []
        ce_valid_sub = []
        ce_test_sub = []
        fe_train_sub = []
        fe_valid_sub = [] 
        fe_test_sub = []
               
        
        for i in range(10):

            # Logistic regression weights
            # TODO:Initialize to random weights here.
            #weights = np.transpose([np.random.normal(0, hyperparameter['weight_regularization'], M+1)])
            weights = np.transpose([np.random.normal(0, 0.1, M+1)])
            
            
            # Verify that your logistic function produces the right gradient.
            # diff should be very close to 0.
            run_check_grad(hyperparameter)
            
            
            # Begin learning with gradient descent
            for t in xrange(hyperparameter['num_iterations']):
                
            
                # TODO: you may need to modify this loop to create plots, etc.
            
                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameter)
                ft, dft, predictionst = logistic_pen(weights, test_inputs, test_targets, hyperparameter)
               
                # Evaluate the prediction.
                cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
                cross_entropy_test, frac_correct_test = evaluate(test_targets, predictionst)
                
            
                if np.isnan(f) or np.isinf(f):
                    raise ValueError("nan/inf error")
            
                # update parameters
                weights = weights - hyperparameter['learning_rate'] * df / N
            
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
                
                
                # Append the values for the last iteration:
                if t == (hyperparameter['num_iterations'] - 1):
                    
                    # Calculate the classification error:
                    frac_error_train = 1 - frac_correct_train 
                    frac_error_valid = 1 - frac_correct_valid
                    frac_error_test = 1 - frac_correct_test
                    
                    ce_train_sub = np.append(ce_train_sub, cross_entropy_train)
                    ce_valid_sub = np.append(ce_valid_sub, cross_entropy_valid)
                    ce_test_sub = np.append(ce_test_sub, cross_entropy_test)                    
                    fe_train_sub = np.append(fe_train_sub, frac_error_train)
                    fe_valid_sub = np.append(fe_valid_sub, frac_error_valid)
                    fe_test_sub = np.append(fe_test_sub, frac_error_test)

        ce_train = np.append(ce_train, ce_train_sub.mean())
        ce_valid = np.append(ce_valid, ce_valid_sub.mean())
        ce_test = np.append(ce_test, ce_test_sub.mean())
        fe_train = np.append(fe_train, fe_train_sub.mean())
        fe_valid = np.append(fe_valid, fe_valid_sub.mean()) 
        fe_test = np.append(fe_test, fe_test_sub.mean()) 
        
        # Report test errrors:
        print "\n [0.001,0.01,0.1,1]"
        print "Training set:\n frac_error: ", fe_train, "\n ce: ", ce_train
        print "Validation set: \n frac_error: ", fe_valid, "\n ce: ", ce_valid  
        print "Testing set: \n frac_error: ", fe_test, "\n ce: ", ce_test   
    
    # Plot cross entropy of training and testing sets
    w = np.log(hyperparameters['weight_regularization'])
    mp.plot(w, ce_train, label="training", linewidth=4)
    mp.plot(w, ce_valid,label="validation", linestyle="--")
    mp.xticks(w, hyperparameters['weight_regularization'])
    mp.title("Plot of Average Cross Entropy")
    mp.xlabel("penalty parameters")
    mp.ylabel("avg. cross entropy")
    mp.savefig("2.3_ce_pen.png")
    mp.clf()
    
    
    # Plot fraction error of training and testing sets
    mp.plot(w, fe_train, label="training", linewidth=4)
    mp.plot(w, fe_valid,label="validation", linestyle="--")
    mp.xticks(w, hyperparameters['weight_regularization'])    
    mp.title("Plot of Classification Error")
    mp.xlabel("penalty parameters")
    mp.ylabel("classification error")
    mp.savefig("2.3_fe_pen.png")
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
    run_logistic_pen_regression()
