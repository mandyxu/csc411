#from util import *
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def TransTargets(targets):  
  maximum = 7 #number of output classes
  result = np.array([np.zeros(maximum)])  
  for i in range(targets.shape[0]):
    new = np.array([np.zeros(maximum)])
    new[0][targets[i]-1] = 1
    result = np.concatenate((result, new), axis=0)    
  return result[1:]



def TrainNN(inputs_train, target_train, inputs_valid, target_valid, num_hiddens, eps, momentum, num_epochs):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """
  inputs_train = inputs_train.T
  inputs_valid = inputs_valid.T
  target_train = TransTargets(target_train).T
  target_valid = TransTargets(target_valid).T
  
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  num_train_cases = inputs_train.shape[1]
  for epoch in range(num_epochs):
       
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    #h_output = np.exp(h_input) / sum(np.exp(h_input))   # Output of hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = np.exp(logit) / sum(np.exp(logit))  # Output prediction.

    train_mce_n = (np.argmax(target_train, axis=0) == np.argmax(prediction,axis=0)).mean() 
    #print train_mce_n

    # Compute derivation 
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

  train_predicted = Predict(inputs_train, W1, W2, b1, b2)
  valid_predicted = Predict(inputs_valid, W1, W2, b1, b2)
  final_train_mce = MCE(train_predicted, target_train)
  final_valid_mce = MCE(valid_predicted, target_valid)

  print ('MCE: Train %.5f Validation %.5f' % (final_train_mce, final_valid_mce))
  
  return W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, final_train_mce, final_valid_mce

def MCE(prediction, target):
  """Calculates the mean classification error of the model on inputs and target."""
  #print "\n", np.argmax(target, axis=0), "\n",np.argmax(prediction,axis=0)
  frac_correct = (np.argmax(target, axis=0) == np.argmax(prediction,axis=0)).mean() 
  return frac_correct

def Predict(inputs, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs.T) + b1  # Input to hidden layer.
  #h_output = np.exp(h_input) / sum(np.exp(h_input)) 
  h_output = 1 / (1 + np.exp(-h_input))
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = np.exp(logit) / sum(np.exp(logit))  # Output prediction. 
  ans = np.argmax(prediction, axis=0) + 1
  return np.reshape(ans,(ans.shape[0],1))


def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print ('Writing model to %s' % modelfile)
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

