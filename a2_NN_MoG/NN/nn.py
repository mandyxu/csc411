from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def TrainNN(num_hiddens, eps, momentum, num_epochs):
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

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  train_mce = []
  valid_mce = []
  num_train_cases = inputs_train.shape[1]
  for epoch in range(num_epochs):
    if epoch // 100 != 0:
          ShowMeans(W1)
          raw_input('Press Enter to show W1 after 10 epochs.')    
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))
    
    # Compute mce of the nth element (mean classification error)
    train_mce_n = (target_train == (prediction < 0.5)).mean() 

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

    valid_CE = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
    valid_mce_n = MCE(inputs_valid, target_valid, W1, W2, b1, b2)

    train_error.append(train_CE)
    valid_error.append(valid_CE)
    train_mce.append(train_mce_n)
    valid_mce.append(valid_mce_n)
    sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f' % (epoch, train_CE, valid_CE))
    sys.stdout.write('\rStep %d Train mce %.5f Validation mce %.5f' % (epoch, train_mce_n, valid_mce_n))
    sys.stdout.flush()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  sys.stdout.write('\n')
  final_train_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_error = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
  final_train_mce = MCE(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_mce = MCE(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_mce = MCE(inputs_test, target_test, W1, W2, b1, b2)
  
  print ('Error: Train %.5f Validation %.5f Test %.5f' % (final_train_error, final_valid_error, final_test_error))
  print ('MCE: Train %.5f Validation %.5f Test %.5f' % (final_train_mce, final_valid_mce, final_test_mce))
  
  return W1, W2, b1, b2, train_error, valid_error, train_mce, valid_mce

def MCE(inputs, targets, W1, W2, b1, b2):
  """Calculates the mean classification error of the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.  
  frac_correct = (targets == (prediction < 0.5)).mean() 
  return frac_correct

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  return CE

def DisplayErrorPlot(train_error, valid_error, name):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel(name)
  plt.legend()
  plt.draw()
  plt.savefig(name + " Plot")
  #raw_input('Press Enter to exit.')

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

def main():
  num_hiddens = 10
  eps = 0.02
  momentum = 0.5
  num_epochs = 2000
  W1, W2, b1, b2, train_error, valid_error, train_mce, valid_mce = TrainNN(num_hiddens, eps, momentum, num_epochs)
  DisplayErrorPlot(train_error, valid_error, 'Cross Entropy') 
  DisplayErrorPlot(train_mce, valid_mce, 'Mean Classification Error')  
 
  ShowMeans(W1)
  
def ShowMeans(means):
  """Show the cluster centers as images."""
  plt.figure(1)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')   ##If you wish to save the model for future use :
  #outputfile = 'model.npz'
  #SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

if __name__ == '__main__':
  main()
