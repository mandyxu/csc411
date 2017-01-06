from kmeans import *
from nn import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  #mu = KMeans(x, K, 5)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    #print 'Iter %d logProb %.5f' % (i, logProbX[i])
    
    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
    
    #print p
    
  #raw_input('Press Enter.')
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def q2():
  iters = 40
  minVary = 0.01
  K = 2
  
  print "\n 2:"
  inputs_train2, inputs_valid2, inputs_test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)  
  p2, mu2, vary2, logProbX2 = mogEM(inputs_train2, K, iters, minVary)
  ShowMeans(mu2)
  ShowMeans(vary2)  
  
  print "\n 3:"  
  inputs_train3, inputs_valid3, inputs_test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True) 
  p3, mu3, vary3, logProbX3 = mogEM(inputs_train3, K, iters, minVary)
  ShowMeans(mu3)
  ShowMeans(vary3)  
  
  print "\n 2 & 3:"  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz') 
  p, mu, vary, logProbX = mogEM(inputs_train, K, iters, minVary)
  ShowMeans(mu)
  ShowMeans(vary)      

def q3():
  iters = 20
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  p, mu, vary, logProbX = mogEM(inputs_train, 20, iters, minVary)
  raw_input('Press Enter.')
  ShowMeans(mu)
  ShowMeans(vary)
  print (p) 
  print (logProbX)
  raw_input('Press Enter to continue.')
  
def q4():
  iters=10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0] 
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  # Classification errors
  ce_train = [] 
  ce_valid = [] 
  ce_test = []
  
  n2 = train2.shape[0] 
  n3 = train3.shape[0]      
  
  for t in xrange(T): 
    K = numComponents[t]
    
    # TRAINING SET: 
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------

    p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary)    
    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    
    p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary)    
    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------    

    px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_train))
    px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_train))  
    
    # P(d=1|x):= digit 2's       
    p2x = px2 * n2 / (px2 * n2 + px3 * n3)     
    # P(d=2|x):= digit 3's
    p3x = 1 - p2x            
    # Classification errors
    ce_train.append(((p2x >= 0.5) == target_train).mean())
    
    # VALIDATION SET:     
    px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_valid))
    px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_valid))
    p2x = px2 * n2 / (px2 * n2 + px3 * n3)   
    ce_valid.append(((p2x >= 0.5) == target_valid).mean())   
    
    
    # TESTING SET:    
    px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_test))
    px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_test))     
    p2x = px2 * n2 / (px2 * n2 + px3 * n3) 
    ce_test.append(((p2x >= 0.5) == target_test).mean())     
    
    
  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
 
  plt.plot(numComponents, ce_train, 'b', label='Train')
  plt.plot(numComponents, ce_valid, 'g', label='Validation')
  plt.plot(numComponents, ce_test, 'r', label='Test')
  plt.xlabel('Number of components')
  plt.ylabel('Classification Errors')
  plt.legend()    
  plt.draw()
  raw_input('Press Enter to continue.')



def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------
  # the best MoG:
  #best()  
  # nn model:
  NN()    
  raw_input('Press Enter to finish.')

def best():
  iters=10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  K = 15
  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  n2 = train2.shape[0] 
  n3 = train3.shape[0] 
  
  # TRAINING SET:   
  p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary)    
  p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary)
  px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_train))
  px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_train))  
  p2x = px2 * n2 / (px2 * n2 + px3 * n3)  
  ce_train=((p2x >= 0.5) == target_train).mean()
  
  # VALIDATION SET:     
  px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_valid))
  px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_valid))
  p2x = px2 * n2 / (px2 * n2 + px3 * n3)   
  ce_valid=((p2x >= 0.5) == target_valid).mean()  
  
  
  # TESTING SET:    
  px2 = np.exp(mogLogProb(p2, mu2, vary2, inputs_test))
  px3 = np.exp(mogLogProb(p3, mu3, vary3, inputs_test))     
  p2x = px2 * n2 / (px2 * n2 + px3 * n3) 
  ce_test=((p2x >= 0.5) == target_test).mean() 
  
  print (ce_train,ce_valid,ce_test)  
  raw_input('Press Enter to end MoG.')
  
  
def NN():
  num_hiddens = 15
  eps = 0.1
  momentum = 0.5
  num_epochs = 700
  W1, W2, b1, b2, train_error, valid_error, train_mce, valid_mce = TrainNN(num_hiddens, eps, momentum, num_epochs)
  DisplayErrorPlot(train_error, valid_error, 'Cross Entropy') 
  DisplayErrorPlot(train_mce, valid_mce, 'Mean Classification Error') 
  
  
if __name__ == '__main__':
  #q2()
  #q3()
  #q4()
  q5()

