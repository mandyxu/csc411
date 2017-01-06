from sklearn.ensemble import AdaBoostClassifier
from utils import *
from sklearn.decomposition import *

def TrainAdaBoost(train_inputs, train_targets, valid_inputs, valid_targets):
    """
    train_inputs: array like [n_examples, n_features]
    train_targets: array like [n_examples, 1]
    Returns a trained model clf, prediction of validation set [n_examples,].
    """    
    clf = AdaBoostClassifier()
    clf = clf.fit(train_inputs, train_targets.T[0])
    predict = clf.predict(valid_inputs)
    score_valid = clf.score(valid_inputs, valid_targets.T[0]) 
    return score_valid, predict 
    
def PredAdaBoost(clf, inputs):
    """
    Returns the prediction [n_examples, 1] of inputs by model clf. 
    """
    predict = clf.predict(inputs)
    return np.reshape(predict_valid,(predict_valid.shape[0],1))

for i in range(10):
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
    n = 150
    pca = PCA(n_components=n, whiten=True).fit(train_inputs)   
    eigenfaces = pca.components_.reshape((n, 32, 32))
    
    train_pre_inputs = pca.transform(train_inputs)
    valid_pre_inputs = pca.transform(valid_inputs)
    #test_pre_inputs = pca.transform(test_inputs)
    score_valid, predict_valid = TrainAdaBoost(train_pre_inputs, train_targets, valid_pre_inputs, valid_targets)
    print score_valid