from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import *
from utils import *



def TrainRandomForests(train_inputs, train_targets, valid_inputs, valid_targets):
    """
    train_inputs: array like [n_examples, n_features]
    train_targets: array like [n_examples, 1]
    Returns a trained model clf, prediction of validation set.
    """
    clf = RandomForestClassifier(n_estimators=17, bootstrap=False)
    clf = clf.fit(train_inputs, train_targets.T[0])
    predict = clf.predict(valid_inputs)
    score_valid = clf.score(valid_inputs, valid_targets.T[0]) 
    return score_valid, predict

def PredRandomForest(clf, inputs):
    predict = clf.predict(inputs)
    return np.reshape(predict,(predict.shape[0],1))
    
scores = []
for i in range(10):
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
    mce, predict = TrainRandomForests(train_inputs, train_targets, valid_inputs, valid_targets)
    print mce
    
    scores.append(mce)    
    
print mean(scores), scores