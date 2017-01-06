from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import *
from preprocess import *

def TrainModel(clf, train_inputs, train_targets, valid_inputs, valid_targets):
    """
    clf: model used in training.
    train_inputs: array like [n_examples, n_features]
    train_targets: array like [n_examples, 1]
    Returns a trained model clf, classification rate on validation set, and prediction of the validation set [n_examples,].
    """    
    clf = clf.fit(train_inputs, train_targets.T[0])
    predict = clf.predict(valid_inputs)
    score_valid = clf.score(valid_inputs, valid_targets.T[0]) 
    return clf, score_valid, predict    

def Predict(clf, inputs):
    """
    Returns the prediction [n_examples, 1] of inputs by model clf. 
    """
    predict = clf.predict(inputs)
    return np.reshape(predict_valid,(predict_valid.shape[0],1))

def TrainAdaBoost(train_inputs, train_targets, valid_inputs, valid_targets):
    clf = AdaBoostClassifier()
    return TrainModel(clf, train_inputs, train_targets, valid_inputs, valid_targets)

def TrainDecisionTree(train_inputs, train_targets, valid_inputs, valid_targets):
    clf = DecisionTreeClassifier()
    return TrainModel(clf, train_inputs, train_targets, valid_inputs, valid_targets)

def TrainRandomForests(train_inputs, train_targets, valid_inputs, valid_targets):
    clf = RandomForestClassifier(n_estimators=17, bootstrap=False)
    return TrainModel(clf, train_inputs, train_targets, valid_inputs, valid_targets)

def TrainSVM(train_inputs, train_targets, valid_inputs, valid_targets):
    #clf = svm.SVC(kernel='rbf', gamma=0.04, C=2)
    param_grid = {'C': [2, 1e3, 5e3, 1e4],
                      'gamma': [0.01, 0.04, 0.1], }
    clf1 = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)    
    return TrainModel(clf1, train_inputs, train_targets, valid_inputs, valid_targets)
    


# loading test data
test_inputs = LoadTest()

for i in range(10):
    
    # loading data
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
    
    # preprocessing
    train_inputs_pca, valid_inputs_pca, test_inputs_pca, eigenfaces = PCA(train_inputs, valid_inputs, test_inputs)
    
    train_inputs_nb, valid_inputs_nb, test_inputs_nb = NB(train_inputs, valid_inputs, test_inputs)
    
    # fitting models
    model, score_valid, predict_valid = TrainAdaBoost(train_inputs, train_targets, valid_inputs, valid_targets)
    
    model, score_valid, predict_valid = TrainSVM(train_inputs, train_targets, valid_inputs, valid_targets)
    
    W1, W2, b1, b2, target_train, train_predicted, target_valid, valid_predicted, train_nn_nb, valid_nn_nb = TrainNN(train_pre_inputs_nb, train_targets, valid_pre_inputs_nb, valid_targets, num_hiddens, eps, momentum, num_epochs)
    
    
    print score_valid


