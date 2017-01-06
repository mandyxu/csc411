from utils import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import *
from sklearn.grid_search import GridSearchCV
from sklearn import svm

scores = []

# Load test set
test_inputs = LoadTest()
hidden_inputs = LoadTest("H")


for i in range(1):
    # Load training data and validation data
    train_inputs, train_targets, valid_inputs, valid_targets = LoadData()
    
    
    n = 150
    pca = PCA(n_components=n, whiten=True).fit(train_inputs)   
    eigenfaces = pca.components_.reshape((n, 32, 32))
    
    train_pre_inputs = pca.transform(train_inputs)
    valid_pre_inputs = pca.transform(valid_inputs)
    test_pre_inputs = pca.transform(test_inputs)
    hidden_pre_inputs = pca.transform(hidden_inputs)

    
    clf1 = LogisticRegression(random_state=1)
    param_grid = {'C': [2, 1e3, 5e3, 1e4],
                      'gamma': [0.01, 0.04, 0.1], }
        
    clf2 = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    clf3 = RandomForestClassifier(n_estimators=17, bootstrap=False)
    
    c1 = clf1.fit(train_pre_inputs, train_targets.T[0])
    c2 = clf2.fit(train_pre_inputs, train_targets.T[0])
    c3 = clf3.fit(train_pre_inputs, train_targets.T[0])
    
    v1 = c1.score(valid_pre_inputs, valid_targets.T[0]) 
    v2 = c2.score(valid_pre_inputs, valid_targets.T[0]) 
    v3 = c3.score(valid_pre_inputs, valid_targets.T[0]) 
    
    print v1,v2,v3
    
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')    
    eclf1 = eclf1.fit(train_pre_inputs, train_targets.T[0])
    score1 = eclf1.score(valid_pre_inputs, valid_targets.T[0]) 
    print score1
    
    eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf2 = eclf2.fit(train_pre_inputs, train_targets.T[0])
    score2 = eclf2.score(valid_pre_inputs, valid_targets.T[0]) 
    print score2   
    
    eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,2,1])    
    eclf3 = eclf3.fit(train_pre_inputs, train_targets.T[0])
    score3 = eclf3.score(valid_pre_inputs, valid_targets.T[0]) 
    print score3  
    
    
print mean(scores), scores