import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
import json
import itertools
from sklearn import preprocessing

clf1 = LogisticRegression(penalty='l2', 
                          C=0.001,
                          random_state=0)
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])

#import data
fileObject = open("testing_data.json", "r")
jsonContent = fileObject.read()
aList = json.loads(jsonContent)
fileObject.close()

fileObject = open("training_data.json", "r")
jsonContent = fileObject.read()
bList = json.loads(jsonContent)
fileObject.close()

# combine train and test datasets - using CV later
ids = []
data = []
labels = []

for i in range(len(aList['testing_data'])):
#for i in range(5):
    ids.append(aList['testing_data'][i]['id'])
    datai = list(itertools.chain.from_iterable(aList['testing_data'][i]['data']))
    data.append(datai)
    labels.append(list(aList['testing_data'][i]['labels'].values()))
    
for i in range(len(bList['training_data'])):
    ids.append(bList['training_data'][i]['id'])
    datai = list(itertools.chain.from_iterable(bList['training_data'][i]['data']))
    data.append(datai)
    labels.append(list(bList['training_data'][i]['labels'].values()))
    
lst = data
pad = len(max(lst, key=len))
X = np.array([i + [0]*(pad-len(i)) for i in lst])  

# train, CV test, score for all classifiers and labels
accuracy_mean = []
accuracy_std = []

#clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
clf_labels = ['Logistic Regression']

#all_clf = [pipe1, clf2, pipe3, mv_clf]
all_clf = [pipe1]

scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X)

for i in range(len(np.array(labels[0]))):
    print("iteration: ", i)
    
    # iterate through each label column
    y_train = np.array(labels)[:,i]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
        accuracy_mean.append(scores.mean())
        accuracy_std.append(scores.std())