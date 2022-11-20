from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import itertools

class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    # reference https://notebook.community/stellaxux/machine-learning-in-python/ch7/implement_majority_vote    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        
        """ A majority vote ensemble classifier

        Parameters
        ----------
        classifiers : array-like, shape = [n_classifiers]
          Different classifiers for the ensemble

        vote : str, {'classlabel', 'probability'} (default='label')
          If 'classlabel' the prediction is based on the argmax of
            class labels. Else if 'probability', the argmax of
            the sum of probabilities is used to predict the class label
            (recommended for calibrated classifiers).

        weights : array-like, shape = [n_classifiers], optional (default=None)
          If a list of `int` or `float` values are provided, the classifiers
          are weighted by importance; Uses uniform weights if `weights=None`.

        """
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        
#setup models 
clf1 = LogisticRegression(penalty='l2', 
                          C=0.001,
                          random_state=0)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

#import in data
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
    
# padding
lst = data
pad = len(max(lst, key=len))
X = np.array([i + [0]*(pad-len(i)) for i in lst])

# train, CV test, score for all classes
accuracy_mean = []
accuracy_std = []

#clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels = ['Majority Voting']
#all_clf = [pipe1, clf2, pipe3, mv_clf]
all_clf = [mv_clf]

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