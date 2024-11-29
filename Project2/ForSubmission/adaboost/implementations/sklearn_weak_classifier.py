# --------------------------------------------------------------------------------------------------
# Custom Weighted Weak Linear Classifier
#
#
#
# --------------------------------------------------------------------------------------------------
from common.weak_classifier_base import WeakClassifierBase
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator 

class SklearnWeakClassifierWrapper(WeakClassifierBase):
    def __init__(self, sklearn_classifier: BaseEstimator):
        self.sklearn_classifier = sklearn_classifier

    def predict(self, X, y=None):
        return self.sklearn_classifier.predict(X)

class SklearnWeakClassifier():
    def create_weak_classifier(self, X, y, sample_weights):
        # sklearn's implementation based on decision stump
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y, sample_weight=sample_weights)
        return clf
