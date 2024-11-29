import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

weak_classifier = DecisionTreeClassifier(max_depth=1)


class CustomWeakClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.threshold = None
        self.n = None
        self.classes_ = None
        self.error = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        best_threshold = None
        min_error = float("inf")

        # Project features
        X_orig, Xproj, n = getProjectedFeaturesOnOV(X, y, sample_weight)
        self.n = n
        thresholds = np.unique(Xproj)
        thresholds = np.sort(thresholds)
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2  # Midpoints

        for threshold in thresholds:
            pred = np.ones(n_samples)
            pred[Xproj[:, 0] < threshold] = -1  # Use one-dimensional array comparison

            error = np.sum(sample_weight * (pred != y))
            if error < min_error:
                min_error = error
                best_threshold = threshold

        self.threshold = best_threshold
        self.error = min_error
        return self

    def predict_Old(self, X):
        if self.threshold is None or self.n is None:
            raise ValueError("Classifier is not fitted yet.")
        
        projections = np.dot(X, self.n)
        pred = np.ones(X.shape[0])
        pred[projections[:, 0] < self.threshold] = -1
        return pred

    @staticmethod
    def create_weak_classifier(X, y, sample_weights):
        # Create a new instance of the classifier each time
        clf = CustomWeakClassifier()
        clf.fit(X, y, sample_weight=sample_weights)
        return clf
    
    def predict(self, X):
        if self.threshold is None or self.n is None:
            raise ValueError("Classifier is not fitted yet.")
        
        projections = np.dot(X, self.n)  # Projection onto orientation vector
        pred = np.ones(X.shape[0])
        pred[projections < self.threshold] = -1  # Use threshold for classification
        return pred


def getProjectedFeaturesOnOV_Old(X, y, weights):
    pos_idx = y == 1
    neg_idx = y == -1

    # Weighted means
    u_pos = np.sum(weights[pos_idx] * X[pos_idx, 0]) / np.sum(weights[pos_idx])
    u_neg = np.sum(weights[neg_idx] * X[neg_idx, 0]) / np.sum(weights[neg_idx])

    # Calculate orientation vector and normalize it
    r = u_pos - u_neg
    r_magnitude = np.linalg.norm(r)
    n = r / r_magnitude if r_magnitude != 0 else np.zeros_like(r)
    
    # Project features onto orientation vector
    projections = np.dot(X, n)
    return X, projections, n

def getProjectedFeaturesOnOV(X, y, weights):
    pos_idx = y == 1
    neg_idx = y == -1

    # Weighted means for all feature dimensions
    u_pos = np.sum(weights[pos_idx, None] * X[pos_idx], axis=0) / np.sum(weights[pos_idx])
    u_neg = np.sum(weights[neg_idx, None] * X[neg_idx], axis=0) / np.sum(weights[neg_idx])

    # Orientation vector
    r = u_pos - u_neg
    r_magnitude = np.linalg.norm(r)
    n = r / r_magnitude if r_magnitude != 0 else np.zeros_like(r)

    # Project features onto the orientation vector
    projections = np.dot(X, n)
    return X, projections, n
