# --------------------------------------------------------------------------------------------------
# Custom implementation of Adaboost algorithm
# - inc. support for interchangable weighted weak learners
#
# --------------------------------------------------------------------------------------------------
import numpy as np

ALPHA_CLIP_VALUE = 25

class AdaBoost:
    def __init__(self, weak_classifier_strategy):
        self.weak_classifier_strategy = weak_classifier_strategy

    def train(self, X, y, T):
        classifiers = []
        alphas = []
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        epsilon = 1e-10  # small constant to prevent division by zero

        for t in range(T):
            clf = self.weak_classifier_strategy.create_weak_classifier(X, y, sample_weights)
            classifiers.append(clf)

            # predict and calculate error
            y_pred = clf.predict(X)
            incorrect = (y_pred != y).astype(float)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # calculate alpha
            if error == 0:
                alpha = ALPHA_CLIP_VALUE
            elif error == 1:
                alpha = -ALPHA_CLIP_VALUE
            else:
                alpha = 0.5 * np.log((1 - error) / error)
            alphas.append(alpha)

            # update sample weights
            adjustments = np.where(
                incorrect,
                1 / (2 * (error + epsilon)),
                1 / (2 * (1 - error + epsilon))
            )
            sample_weights *= adjustments
            sample_weights /= np.sum(sample_weights)

        return classifiers, alphas

    def predict(self, X, classifiers, alphas):     
        # weighted sum from all weak classifiers
        adaboost_classifier_prediction = np.array([
            alpha * clf.predict(X) for clf, alpha in zip(classifiers, alphas)
        ])
        predicted_value = np.sum(adaboost_classifier_prediction, axis=0)
        return np.sign(predicted_value), predicted_value
