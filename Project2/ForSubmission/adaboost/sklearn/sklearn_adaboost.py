# --------------------------------------------------------------------------------------------------
# Sklearn implementation of Adaboost algorithm
# - inc. support for interchangable weighted weak learners
#
# Used as an aid during development of deliverable components
# --------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from common.algorithm import AdaBoost
import common.plot_library as cpl
from implementations.sklearn_weak_classifier import SklearnWeakClassifier
from implementations.custom_weak_classifier import CustomWeakClassifier


# Load data
train_file = './adaboost-train-24.txt'
test_file = './adaboost-test-24.txt'
column_names = ['X1', 'X2', 'Y']
df_train = pd.read_csv(train_file, sep=r'\s+', header=None, names=column_names)
df_test = pd.read_csv(test_file, sep=r'\s+', header=None, names=column_names)

X_train = df_train[['X1', 'X2']].to_numpy()
Y_train = df_train['Y'].to_numpy()
X_test = df_test[['X1', 'X2']].to_numpy()
Y_test = df_test['Y'].to_numpy()


# Number of classifiers we want
num_classifiers = 50


# A: Weighted weak linear classifier (wwlc). Choose from:
# wwlc = CustomWeakClassifier()
# wwlc = DecisionTreeClassifier(max_depth=1)
wwlc = CustomWeakClassifier()

# B: AdaBoost algorithm 
ada_boost = AdaBoostClassifier(estimator=wwlc, 
                               n_estimators=num_classifiers, 
                               algorithm='SAMME', 
                               learning_rate=1.0, 
                               random_state=42)

# Train
ada_boost.fit(X_train, Y_train)

# Prediction
y_pred = ada_boost.predict(X_test)

# Accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print(f"Sklearn Strong Classifier Accuracy: {accuracy:.4f}")
