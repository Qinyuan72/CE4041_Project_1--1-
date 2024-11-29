# --------------------------------------------------------------------------------------------------
# Main program
#
# 27/11/24
# Ver 1.0  Project submission Nov 2024
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# library imports
# --------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from common.algorithm import AdaBoost
import common.plot_library as cpl
from implementations.sklearn_weak_classifier import SklearnWeakClassifier
from implementations.custom_weak_classifier import CustomWeakClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime


# --------------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------------
ITERATIONS = 50

# --------------------------------------------------------------------------------------------------
# Load data into numpy arrays
# --------------------------------------------------------------------------------------------------
train_file = './adaboost-train-24.txt'
test_file = './adaboost-test-24.txt'
column_names = ['X1', 'X2', 'Y']
df_train = pd.read_csv(train_file, sep=r'\s+', header=None, names=column_names)
df_test = pd.read_csv(test_file, sep=r'\s+', header=None, names=column_names)
X_train = df_train[['X1', 'X2']].to_numpy()
Y_train = df_train['Y'].to_numpy()
X_test = df_test[['X1', 'X2']].to_numpy()
Y_test = df_test['Y'].to_numpy()
#print(df_test.describe())

# -----------------------------------------------------------------------------------------------------
# Main 
# -----------------------------------------------------------------------------------------------------
def main():

    # 1 - TEST ACCURACY AFTER N ITERATIONS - Ntest = 26
    Ntest=26
    weak_classifier_ = CustomWeakClassifier() # simlply replace CustomWeakClassifier() with SklearnWeakClassifier() if using sklearn implementation
    adaboost_strong_classifier_ = AdaBoost(weak_classifier_strategy=weak_classifier_)    
    classifiers_, alphas_ = adaboost_strong_classifier_.train(X_train, Y_train, Ntest)
    y_test_pred, y_test_pred_value = adaboost_strong_classifier_.predict(X_test, classifiers_[:Ntest], alphas_[:Ntest])
    accuracy = accuracy_score(Y_test, y_test_pred)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Step 1 - Strong Classifier Test Accuracy: {accuracy:.2%}")
    
    # 2 - MEASURE TRAIN & TEST ACCURACY OVER Ntest plus ITERATIONS 
    cpl.train_acc, cpl.test_acc = measure_accuracy_of_predictions(adaboost_strong_classifier_, classifiers_, alphas_, ITERATIONS)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Step 2 - Accuracy: Train {cpl.train_acc[-1]:.2%}, Test {cpl.test_acc[-1]:.2%}")
    
    # 3 - GENERATE DECISION BOUNDRY DATA - using training data
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Step 3 - Generate Decision Boundry Data")
    cpl.X, cpl.y, cpl.Z, cpl.xx, cpl.yy = generate_decision_boundry_data(X_train, Y_train, adaboost_strong_classifier_, classifiers_, alphas_, Ntest)
    
    # 4 - GENERATE CONTOUR DATA - using test data
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Step 4 - Generate Contour Data")
    cpl.Z_values, cpl.xx, cpl.yy = generate_contour_data(X_train, adaboost_strong_classifier_, classifiers_, alphas_, Ntest)
       
    # 5 - PLOTS 
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Step 5 - Plots")
    cpl.generate_plots()
 


# -----------------------------------------------------------------------------------------------------
# measure_accuracy_of_predictions 
# -----------------------------------------------------------------------------------------------------
def measure_accuracy_of_predictions(adaboost_classifier, classifiers, alphas, iterations):
    train_acc = []
    test_acc = []
    
    for t in range(1, iterations + 1):
        y_train_pred, not_used = adaboost_classifier.predict(X_train, classifiers[:t], alphas[:t])
        y_test_pred, not_used = adaboost_classifier.predict(X_test, classifiers[:t], alphas[:t])
        # train_acc, test_acc store the accuracy data for later plotting
        train_acc.append(accuracy_score(Y_train, y_train_pred))
        test_acc.append(accuracy_score(Y_test, y_test_pred))
    return train_acc, test_acc


# -----------------------------------------------------------------------------------------------------
# generate_decision_boundry_data 
# -----------------------------------------------------------------------------------------------------
def generate_decision_boundry_data(X, y, adaboost_classifier, classifiers, alphas, N_value):
    resolution=0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # create a mesh grid covering the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))   
    # flatten the grid and predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z, not_used = zip(*[
            adaboost_classifier.predict(point.reshape(1, -1), classifiers[:N_value], alphas[:N_value])
            for point in grid_points
        ])
    # Z contains the predicted class for train data
    Z = np.array(Z).reshape(xx.shape)  

    return X, y, Z, xx, yy
    
# ---------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------
# generate_contour_data 
# -----------------------------------------------------------------------------------------------------
def generate_contour_data(X, adaboost_classifier, classifiers, alphas, N_value):
    # generate grid using mgrid
    resolution = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))  

    # flatten the grid and predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    not_used, Z_values = zip(*[
        adaboost_classifier.predict(point.reshape(1, -1), classifiers[:N_value], alphas[:N_value])
        for point in grid_points
    ])
    # Z_values contains the predicted values for train data
    Z_values = np.array(Z_values).reshape(xx.shape)
    
    return Z_values, xx, yy

# ---------------------------------------------------------------------------------------

# make main run first
if __name__ == "__main__":
    main()
    
