
import numpy as np
import matplotlib.pyplot as plt
from adaboost import ITERATIONS
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------------------------------
# some globals
# --------------------------------------------------------------------------------------------------
train_acc = []
test_acc = []
xx = []
yy = []
X  = []
Z = []


# --------------------------------------------------------------------------------------------------
# Plot
# Train v Test Accuracy
# --------------------------------------------------------------------------------------------------
def plot_accuracy():
    
    step = 5  # otherwise plot is very noisy
    x_values = range(1, ITERATIONS + 1, step)
    train_acc_downsampled = train_acc[::step]
    test_acc_downsampled = test_acc[::step]

    plt.plot(x_values, train_acc_downsampled, label="Train", linestyle="--", marker="o")
    plt.plot(x_values, test_acc_downsampled, label="Test", linestyle="--", marker="x")
    plt.xlabel("Number of Boosting ITERATIONS (T)")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost Accuracy")
    plt.legend()
    plt.grid()
    

    
# --------------------------------------------------------------------------------------------------
# Plot
# Decision Boundry
# --------------------------------------------------------------------------------------------------
def plot_decision_boundary():
    
    custom_cmap = ListedColormap(['pink', 'lightblue'])
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=custom_cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['magenta', 'blue']))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary of AdaBoost Classifier')
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    
    
# --------------------------------------------------------------------------------------------------
# Plot
# Contour
# --------------------------------------------------------------------------------------------------
def plot_contour():
    
    custom_cmap = ListedColormap(['pink', 'lightblue'])
    plt.contourf(xx, yy, Z_values, alpha=0.8, cmap=custom_cmap)
    plt.contour(xx, yy, Z_values, colors='k', levels=[0.5], linewidths=2)

    # Scatter plot of the training data
    #plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['magenta', 'blue']))

    # Plot aesthetics
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('AdaBoost Classifier Predicted Values Contour')
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.grid(True)
    plt.show()
    
    



# --------------------------------------------------------------------------------------------------
# Plot
# main function that calls the individual plot functions
# --------------------------------------------------------------------------------------------------     
def generate_plots():
    # Create a new figure for combined plots
    plt.figure(figsize=(20, 8))  # Wide figure for two subplots
    
    # Call plot_accuracy in the first subplot
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st plot
    plot_accuracy()  # plot_accuracy draws on the current subplot
    
    # Call plot_decision_boundary in the second subplot
    plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd plot
    plot_decision_boundary()  # plot_decision_boundary draws on the current subplot
    
    # Call plot_contour in the third subplot
    plt.subplot(1, 3, 3)  
    plot_contour() 
    
    # Display all plots together
    plt.tight_layout()
    plt.show()
   