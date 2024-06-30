# `D:\src\scipysrc\scikit-learn\examples\semi_supervised\plot_semi_supervised_versus_svm_iris.py`

```
"""
===============================================================================
Decision boundary of semi-supervised classifiers versus SVM on the Iris dataset
===============================================================================

A comparison for the decision boundaries generated on the iris dataset
by Label Spreading, Self-training and SVM.

This example demonstrates that Label Spreading and Self-training can learn
good boundaries even when small amounts of labeled data are available.

Note that Self-training with 100% of the data is omitted as it is functionally
identical to training the SVC on 100% of the data.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC

# Load the Iris dataset
iris = datasets.load_iris()

# Extract only the first two features for visualization
X = iris.data[:, :2]
y = iris.target

# Step size for the mesh grid
h = 0.02

# Initialize a random number generator
rng = np.random.RandomState(0)

# Generate random numbers to mark samples as unlabeled
y_rand = rng.rand(y.shape[0])

# Create versions of y with 30% and 50% of data labeled as -1 (unlabeled)
y_30 = np.copy(y)
y_30[y_rand < 0.3] = -1
y_50 = np.copy(y)
y_50[y_rand < 0.5] = -1

# Create instances of Label Spreading with different percentages of labeled data
ls30 = (LabelSpreading().fit(X, y_30), y_30, "Label Spreading 30% data")
ls50 = (LabelSpreading().fit(X, y_50), y_50, "Label Spreading 50% data")
ls100 = (LabelSpreading().fit(X, y), y, "Label Spreading 100% data")

# The base classifier for Self-training is an SVM with radial basis function kernel
base_classifier = SVC(kernel="rbf", gamma=0.5, probability=True)

# Create instances of Self-training Classifier with different percentages of labeled data
st30 = (SelfTrainingClassifier(base_classifier).fit(X, y_30), y_30, "Self-training 30% data")
st50 = (SelfTrainingClassifier(base_classifier).fit(X, y_50), y_50, "Self-training 50% data")

# Create an SVM classifier with radial basis function kernel
rbf_svc = (SVC(kernel="rbf", gamma=0.5).fit(X, y), y, "SVC with rbf kernel")

# Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Define a color map for different classes
color_map = {-1: (1, 1, 1), 0: (0, 0, 0.9), 1: (1, 0, 0), 2: (0.8, 0.6, 0)}

# List of classifiers to plot
classifiers = (ls30, st30, ls50, st50, ls100, rbf_svc)

# Iterate over classifiers to plot decision boundaries and data points
for i, (clf, y_train, title) in enumerate(classifiers):
    # Plot each classifier in a subplot
    plt.subplot(3, 2, i + 1)
    
    # Predict the labels for mesh grid points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape predictions into the mesh grid shape
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries as filled contours
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("off")  # Turn off axis labels and ticks

    # Plot the training points colored by their true class labels
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="black")
    
    # Set title for the subplot
    plt.title(title)

# Overall title for the entire plot
plt.suptitle("Unlabeled points are colored white", y=0.1)

# Display the plot
plt.show()
```