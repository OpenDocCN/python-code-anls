# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_forest_iris.py`

```
"""
====================================================================
Plot the decision surfaces of ensembles of trees on the iris dataset
====================================================================

Plot the decision surfaces of forests of randomized trees trained on pairs of
features of the iris dataset.

This plot compares the decision surfaces learned by a decision tree classifier
(first column), by a random forest classifier (second column), by an extra-
trees classifier (third column) and by an AdaBoost classifier (fourth column).

In the first row, the classifiers are built using the sepal width and
the sepal length features only, on the second row using the petal length and
sepal length only, and on the third row using the petal width and the
petal length only.

In descending order of quality, when trained (outside of this example) on all
4 features using 30 estimators and scored using 10 fold cross validation,
we see::

    ExtraTreesClassifier()  # 0.95 score
    RandomForestClassifier()  # 0.94 score
    AdaBoost(DecisionTree(max_depth=3))  # 0.94 score
    DecisionTree(max_depth=None)  # 0.94 score

Increasing `max_depth` for AdaBoost lowers the standard deviation of
the scores (but the average score does not improve).

See the console's output for further details about each model.

In this example you might try to:

1) vary the ``max_depth`` for the ``DecisionTreeClassifier`` and
   ``AdaBoostClassifier``, perhaps try ``max_depth=3`` for the
   ``DecisionTreeClassifier`` or ``max_depth=None`` for ``AdaBoostClassifier``
2) vary ``n_estimators``

It is worth noting that RandomForests and ExtraTrees can be fitted in parallel
on many cores as each tree is built independently of the others. AdaBoost's
samples are built sequentially and so do not use multiple cores.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
iris = load_iris()

plot_idx = 1

# Define the models to be compared
models = [
    DecisionTreeClassifier(max_depth=None),  # Decision Tree classifier with default max depth
    RandomForestClassifier(n_estimators=n_estimators),  # Random Forest classifier with specified number of estimators
    ExtraTreesClassifier(n_estimators=n_estimators),  # Extra Trees classifier with specified number of estimators
    AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),  # AdaBoost using Decision Tree base estimator with max depth 3
        n_estimators=n_estimators,
        algorithm="SAMME",
    ),
]

# Plotting loop for different pairs of features
for pair in ([0, 1], [0, 2], [2, 3]):
    plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
    plt.axis("tight")
    plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
    plt.show()  # Show the plot
```