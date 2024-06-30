# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_train_error_vs_test_error.py`

```
"""
=========================
Train error vs Test error
=========================

Illustration of how the performance of an estimator on unseen data (test data)
is not the same as the performance on training data. As the regularization
increases the performance on train decreases while the performance on test
is optimal within a range of values of the regularization parameter.
The example with an Elastic-Net regression model and the performance is
measured using the explained variance a.k.a. R^2.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate sample data
# --------------------
import numpy as np

from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Define the number of samples for training and testing, and number of features
n_samples_train, n_samples_test, n_features = 75, 150, 500

# Generate synthetic data for regression with specific properties
X, y, coef = make_regression(
    n_samples=n_samples_train + n_samples_test,
    n_features=n_features,
    n_informative=50,
    shuffle=False,
    noise=1.0,
    coef=True,
)

# Split the generated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=n_samples_train, test_size=n_samples_test, shuffle=False
)

# %%
# Compute train and test errors
# -----------------------------
# Define a range of alpha values for regularization
alphas = np.logspace(-5, 1, 60)

# Initialize ElasticNet model with a specific l1_ratio and maximum iterations
enet = linear_model.ElasticNet(l1_ratio=0.7, max_iter=10000)

# Lists to store train and test errors for each alpha
train_errors = list()
test_errors = list()

# Iterate over each alpha, fit the model, and compute scores
for alpha in alphas:
    enet.set_params(alpha=alpha)
    enet.fit(X_train, y_train)
    train_errors.append(enet.score(X_train, y_train))
    test_errors.append(enet.score(X_test, y_test))

# Identify the index of alpha that gives the maximum test error
i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]

# Print the optimal regularization parameter found
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with the optimal regularization parameter
enet.set_params(alpha=alpha_optim)
coef_ = enet.fit(X, y).coef_

# %%
# Plot results functions
# ----------------------

import matplotlib.pyplot as plt

# Create a subplot for plotting train and test errors vs alpha
plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label="Train")
plt.semilogx(alphas, test_errors, label="Test")

# Highlight the optimum on test error plot
plt.vlines(
    alpha_optim,
    plt.ylim()[0],
    np.max(test_errors),
    color="k",
    linewidth=3,
    label="Optimum on test",
)

# Add legends, labels, and adjust plot parameters
plt.legend(loc="lower right")
plt.ylim([0, 1.2])
plt.xlabel("Regularization parameter")
plt.ylabel("Performance")

# Show estimated coef_ vs true coef in another subplot
plt.subplot(2, 1, 2)
plt.plot(coef, label="True coef")
plt.plot(coef_, label="Estimated coef")
plt.legend()

# Adjust subplot parameters and display the plot
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
plt.show()
```