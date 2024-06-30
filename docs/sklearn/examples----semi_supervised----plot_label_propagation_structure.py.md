# `D:\src\scipysrc\scikit-learn\examples\semi_supervised\plot_label_propagation_structure.py`

```
"""
==============================================
Label Propagation learning a complex structure
==============================================

Example of LabelPropagation learning a complex internal structure
to demonstrate "manifold learning". The outer circle should be
labeled "red" and the inner circle "blue". Because both label groups
lie inside their own distinct shape, we can see that the labels
propagate correctly around the circle.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# We generate a dataset with two concentric circles. In addition, a label
# is associated with each sample of the dataset that is: 0 (belonging to
# the outer circle), 1 (belonging to the inner circle), and -1 (unknown).
# Here, all labels but two are tagged as unknown.

import numpy as np

from sklearn.datasets import make_circles

# Number of samples to generate
n_samples = 200

# Generate the dataset of two concentric circles
X, y = make_circles(n_samples=n_samples, shuffle=False)

# Define labels for the outer and inner circles
outer, inner = 0, 1

# Initialize labels array with all samples marked as unknown (-1)
labels = np.full(n_samples, -1.0)

# Mark the first and last samples as belonging to outer and inner circles, respectively
labels[0] = outer
labels[-1] = inner

# %%
# Plot raw data
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))

# Plot samples labeled as outer circle
plt.scatter(
    X[labels == outer, 0],
    X[labels == outer, 1],
    color="navy",
    marker="s",
    lw=0,
    label="outer labeled",
    s=10,
)

# Plot samples labeled as inner circle
plt.scatter(
    X[labels == inner, 0],
    X[labels == inner, 1],
    color="c",
    marker="s",
    lw=0,
    label="inner labeled",
    s=10,
)

# Plot samples labeled as unknown
plt.scatter(
    X[labels == -1, 0],
    X[labels == -1, 1],
    color="darkorange",
    marker=".",
    label="unlabeled",
)

plt.legend(scatterpoints=1, shadow=False, loc="center")
_ = plt.title("Raw data (2 classes=outer and inner)")

# %%
#
# The aim of :class:`~sklearn.semi_supervised.LabelSpreading` is to associate
# a label to sample where the label is initially unknown.
from sklearn.semi_supervised import LabelSpreading

# Initialize LabelSpreading with parameters
label_spread = LabelSpreading(kernel="knn", alpha=0.8)

# Fit LabelSpreading model to the data
label_spread.fit(X, labels)

# %%
# Now, we can check which labels have been associated with each sample
# when the label was unknown.
output_labels = label_spread.transduction_
output_label_array = np.asarray(output_labels)

# Find indices of samples labeled as outer and inner circles by the model
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]

plt.figure(figsize=(4, 4))

# Plot samples predicted as outer circle
plt.scatter(
    X[outer_numbers, 0],
    X[outer_numbers, 1],
    color="navy",
    marker="s",
    lw=0,
    s=10,
    label="outer learned",
)

# Plot samples predicted as inner circle
plt.scatter(
    X[inner_numbers, 0],
    X[inner_numbers, 1],
    color="c",
    marker="s",
    lw=0,
    s=10,
    label="inner learned",
)

plt.legend(scatterpoints=1, shadow=False, loc="center")
plt.title("Labels learned with Label Spreading (KNN)")
plt.show()
```