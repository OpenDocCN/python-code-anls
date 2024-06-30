# `D:\src\scipysrc\scikit-learn\examples\datasets\plot_random_multilabel_dataset.py`

```
"""
==============================================
Plot randomly generated multilabel dataset
==============================================

This illustrates the :func:`~sklearn.datasets.make_multilabel_classification`
dataset generator. Each sample consists of counts of two features (up to 50 in
total), which are differently distributed in each of two classes.

Points are labeled as follows, where Y means the class is present:

    =====  =====  =====  ======
      1      2      3    Color
    =====  =====  =====  ======
      Y      N      N    Red
      N      Y      N    Blue
      N      N      Y    Yellow
      Y      Y      N    Purple
      Y      N      Y    Orange
      Y      Y      N    Green
      Y      Y      Y    Brown
    =====  =====  =====  ======

A star marks the expected sample for each class; its size reflects the
probability of selecting that class label.

The left and right examples highlight the ``n_labels`` parameter:
more of the samples in the right plot have 2 or 3 labels.

Note that this two-dimensional example is very degenerate:
generally the number of features would be much greater than the
"document length", while here we have much larger documents than vocabulary.
Similarly, with ``n_classes > n_features``, it is much less likely that a
feature distinguishes a particular class.

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_multilabel_classification as make_ml_clf

COLORS = np.array(
    [
        "!",
        "#FF3333",  # red
        "#0198E1",  # blue
        "#BF5FFF",  # purple
        "#FCD116",  # yellow
        "#FF7216",  # orange
        "#4DBD33",  # green
        "#87421F",  # brown
    ]
)

# Use same random seed for multiple calls to make_multilabel_classification to
# ensure same distributions
RANDOM_SEED = np.random.randint(2**10)


def plot_2d(ax, n_labels=1, n_classes=3, length=50):
    """
    Generate and plot a 2D multilabel dataset.

    Parameters:
    - ax: matplotlib axis object to draw the plot on
    - n_labels: number of labels per instance (default is 1)
    - n_classes: number of classes (default is 3)
    - length: total number of features (default is 50)

    Returns:
    - p_c: class probabilities
    - p_w_c: feature probabilities given classes
    """
    X, Y, p_c, p_w_c = make_ml_clf(
        n_samples=150,
        n_features=2,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        allow_unlabeled=False,
        return_distributions=True,
        random_state=RANDOM_SEED,
    )

    # Scatter plot of data points colored by class
    ax.scatter(
        X[:, 0], X[:, 1], color=COLORS.take((Y * [1, 2, 4]).sum(axis=1)), marker="."
    )

    # Scatter plot of expected samples marked with stars
    ax.scatter(
        p_w_c[0] * length,
        p_w_c[1] * length,
        marker="*",
        linewidth=0.5,
        edgecolor="black",
        s=20 + 1500 * p_c**2,
        color=COLORS.take([1, 2, 4]),
    )

    # Set labels for axes
    ax.set_xlabel("Feature 0 count")
    ax.set_ylabel("Feature 1 count")

    return p_c, p_w_c


# Create a 1x2 subplot grid for visualization
_, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
plt.subplots_adjust(bottom=0.15)

# Plot first subplot with n_labels=1
p_c, p_w_c = plot_2d(ax1, n_labels=1)
ax1.set_title("n_labels=1, length=50")

# Plot second subplot with n_labels=3
plot_2d(ax2, n_labels=3)
ax2.set_title("n_labels=3, length=50")
ax2.set_xlim(left=0, auto=True)
ax2.set_ylim(bottom=0, auto=True)

# Show the plots
plt.show()

# Print the random seed used for data generation
print("The data was generated from (random_state=%d):" % RANDOM_SEED)
# 打印表头，包括类别、类别的概率、以及各类别下每个单词的条件概率
print("Class", "P(C)", "P(w0|C)", "P(w1|C)", sep="\t")
# 使用 zip 函数同时迭代类别名称 ["red", "blue", "yellow"]，类别概率列表 p_c，以及条件概率矩阵 p_w_c 的每一行
for k, p, p_w in zip(["red", "blue", "yellow"], p_c, p_w_c.T):
    # 打印每一行的信息，包括类别名称、类别概率、以及每个类别下第一个和第二个单词的条件概率
    print("%s\t%0.2f\t%0.2f\t%0.2f" % (k, p, p_w[0], p_w[1]))
```