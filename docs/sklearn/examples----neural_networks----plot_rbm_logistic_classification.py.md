# `D:\src\scipysrc\scikit-learn\examples\neural_networks\plot_rbm_logistic_classification.py`

```
"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.

import numpy as np
from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

# %%
# Function to nudge dataset
# -------------------------
#
# This function expands the dataset by creating new samples that are shifted
# by 1 pixel in various directions (left, right, down, up).

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    # Generate shifted images and concatenate them to X
    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    # Replicate labels Y accordingly for the expanded dataset
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# %%
# Load and preprocess dataset
# ---------------------------
#
# Load the digits dataset, convert data to float32, expand dataset using nudge_dataset,
# and perform min-max scaling to normalize features between 0 and 1.

X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# Models definition
# -----------------
#
# Define a pipeline for classification using a BernoulliRBM feature extractor
# followed by a LogisticRegression classifier.

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# Training
# --------
#
# Set hyperparameters for RBM model. These parameters were tuned previously
# using grid search but are set directly here for simplicity and efficiency.

from sklearn.base import clone

rbm.learning_rate = 0.06
rbm.n_iter = 10

# More components tend to give better prediction performance, but larger
# fitting time
# 设置 RBM 模型的主成分数量为 100
rbm.n_components = 100
# 设置 Logistic 回归模型的正则化参数 C 为 6000
logistic.C = 6000

# Training RBM-Logistic Pipeline
# 使用训练数据 X_train 和标签 Y_train 训练 RBM-Logistic 管道模型
rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
# 克隆 Logistic 回归模型，并设置其正则化参数 C 为 100.0，然后使用训练数据 X_train 和标签 Y_train 进行训练
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)

# %%
# Evaluation
# ----------

from sklearn import metrics

# 使用 RBM 特征的 Logistic 回归模型对测试数据 X_test 进行预测，并输出分类报告
Y_pred = rbm_features_classifier.predict(X_test)
print(
    "Logistic regression using RBM features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# 使用原始像素特征的 Logistic 回归模型对测试数据 X_test 进行预测，并输出分类报告
Y_pred = raw_pixel_classifier.predict(X_test)
print(
    "Logistic regression using raw pixel features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# The features extracted by the BernoulliRBM help improve the classification
# accuracy with respect to the logistic regression on raw pixels.

# %%
# Plotting
# --------

import matplotlib.pyplot as plt

# 创建一个图像窗口，显示 RBM 提取的 100 个组件的可视化结果
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("100 components extracted by RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
```