# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_regularization.py`

```
"""
================================
Gradient Boosting regularization
================================

Illustration of the effect of different regularization strategies
for Gradient Boosting. The example is taken from Hastie et al 2009 [1]_.

The loss function used is binomial deviance. Regularization via
shrinkage (`learning_rate < 1.0`) improves performance considerably.
In combination with shrinkage, stochastic gradient boosting
(`subsample < 1.0`) can produce more accurate models by reducing the
variance via bagging.
Subsampling without shrinkage usually does poorly.
Another strategy to reduce the variance is by subsampling the features
analogous to the random splits in Random Forests
(via the `max_features` parameter).

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
    Learning Ed. 2", Springer, 2009.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn import datasets, ensemble  # 导入scikit-learn中的数据集和集成方法
from sklearn.metrics import log_loss  # 导入log_loss评估指标
from sklearn.model_selection import train_test_split  # 导入数据集划分工具

X, y = datasets.make_hastie_10_2(n_samples=4000, random_state=1)

# map labels from {-1, 1} to {0, 1}
labels, y = np.unique(y, return_inverse=True)  # 将标签转换为0和1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)  # 划分训练集和测试集

original_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
}

plt.figure()  # 创建一个新的图形窗口

for label, color, setting in [
    ("No shrinkage", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
    ("learning_rate=0.2", "turquoise", {"learning_rate": 0.2, "subsample": 1.0}),
    ("subsample=0.5", "blue", {"learning_rate": 1.0, "subsample": 0.5}),
    (
        "learning_rate=0.2, subsample=0.5",
        "gray",
        {"learning_rate": 0.2, "subsample": 0.5},
    ),
    (
        "learning_rate=0.2, max_features=2",
        "magenta",
        {"learning_rate": 0.2, "max_features": 2},
    ),
]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)  # 创建梯度提升分类器对象
    clf.fit(X_train, y_train)  # 在训练集上拟合模型

    # compute test set deviance
    test_deviance = np.zeros((params["n_estimators"],), dtype=np.float64)  # 初始化测试集偏差数组

    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
        test_deviance[i] = 2 * log_loss(y_test, y_proba[:, 1])  # 计算测试集偏差

    plt.plot(
        (np.arange(test_deviance.shape[0]) + 1)[::5],  # 绘制横坐标，每5个迭代取一个值
        test_deviance[::5],  # 绘制纵坐标，每5个迭代取一个值
        "-",  # 使用实线连接数据点
        color=color,  # 设置线条颜色
        label=label,  # 设置图例标签
    )

plt.legend(loc="upper right")  # 显示图例，位置在右上角
plt.xlabel("Boosting Iterations")  # 设置横坐标标签
plt.ylabel("Test Set Deviance")  # 设置纵坐标标签

plt.show()  # 显示绘制的图形
```