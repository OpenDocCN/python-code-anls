# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_adaboost_regression.py`

```
# %%
# Preparing the data
# ------------------
# 首先，我们准备具有正弦关系和一些高斯噪声的虚拟数据集。

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

# %%
# Training and prediction with DecisionTree and AdaBoost Regressors
# -----------------------------------------------------------------
# 现在，我们定义分类器并将其拟合到数据集。
# 然后我们对相同的数据进行预测，以查看它们拟合数据的效果。
# 第一个回归器是一个 `DecisionTreeRegressor`，使用 `max_depth=4`。
# 第二个回归器是一个 `AdaBoostRegressor`，使用 `max_depth=4` 的 `DecisionTreeRegressor`
# 作为基学习器，并且将由 `n_estimators=300` 个这样的基学习器组成。

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# %%
# Plotting the results
# --------------------
# 最后，我们绘制两个回归器（单一决策树回归器和 AdaBoost 回归器）拟合数据的效果。

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("colorblind")

plt.figure()
plt.scatter(X, y, color=colors[0], label="training samples")
plt.plot(X, y_1, color=colors[1], label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, color=colors[2], label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
```