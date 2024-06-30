# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ridge_path.py`

```
"""
===========================================================
Plot Ridge coefficients as a function of the regularization
===========================================================

Shows the effect of collinearity in the coefficients of an estimator.

.. currentmodule:: sklearn.linear_model

:class:`Ridge` Regression is the estimator used in this example.
Each color represents a different feature of the
coefficient vector, and this is displayed as a function of the
regularization parameter.

This example also shows the usefulness of applying Ridge regression
to highly ill-conditioned matrices. For such matrices, a slight
change in the target variable can cause huge variances in the
calculated weights. In such cases, it is useful to set a certain
regularization (alpha) to reduce this variation (noise).

When alpha is very large, the regularization effect dominates the
squared loss function and the coefficients tend to zero.
At the end of the path, as alpha tends toward zero
and the solution tends towards the ordinary least squares, coefficients
exhibit big oscillations. In practice it is necessary to tune alpha
in such a way that a balance is maintained between both.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

# X is the 10x10 Hilbert matrix
# 创建一个10x10的希尔伯特矩阵作为特征矩阵X
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# 目标变量y为10个单位向量
y = np.ones(10)

# %%
# Compute paths
# -------------
# 计算不同alpha取值下的岭回归系数路径

n_alphas = 200
# 在对数尺度上均匀地生成200个alpha值，范围从10的-10次方到10的-2次方
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
# 遍历每个alpha值
for a in alphas:
    # 创建一个岭回归模型，设置alpha值，并且不拟合截距项
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    # 在特征矩阵X和目标变量y上拟合岭回归模型
    ridge.fit(X, y)
    # 将每个alpha值对应的系数向量存储起来
    coefs.append(ridge.coef_)

# %%
# Display results
# ---------------
# 显示结果，绘制alpha值与系数的关系图

ax = plt.gca()

# 绘制不同alpha值下的系数路径
ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis，反转x轴
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()
```