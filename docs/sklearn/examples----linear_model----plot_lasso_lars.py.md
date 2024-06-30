# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_lasso_lars.py`

```
"""
=====================
Lasso path using LARS
=====================

Computes Lasso Path along the regularization parameter using the LARS
algorithm on the diabetes dataset. Each color represents a different
feature of the coefficient vector, and this is displayed as a function
of the regularization parameter.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入绘图库 matplotlib 和数值计算库 numpy
import matplotlib.pyplot as plt
import numpy as np

# 导入 scikit-learn 中的数据集和线性模型
from sklearn import datasets, linear_model

# 加载糖尿病数据集，X 是特征矩阵，y 是目标向量
X, y = datasets.load_diabetes(return_X_y=True)

# 打印消息，指示正在使用 LARS 方法计算正则化路径
print("Computing regularization path using the LARS ...")

# 调用 LARS 算法计算 Lasso 路径，获取正则化路径系数
_, _, coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)

# 计算每个特征的绝对值之和，并进行归一化
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

# 绘制 LASSO 路径图
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle="dashed")
plt.xlabel("|coef| / max|coef|")
plt.ylabel("Coefficients")
plt.title("LASSO Path")
plt.axis("tight")
plt.show()
```