# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_lasso_coordinate_descent_path.py`

```
"""
=====================
Lasso and Elastic Net
=====================

Lasso and elastic net (L1 and L2 penalisation) implemented using a
coordinate descent.

The coefficients can be forced to be positive.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 itertools 导入 cycle 函数
from itertools import cycle

# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt

# 导入 sklearn 库中的 datasets 模块
from sklearn import datasets
# 从 sklearn.linear_model 中导入 enet_path 和 lasso_path 函数
from sklearn.linear_model import enet_path, lasso_path

# 使用 sklearn 自带的糖尿病数据集加载数据，并将返回的 X 和 y 分别赋值
X, y = datasets.load_diabetes(return_X_y=True)

# 对数据 X 进行标准化处理，便于设置 l1_ratio 参数
X /= X.std(axis=0)

# 计算路径

# 设置一个较小的 eps 值，用于计算正则化路径的长度
eps = 5e-3

# 打印提示信息
print("Computing regularization path using the lasso...")
# 使用 lasso_path 函数计算 Lasso 正则化路径，返回正则化参数 alphas_lasso、系数 coefs_lasso 和轮数 _
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)

# 打印提示信息
print("Computing regularization path using the positive lasso...")
# 使用 positive=True 的 lasso_path 函数计算正向 Lasso 正则化路径，返回 alphas_positive_lasso、coefs_positive_lasso 和 _
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps=eps, positive=True
)

# 打印提示信息
print("Computing regularization path using the elastic net...")
# 使用 enet_path 函数计算 Elastic Net 正则化路径，返回 alphas_enet、coefs_enet 和 _
alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8)

# 打印提示信息
print("Computing regularization path using the positive elastic net...")
# 使用 positive=True 的 enet_path 函数计算正向 Elastic Net 正则化路径，返回 alphas_positive_enet、coefs_positive_enet 和 _
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True
)

# 显示结果

# 创建 Figure 对象，编号为 1
plt.figure(1)
# 设置颜色循环为 ["b", "r", "g", "c", "k"]
colors = cycle(["b", "r", "g", "c", "k"])
# 遍历 coefs_lasso、coefs_enet 和 colors，绘制 Lasso 和 Elastic-Net 路径的系数
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.semilogx(alphas_lasso, coef_l, c=c)
    l2 = plt.semilogx(alphas_enet, coef_e, linestyle="--", c=c)

# 设置 x 轴标签
plt.xlabel("alpha")
# 设置 y 轴标签
plt.ylabel("coefficients")
# 设置标题
plt.title("Lasso and Elastic-Net Paths")
# 设置图例
plt.legend((l1[-1], l2[-1]), ("Lasso", "Elastic-Net"), loc="lower right")
# 自动调整坐标轴范围
plt.axis("tight")

# 创建 Figure 对象，编号为 2
plt.figure(2)
# 遍历 coefs_lasso、coefs_positive_lasso 和 colors，绘制 Lasso 和正向 Lasso 路径的系数
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.semilogy(alphas_lasso, coef_l, c=c)
    l2 = plt.semilogy(alphas_positive_lasso, coef_pl, linestyle="--", c=c)

# 设置 x 轴标签
plt.xlabel("alpha")
# 设置 y 轴标签
plt.ylabel("coefficients")
# 设置标题
plt.title("Lasso and positive Lasso")
# 设置图例
plt.legend((l1[-1], l2[-1]), ("Lasso", "positive Lasso"), loc="lower right")
# 自动调整坐标轴范围
plt.axis("tight")

# 创建 Figure 对象，编号为 3
plt.figure(3)
# 遍历 coefs_enet、coefs_positive_enet 和 colors，绘制 Elastic-Net 和正向 Elastic-Net 路径的系数
for coef_e, coef_pe, c in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.semilogx(alphas_enet, coef_e, c=c)
    l2 = plt.semilogx(alphas_positive_enet, coef_pe, linestyle="--", c=c)

# 设置 x 轴标签
plt.xlabel("alpha")
# 设置 y 轴标签
plt.ylabel("coefficients")
# 设置标题
plt.title("Elastic-Net and positive Elastic-Net")
# 设置图例
plt.legend((l1[-1], l2[-1]), ("Elastic-Net", "positive Elastic-Net"), loc="lower right")
# 自动调整坐标轴范围
plt.axis("tight")
# 显示图形
plt.show()
```