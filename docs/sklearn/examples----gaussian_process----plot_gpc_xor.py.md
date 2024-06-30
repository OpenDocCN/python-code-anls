# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpc_xor.py`

```
"""
========================================================================
Illustration of Gaussian process classification (GPC) on the XOR dataset
========================================================================

This example illustrates GPC on XOR data. Compared are a stationary, isotropic
kernel (RBF) and a non-stationary kernel (DotProduct). On this particular
dataset, the DotProduct kernel obtains considerably better results because the
class-boundaries are linear and coincide with the coordinate axes. In general,
stationary kernels often obtain better results.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库
import numpy as np  # 导入 NumPy 数学库

from sklearn.gaussian_process import GaussianProcessClassifier  # 导入高斯过程分类器
from sklearn.gaussian_process.kernels import RBF, DotProduct  # 导入 RBF 和 DotProduct 内核

# 创建一个网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))

# 使用随机种子创建随机数生成器
rng = np.random.RandomState(0)
# 生成符合正态分布的随机样本作为输入特征 X
X = rng.randn(200, 2)
# 根据 XOR 逻辑生成目标标签 Y
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# 绘制图形的主体部分
plt.figure(figsize=(10, 5))

# 定义两种不同的内核函数：RBF 和 DotProduct
kernels = [1.0 * RBF(length_scale=1.15), 1.0 * DotProduct(sigma_0=1.0) ** 2]
for i, kernel in enumerate(kernels):
    # 创建高斯过程分类器对象，使用指定的内核，使用 warm_start=True 开启热启动
    clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X, Y)

    # 在网格上预测每个数据点的决策函数值，并将结果转换成概率
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)

    # 绘制子图
    plt.subplot(1, 2, i + 1)
    image = plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=["k"])
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    plt.title(
        "%s\n Log-Marginal-Likelihood:%.3f"
        % (clf.kernel_, clf.log_marginal_likelihood(clf.kernel_.theta)),
        fontsize=12,
    )

# 调整布局使子图紧凑显示
plt.tight_layout()
# 展示绘制的图形
plt.show()
```