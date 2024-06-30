# `D:\src\scipysrc\scikit-learn\examples\mixture\plot_gmm_pdf.py`

```
"""
=========================================
Density Estimation for a Gaussian mixture
=========================================

Plot the density estimation of a mixture of two Gaussians. Data is
generated from two Gaussians with different centers and covariance
matrices.

"""

import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算
from matplotlib.colors import LogNorm  # 导入LogNorm用于对数标准化

from sklearn import mixture  # 导入sklearn中的高斯混合模型

n_samples = 300  # 设定生成样本数量

# 生成随机样本，包含两个分量
np.random.seed(0)

# 生成位于(20, 20)附近的球形数据
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# 生成以零为中心的拉伸高斯数据
C = np.array([[0.0, -0.7], [3.5, 0.7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# 将两个数据集合并成最终的训练集
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# 使用两个分量拟合一个高斯混合模型
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
clf.fit(X_train)

# 将模型预测的分数显示为等高线图
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
)
CB = plt.colorbar(CS, shrink=0.8, extend="both")
plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)

plt.title("Negative log-likelihood predicted by a GMM")  # 设置图表标题
plt.axis("tight")  # 设置坐标轴范围适应数据
plt.show()  # 显示图像
```