# `D:\src\scipysrc\scikit-learn\examples\cross_decomposition\plot_compare_cross_decomposition.py`

```
"""
===================================
Compare cross decomposition methods
===================================

Simple usage of various cross decomposition algorithms:

- PLSCanonical
- PLSRegression, with multivariate response, a.k.a. PLS2
- PLSRegression, with univariate response, a.k.a. PLS1
- CCA

Given 2 multivariate covarying two-dimensional datasets, X, and Y,
PLS extracts the 'directions of covariance', i.e. the components of each
datasets that explain the most shared variance between both datasets.
This is apparent on the **scatterplot matrix** display: components 1 in
dataset X and dataset Y are maximally correlated (points lie around the
first diagonal). This is also true for components 2 in both dataset,
however, the correlation across datasets for different components is
weak: the point cloud is very spherical.
"""

# %%
# Dataset based latent variables model
# ------------------------------------

import numpy as np

n = 500
# 2 latents vars:
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)

# 生成一个包含四列的潜变量数据集，其中两列重复两次以模拟共变性
latents = np.array([l1, l1, l2, l2]).T
# 为 X 和 Y 添加随机噪声以创建观测数据集
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

# 将数据集分为训练集和测试集
X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]

# 打印 X 和 Y 数据集的相关性矩阵
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))

# %%
# Canonical (symmetric) PLS
# -------------------------
#
# Transform data
# ~~~~~~~~~~~~~~

from sklearn.cross_decomposition import PLSCanonical

# 初始化并拟合 Canonical PLS 模型
plsca = PLSCanonical(n_components=2)
plsca.fit(X_train, Y_train)
# 转换训练集和测试集数据
X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

# %%
# Scatter plot of scores
# ~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt

# 在散点图中绘制各成分的分数
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 1: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

plt.subplot(224)
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 2: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

# 在散点图中绘制不同成分之间的关系
plt.subplot(222)
plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train", marker="*", s=50)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test", marker="*", s=50)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title(
    "X comp. 1 vs X comp. 2 (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
    # 计算测试集数据 X_test_r 的两列之间的皮尔逊相关系数
    # np 是 NumPy 库的别名，corrcoef 函数用于计算相关系数矩阵
    # X_test_r[:, 0] 选取所有行的第一列数据，X_test_r[:, 1] 选取所有行的第二列数据
    # [0, 1] 表示取相关系数矩阵的第一行第二列元素，即第一列和第二列之间的相关系数
    % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1]
)
plt.legend(loc="best")  # 设置图例位置为最佳，用于标识散点图中的数据点来源
plt.xticks(())  # 设置 X 轴刻度为空，即不显示 X 轴刻度
plt.yticks(())  # 设置 Y 轴刻度为空，即不显示 Y 轴刻度

plt.subplot(223)  # 在一个 2x2 的子图中选择第三个子图进行绘制
plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train", marker="*", s=50)  # 绘制训练集的散点图
plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test", marker="*", s=50)  # 绘制测试集的散点图
plt.xlabel("Y comp. 1")  # 设置 X 轴标签
plt.ylabel("Y comp. 2")  # 设置 Y 轴标签
plt.title(
    "Y comp. 1 vs Y comp. 2 , (test corr = %.2f)"
    % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1]
)  # 设置子图标题，显示测试集的两个主成分的相关性系数
plt.legend(loc="best")  # 设置图例位置为最佳，用于标识散点图中的数据点来源
plt.xticks(())  # 设置 X 轴刻度为空，即不显示 X 轴刻度
plt.yticks(())  # 设置 Y 轴刻度为空，即不显示 Y 轴刻度
plt.show()  # 显示绘制的图形

# %%
# PLS regression, with multivariate response, a.k.a. PLS2
# -------------------------------------------------------

from sklearn.cross_decomposition import PLSRegression

n = 1000  # 样本数
q = 3  # 响应变量的数量
p = 10  # 自变量的数量
X = np.random.normal(size=n * p).reshape((n, p))  # 生成服从正态分布的自变量 X
B = np.array([[1, 2] + [0] * (p - 2)] * q).T  # 真实系数矩阵 B，用于生成响应变量 Y
# 每个 Yj = 1*X1 + 2*X2 + 噪声
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5  # 生成响应变量 Y

pls2 = PLSRegression(n_components=3)  # 创建 PLS2 模型对象，指定主成分数量为 3
pls2.fit(X, Y)  # 拟合 PLS2 模型
print("True B (such that: Y = XB + Err)")  # 打印真实的系数矩阵 B
print(B)
# 比较 pls2.coef_ 与真实系数矩阵 B
print("Estimated B")
print(np.round(pls2.coef_, 1))  # 打印估计得到的系数矩阵，四舍五入保留一位小数
pls2.predict(X)  # 使用拟合好的模型进行预测

# %%
# PLS regression, with univariate response, a.k.a. PLS1
# -----------------------------------------------------

n = 1000  # 样本数
p = 10  # 自变量的数量
X = np.random.normal(size=n * p).reshape((n, p))  # 生成服从正态分布的自变量 X
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5  # 生成服从线性关系的单变量响应变量 y
pls1 = PLSRegression(n_components=3)  # 创建 PLS1 模型对象，指定主成分数量为 3
pls1.fit(X, y)  # 拟合 PLS1 模型
# 注意，主成分数量超过了 y 的维度（为 1）
print("Estimated betas")
print(np.round(pls1.coef_, 1))  # 打印估计得到的系数，四舍五入保留一位小数

# %%
# CCA (PLS mode B with symmetric deflation)
# -----------------------------------------

from sklearn.cross_decomposition import CCA

cca = CCA(n_components=2)  # 创建 CCA 模型对象，指定主成分数量为 2
cca.fit(X_train, Y_train)  # 拟合 CCA 模型，用训练集 X_train 和 Y_train
X_train_r, Y_train_r = cca.transform(X_train, Y_train)  # 对训练集进行变换得到投影后的数据
X_test_r, Y_test_r = cca.transform(X_test, Y_test)  # 对测试集进行变换得到投影后的数据
```