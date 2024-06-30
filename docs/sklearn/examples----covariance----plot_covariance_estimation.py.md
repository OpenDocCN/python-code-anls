# `D:\src\scipysrc\scikit-learn\examples\covariance\plot_covariance_estimation.py`

```
"""
=======================================================================
Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood
=======================================================================

当进行协方差估计时，通常的方法是使用最大似然估计器，例如
:class:`~sklearn.covariance.EmpiricalCovariance`。它是无偏的，即当给定许多
观测时收敛到真实（总体）协方差。然而，对其进行正则化也可能是有益的，
以减少其方差；这进而引入了一些偏差。本例说明了在
:ref:`shrunk_covariance` 估计器中使用的简单正则化。具体而言，重点在于
如何设置正则化的量，即如何选择偏差-方差的权衡。

"""

# %%
# Generate sample data
# --------------------

import numpy as np

# 定义数据特征数和样本数
n_features, n_samples = 40, 20

# 设置随机种子以便结果可复现
np.random.seed(42)

# 生成训练数据和测试数据，服从正态分布
base_X_train = np.random.normal(size=(n_samples, n_features))
base_X_test = np.random.normal(size=(n_samples, n_features))

# 生成颜色矩阵
coloring_matrix = np.random.normal(size=(n_features, n_features))

# 将训练数据和测试数据通过颜色矩阵进行线性变换
X_train = np.dot(base_X_train, coloring_matrix)
X_test = np.dot(base_X_test, coloring_matrix)


# %%
# Compute the likelihood on test data
# -----------------------------------

from scipy import linalg

# 导入协方差估计相关库
from sklearn.covariance import ShrunkCovariance, empirical_covariance, log_likelihood

# 定义一系列可能的收缩系数值
shrinkages = np.logspace(-2, 0, 30)

# 计算负对数似然值，使用不同的收缩系数对训练数据进行拟合，并在测试数据上评分
negative_logliks = [
    -ShrunkCovariance(shrinkage=s).fit(X_train).score(X_test) for s in shrinkages
]

# 在真实模型下计算对数似然值，这在实际设置中是无法获得的
real_cov = np.dot(coloring_matrix.T, coloring_matrix)
emp_cov = empirical_covariance(X_train)
loglik_real = -log_likelihood(emp_cov, linalg.inv(real_cov))


# %%
# Compare different approaches to setting the regularization parameter
# --------------------------------------------------------------------
#
# 这里我们比较了3种设置正则化参数的方法：
#
# * 通过对三折交叉验证似然性在一系列潜在的收缩参数网格上进行参数设置。
#
# * 由Ledoit和Wolf提出的一个近似最优正则化参数的闭式公式，得到
#   :class:`~sklearn.covariance.LedoitWolf` 协方差估计。
#
# * Ledoit-Wolf收缩的改进，由Chen等人提出的
#   :class:`~sklearn.covariance.OAS`。在假设数据为高斯分布时，其收敛性显著更好，
#   特别是在小样本情况下。

from sklearn.covariance import OAS, LedoitWolf
from sklearn.model_selection import GridSearchCV

# 使用GridSearch寻找最优的收缩系数
tuned_parameters = [{"shrinkage": shrinkages}]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
cv.fit(X_train)
# Ledoit-Wolf optimal shrinkage coefficient estimate
lw = LedoitWolf()
# 训练 Ledoit-Wolf 模型并计算在测试集上的对数似然分数
loglik_lw = lw.fit(X_train).score(X_test)

# OAS coefficient estimate
oa = OAS()
# 训练 OAS 模型并计算在测试集上的对数似然分数
loglik_oa = oa.fit(X_train).score(X_test)

# %%
# Plot results
# ------------
#
#
# To quantify estimation error, we plot the likelihood of unseen data for
# different values of the shrinkage parameter. We also show the choices by
# cross-validation, or with the LedoitWolf and OAS estimates.

import matplotlib.pyplot as plt

# 创建一个新的图形对象
fig = plt.figure()
# 设置图形标题
plt.title("Regularized covariance: likelihood and shrinkage coefficient")
# 设置 x 轴标签
plt.xlabel("Regularization parameter: shrinkage coefficient")
# 设置 y 轴标签
plt.ylabel("Error: negative log-likelihood on test data")

# 绘制收缩参数与负对数似然值之间的双对数曲线
plt.loglog(shrinkages, negative_logliks, label="Negative log-likelihood")

# 绘制真实协方差估计的理论最大似然值
plt.plot(plt.xlim(), 2 * [loglik_real], "--r", label="Real covariance likelihood")

# 调整视图范围
lik_max = np.amax(negative_logliks)
lik_min = np.amin(negative_logliks)
ymin = lik_min - 6.0 * np.log((plt.ylim()[1] - plt.ylim()[0]))
ymax = lik_max + 10.0 * np.log(lik_max - lik_min)
xmin = shrinkages[0]
xmax = shrinkages[-1]

# 绘制 Ledoit-Wolf 估计的收缩参数对应的负对数似然值
plt.vlines(
    lw.shrinkage_,
    ymin,
    -loglik_lw,
    color="magenta",
    linewidth=3,
    label="Ledoit-Wolf estimate",
)
# 绘制 OAS 估计的收缩参数对应的负对数似然值
plt.vlines(
    oa.shrinkage_, ymin, -loglik_oa, color="purple", linewidth=3, label="OAS estimate"
)
# 绘制交叉验证选择的最佳估计的收缩参数对应的负对数似然值
plt.vlines(
    cv.best_estimator_.shrinkage,
    ymin,
    -cv.best_estimator_.score(X_test),
    color="cyan",
    linewidth=3,
    label="Cross-validation best estimate",
)

# 设置 y 轴和 x 轴的显示范围
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
# 添加图例
plt.legend()

# 展示图形
plt.show()

# %%
# .. note::
#
#    The maximum likelihood estimate corresponds to no shrinkage,
#    and thus performs poorly. The Ledoit-Wolf estimate performs really well,
#    as it is close to the optimal and is not computationally costly. In this
#    example, the OAS estimate is a bit further away. Interestingly, both
#    approaches outperform cross-validation, which is significantly more
#    computationally costly.
```