# `D:\src\scipysrc\scikit-learn\examples\cross_decomposition\plot_pcr_vs_pls.py`

```
"""
==================================================================
Principal Component Regression vs Partial Least Squares Regression
==================================================================

This example compares `Principal Component Regression
<https://en.wikipedia.org/wiki/Principal_component_regression>`_ (PCR) and
`Partial Least Squares Regression
<https://en.wikipedia.org/wiki/Partial_least_squares_regression>`_ (PLS) on a
toy dataset. Our goal is to illustrate how PLS can outperform PCR when the
target is strongly correlated with some directions in the data that have a
low variance.

PCR is a regressor composed of two steps: first,
:class:`~sklearn.decomposition.PCA` is applied to the training data, possibly
performing dimensionality reduction; then, a regressor (e.g. a linear
regressor) is trained on the transformed samples. In
:class:`~sklearn.decomposition.PCA`, the transformation is purely
unsupervised, meaning that no information about the targets is used. As a
result, PCR may perform poorly in some datasets where the target is strongly
correlated with *directions* that have low variance. Indeed, the
dimensionality reduction of PCA projects the data into a lower dimensional
space where the variance of the projected data is greedily maximized along
each axis. Despite them having the most predictive power on the target, the
directions with a lower variance will be dropped, and the final regressor
will not be able to leverage them.

PLS is both a transformer and a regressor, and it is quite similar to PCR: it
also applies a dimensionality reduction to the samples before applying a
linear regressor to the transformed data. The main difference with PCR is
that the PLS transformation is supervised. Therefore, as we will see in this
example, it does not suffer from the issue we just mentioned.

"""

# %%
# The data
# --------
#
# We start by creating a simple dataset with two features. Before we even dive
# into PCR and PLS, we fit a PCA estimator to display the two principal
# components of this dataset, i.e. the two directions that explain the most
# variance in the data.

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from sklearn.decomposition import PCA  # 导入 PCA 模型，用于主成分分析

rng = np.random.RandomState(0)  # 创建随机数生成器 rng，种子为 0
n_samples = 500  # 设定样本数量为 500
cov = [[3, 3], [3, 4]]  # 设定协方差矩阵
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)  # 生成多变量正态分布样本 X
pca = PCA(n_components=2).fit(X)  # 对样本 X 进行 PCA 拟合，保留两个主成分

plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")  # 绘制散点图，显示样本 X 的分布
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # 按其方差解释力量缩放主成分
    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )  # 绘制主成分的方向向量
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)  # 设置图形的标题和坐标轴标签
plt.legend()  # 显示图例
plt.show()  # 展示图形

# %%
# 定义目标变量 `y`，它与具有较小方差的方向强相关。为此，我们将 `X` 投影到第二个主成分上，并添加一些噪音。

y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2

# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# 在第一个子图上绘制散点图，横轴为数据在第一主成分上的投影，纵轴为 `y`
axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first PCA component", ylabel="y")

# 在第二个子图上绘制散点图，横轴为数据在第二主成分上的投影，纵轴为 `y`
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second PCA component", ylabel="y")

# 调整布局使子图之间的空间合理分配，并显示图形
plt.tight_layout()
plt.show()

# %%
# 投影到单个成分及预测能力
# ------------------------------------------------
#
# 我们现在创建两个回归器：PCR 和 PLS，并且为了说明，我们将成分数设置为1。在将数据传递给PCR的PCA步骤之前，
# 我们首先对其进行标准化，这是良好实践的推荐。PLS估计器具有内置的缩放能力。
#
# 对于这两个模型，我们绘制数据在第一个成分上的投影与目标之间的关系。在两种情况下，这些投影数据将作为训练数据供回归器使用。
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# 创建PCR管道：标准化 -> PCA -> 线性回归
pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)
pca = pcr.named_steps["pca"]  # 获取管道中的PCA步骤

# 创建PLS回归器
pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

# 创建包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# 在第一个子图上绘制散点图：PCA投影数据与真实目标数据，以及PCA预测结果
axes[0].scatter(pca.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[0].scatter(
    pca.transform(X_test), pcr.predict(X_test), alpha=0.3, label="predictions"
)
axes[0].set(
    xlabel="Projected data onto first PCA component", ylabel="y", title="PCR / PCA"
)
axes[0].legend()

# 在第二个子图上绘制散点图：PLS投影数据与真实目标数据，以及PLS预测结果
axes[1].scatter(pls.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[1].scatter(
    pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="predictions"
)
axes[1].set(xlabel="Projected data onto first PLS component", ylabel="y", title="PLS")
axes[1].legend()

# 调整布局使子图之间的空间合理分配，并显示图形
plt.tight_layout()
plt.show()

# %%
# 如预期，PCR的无监督PCA转换已经丢弃了第二个成分，即具有最低方差的方向，尽管它是最具预测力的方向。
# 这是因为PCA是完全无监督的转换，导致投影数据对目标的预测能力较低。
#
# 另一方面，PLS回归器能够捕捉到具有最低方差的方向的效应，这要归功于它使用目标信息。
# 输出 PCR 模型在测试集上的 R-squared 分数，表示模型对目标变量的解释能力
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")

# 输出 PLS 模型在测试集上的 R-squared 分数，表示模型对目标变量的解释能力
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")

# %%
# 最后一点说明：我们注意到使用 2 个主成分的 PCR 表现与 PLS 相当好：
# 这是因为在这种情况下，PCR 能够利用第二个主成分，后者对目标变量具有最高的预测能力。

# 创建一个包含 PCA 和线性回归的管道，设置主成分数为 2，并进行模型拟合
pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
pca_2.fit(X_train, y_train)

# 输出使用 2 个主成分的 PCR 模型在测试集上的 R-squared 分数，衡量模型的性能
print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")
```