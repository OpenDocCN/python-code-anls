# `D:\src\scipysrc\scikit-learn\examples\classification\plot_lda_qda.py`

```
"""
====================================================================
Linear and Quadratic Discriminant Analysis with covariance ellipsoid
====================================================================

This example plots the covariance ellipsoids of each class and the decision boundary
learned by :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA) and
:class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` (QDA). The
ellipsoids display the double standard deviation for each class. With LDA, the standard
deviation is the same for all the classes, while each class has its own standard
deviation with QDA.
"""

# %%
# Data generation
# ---------------
#
# First, we define a function to generate synthetic data. It creates two blobs centered
# at `(0, 0)` and `(1, 1)`. Each blob is assigned a specific class. The dispersion of
# the blob is controlled by the parameters `cov_class_1` and `cov_class_2`, that are the
# covariance matrices used when generating the samples from the Gaussian distributions.
import numpy as np

def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):
    rng = np.random.RandomState(seed)
    X = np.concatenate(
        [
            rng.randn(n_samples, n_features) @ cov_class_1,
            rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1]),
        ]
    )
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    return X, y


# %%
# We generate three datasets. In the first dataset, the two classes share the same
# covariance matrix, and this covariance matrix has the specificity of being spherical
# (isotropic). The second dataset is similar to the first one but does not enforce the
# covariance to be spherical. Finally, the third dataset has a non-spherical covariance
# matrix for each class.
covariance = np.array([[1, 0], [0, 1]])
X_isotropic_covariance, y_isotropic_covariance = make_data(
    n_samples=1_000,
    n_features=2,
    cov_class_1=covariance,
    cov_class_2=covariance,
    seed=0,
)
covariance = np.array([[0.0, -0.23], [0.83, 0.23]])
X_shared_covariance, y_shared_covariance = make_data(
    n_samples=300,
    n_features=2,
    cov_class_1=covariance,
    cov_class_2=covariance,
    seed=0,
)
cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
cov_class_2 = cov_class_1.T
X_different_covariance, y_different_covariance = make_data(
    n_samples=300,
    n_features=2,
    cov_class_1=cov_class_1,
    cov_class_2=cov_class_2,
    seed=0,
)


# %%
# Plotting Functions
# ------------------
#
# The code below is used to plot several pieces of information from the estimators used,
# i.e., :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA) and
# :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` (QDA). The
# displayed information includes:
#
# - the decision boundary based on the probability estimate of the estimator;
# 导入必要的库和模块
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 定义函数：绘制椭圆表示协方差矩阵
def plot_ellipse(mean, cov, color, ax):
    # 计算特征值和特征向量
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # 转换为角度

    # 创建椭圆对象，表示在均值周围2倍标准差的高斯分布
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)

# 定义函数：绘制分类结果的图像
def plot_result(estimator, X, y, ax):
    # 颜色映射
    cmap = colors.ListedColormap(["tab:red", "tab:blue"])

    # 绘制决策边界的显示
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="pcolormesh",
        ax=ax,
        cmap="RdBu",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="contour",
        ax=ax,
        alpha=1.0,
        levels=[0.5],
    )

    # 预测结果
    y_pred = estimator.predict(X)

    # 分类正确的样本和错误的样本
    X_right, y_right = X[y == y_pred], y[y == y_pred]
    X_wrong, y_wrong = X[y != y_pred], y[y != y_pred]

    # 绘制散点图：正确分类的样本
    ax.scatter(X_right[:, 0], X_right[:, 1], c=y_right, s=20, cmap=cmap, alpha=0.5)
    # 绘制散点图：错误分类的样本
    ax.scatter(
        X_wrong[:, 0],
        X_wrong[:, 1],
        c=y_wrong,
        s=30,
        cmap=cmap,
        alpha=0.9,
        marker="x",
    )

    # 绘制均值点，用星号标记
    ax.scatter(
        estimator.means_[:, 0],
        estimator.means_[:, 1],
        c="yellow",
        s=200,
        marker="*",
        edgecolor="black",
    )

    # 根据估计器类型选择协方差矩阵
    if isinstance(estimator, LinearDiscriminantAnalysis):
        covariance = [estimator.covariance_] * 2
    else:
        covariance = estimator.covariance_

    # 绘制表示协方差的椭圆
    plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
    plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)

    # 设置图像纵横比为1
    ax.set_box_aspect(1)
    # 隐藏图像的四个边框
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置不显示刻度
    ax.set(xticks=[], yticks=[])

# %%
# LDA和QDA的比较
# -------------------------
#
# 在所有三个数据集上比较两种估计器LDA和QDA。

# 创建一个包含3行2列子图的图像
fig, axs = plt.subplots(nrows=3, ncols=2, sharex="row", sharey="row", figsize=(8, 12))

# 创建LDA估计器对象
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
# 创建一个 QuadraticDiscriminantAnalysis 对象 qda，设置存储协方差矩阵的选项为True

for ax_row, X, y in zip(
    axs,
    (X_isotropic_covariance, X_shared_covariance, X_different_covariance),
    (y_isotropic_covariance, y_shared_covariance, y_different_covariance),
):
    # 遍历 axs、X、y，分别为图形对象、特征数据集和标签数据集
    lda.fit(X, y)
    # 使用线性判别分析模型 lda 对数据 X 和标签 y 进行拟合
    plot_result(lda, X, y, ax_row[0])
    # 调用 plot_result 函数绘制 lda 拟合后的结果到 ax_row[0] 上
    qda.fit(X, y)
    # 使用二次判别分析模型 qda 对数据 X 和标签 y 进行拟合
    plot_result(qda, X, y, ax_row[1])
    # 调用 plot_result 函数绘制 qda 拟合后的结果到 ax_row[1] 上

axs[0, 0].set_title("Linear Discriminant Analysis")
# 设置 axs[0, 0] 的标题为 "Linear Discriminant Analysis"
axs[0, 0].set_ylabel("Data with fixed and spherical covariance")
# 设置 axs[0, 0] 的 y 轴标签为 "Data with fixed and spherical covariance"
axs[1, 0].set_ylabel("Data with fixed covariance")
# 设置 axs[1, 0] 的 y 轴标签为 "Data with fixed covariance"
axs[0, 1].set_title("Quadratic Discriminant Analysis")
# 设置 axs[0, 1] 的标题为 "Quadratic Discriminant Analysis"
axs[2, 0].set_ylabel("Data with varying covariances")
# 设置 axs[2, 0] 的 y 轴标签为 "Data with varying covariances"
fig.suptitle(
    "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
    y=0.94,
    fontsize=15,
)
# 设置整体图形的标题为 "Linear Discriminant Analysis vs Quadratic Discriminant Analysis"，
# 设置标题的垂直位置为 0.94，字体大小为 15
plt.show()
# 显示绘制的图形

# %%
# 首先需要注意的是，对于第一个和第二个数据集，LDA 和 QDA 是等价的。主要区别在于，
# LDA 假设每个类的协方差矩阵相等，而 QDA 对每个类估计一个协方差矩阵。因为在这些
# 情况下，数据生成过程为两个类使用相同的协方差矩阵，QDA 估计的两个协方差矩阵几乎
# 相等，因此等效于 LDA 估计的协方差矩阵。
#
# 在第一个数据集中，用于生成数据集的协方差矩阵是球形的，导致判别边界与两个均值的
# 垂直平分线重合。但对于第二个数据集，情况则不同，判别边界只通过两个均值的中间。
#
# 最后，在第三个数据集中，我们观察到 LDA 和 QDA 的真正区别。QDA 拟合两个协方差
# 矩阵，并提供非线性判别边界，而 LDA 由于假设两个类共享一个协方差矩阵，因此拟合效果不佳，欠拟合。
```