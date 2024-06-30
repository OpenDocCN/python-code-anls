# `D:\src\scipysrc\scikit-learn\examples\mixture\plot_concentration_prior.py`

```
"""
========================================================================
Concentration Prior Type Analysis of Variation Bayesian Gaussian Mixture
========================================================================

This example plots the ellipsoids obtained from a toy dataset (mixture of three
Gaussians) fitted by the ``BayesianGaussianMixture`` class models with a
Dirichlet distribution prior
(``weight_concentration_prior_type='dirichlet_distribution'``) and a Dirichlet
process prior (``weight_concentration_prior_type='dirichlet_process'``). On
each figure, we plot the results for three different values of the weight
concentration prior.

The ``BayesianGaussianMixture`` class can adapt its number of mixture
components automatically. The parameter ``weight_concentration_prior`` has a
direct link with the resulting number of components with non-zero weights.
Specifying a low value for the concentration prior will make the model put most
of the weight on few components set the remaining components weights very close
to zero. High values of the concentration prior will allow a larger number of
components to be active in the mixture.

The Dirichlet process prior allows to define an infinite number of components
and automatically selects the correct number of components: it activates a
component only if it is necessary.

On the contrary the classical finite mixture model with a Dirichlet
distribution prior will favor more uniformly weighted components and therefore
tends to divide natural clusters into unnecessary sub-components.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import BayesianGaussianMixture


def plot_ellipses(ax, weights, means, covars):
    """
    Plot ellipses representing Gaussian components.

    Parameters:
    - ax: matplotlib axis object
        The axis to draw the ellipses on.
    - weights: array-like, shape (n_components,)
        The weights of Gaussian components.
    - means: array-like, shape (n_components, n_features)
        The means of Gaussian components.
    - covars: array-like, shape (n_components, n_features, n_features)
        The covariance matrices of Gaussian components.
    """
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor("#56B4E9")
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    """
    Plot the clustering results and ellipses.

    Parameters:
    - ax1: matplotlib axis object
        The first axis to plot the data and ellipses.
    - ax2: matplotlib axis object
        The second axis for additional plot settings.
    - estimator: BayesianGaussianMixture object
        The fitted Bayesian Gaussian Mixture model.
    - X: array-like, shape (n_samples, n_features)
        The input data samples.
    - y: array-like, shape (n_samples,)
        The true labels (not used in this function).
    - title: str
        The title of the plot.
    - plot_title: bool, optional
        Whether to plot the title or not.
    """
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker="o", color=colors[y], alpha=0.8)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-3.0, 3.0)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction="out")
    ax2.yaxis.grid(True, alpha=0.7)
    # 使用 enumerate() 函数遍历 estimator.weights_ 中的索引和对应的权重 w
    for k, w in enumerate(estimator.weights_):
        # 在 ax2 上绘制柱状图，显示权重 w，使用指定的颜色、边缘颜色和对齐方式
        ax2.bar(
            k,
            w,
            width=0.9,
            color="#56B4E9",
            zorder=3,
            align="center",
            edgecolor="black",
        )
        # 在每个柱状图上方添加文本，显示权重 w 的百分比格式
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.0), horizontalalignment="center")
    
    # 设置 ax2 的 x 轴和 y 轴的显示范围
    ax2.set_xlim(-0.6, 2 * n_components - 0.4)
    ax2.set_ylim(0.0, 1.1)
    
    # 针对 ax2 的 y 轴和 x 轴设置不显示左侧、右侧、标签
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax2.tick_params(axis="x", which="both", top=False)

    # 如果 plot_title 为真，则设置 ax1 和 ax2 的 ylabel 分别为 "Estimated Mixtures" 和 "Weight of each component"
    if plot_title:
        ax1.set_ylabel("Estimated Mixtures")
        ax2.set_ylabel("Weight of each component")
# Parameters of the dataset
# 设置数据集的参数

random_state, n_components, n_features = 2, 3, 2
# 随机数种子、成分数量、特征数量

colors = np.array(["#0072B2", "#F0E442", "#D55E00"])
# 颜色数组，用于图形显示

covars = np.array(
    [[[0.7, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]]]
)
# 协方差矩阵数组，每个成分的协方差矩阵

samples = np.array([200, 500, 200])
# 每个成分的样本数量

means = np.array([[0.0, -0.70], [0.0, 0.0], [0.0, 0.70]])
# 每个成分的均值数组

# mean_precision_prior= 0.8 to minimize the influence of the prior
# mean_precision_prior= 0.8 减少先验的影响

estimators = [
    (
        "Finite mixture with a Dirichlet distribution\nprior and " r"$\gamma_0=$",
        BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=2 * n_components,
            reg_covar=0,
            init_params="random",
            max_iter=1500,
            mean_precision_prior=0.8,
            random_state=random_state,
        ),
        [0.001, 1, 1000],
    ),
    (
        "Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
        BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * n_components,
            reg_covar=0,
            init_params="random",
            max_iter=1500,
            mean_precision_prior=0.8,
            random_state=random_state,
        ),
        [1, 1000, 100000],
    ),
]
# 估计器列表，包含标题、估计器对象和集中参数先验

# Generate data
# 生成数据

rng = np.random.RandomState(random_state)
# 随机数生成器，使用指定的随机种子

X = np.vstack(
    [
        rng.multivariate_normal(means[j], covars[j], samples[j])
        for j in range(n_components)
    ]
)
# 生成样本数据 X，按照每个成分的均值和协方差生成对应数量的样本

y = np.concatenate([np.full(samples[j], j, dtype=int) for j in range(n_components)])
# 生成目标标签 y，每个成分对应的样本数量

# Plot results in two different figures
# 在两个不同的图中绘制结果

for title, estimator, concentrations_prior in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    # 创建图形窗口，指定尺寸

    plt.subplots_adjust(
        bottom=0.04, top=0.90, hspace=0.05, wspace=0.05, left=0.03, right=0.99
    )
    # 调整子图的布局参数

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    # 创建网格布局，3行，列数由集中参数先验的数量决定

    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        # 设置估计器的权重集中参数先验

        estimator.fit(X)
        # 使用样本数据 X 拟合估计器

        plot_results(
            plt.subplot(gs[0:2, k]),
            plt.subplot(gs[2, k]),
            estimator,
            X,
            y,
            r"%s$%.1e$" % (title, concentration),
            plot_title=k == 0,
        )
        # 绘制结果图像，包括标题和子图

plt.show()
# 显示图形
```