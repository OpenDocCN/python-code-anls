# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_pca_vs_fa_model_selection.py`

```
"""
===============================================================
Model selection with Probabilistic PCA and Factor Analysis (FA)
===============================================================

Probabilistic PCA and Factor Analysis are probabilistic models.
The consequence is that the likelihood of new data can be used
for model selection and covariance estimation.
Here we compare PCA and FA with cross-validation on low rank data corrupted
with homoscedastic noise (noise variance
is the same for each feature) or heteroscedastic noise (noise variance
is the different for each feature). In a second step we compare the model
likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed
in recovering the size of the low rank subspace. The likelihood with PCA
is higher than FA in this case. However PCA fails and overestimates
the rank when heteroscedastic noise is present. Under appropriate
circumstances (choice of the number of components), the held-out
data is more likely for low rank models than for shrinkage models.

The automatic estimation from
Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
by Thomas P. Minka is also compared.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Create the data
# ---------------

import numpy as np
from scipy import linalg

n_samples, n_features, rank = 500, 25, 5
sigma = 1.0
rng = np.random.RandomState(42)
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.0
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

# %%
# Fit the models
# --------------

import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf, ShrunkCovariance
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score

n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    """
    Compute cross-validation scores for PCA and Factor Analysis (FA) models.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns:
    --------
    pca_scores : list of float
        Cross-validation scores for PCA.
    fa_scores : list of float
        Cross-validation scores for Factor Analysis.
    """
    pca = PCA(svd_solver="full")
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    """
    Compute cross-validation score for Shrunk Covariance estimator.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns:
    --------
    float
        Cross-validation score for Shrunk Covariance estimator.
    """
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    """
    Compute cross-validation score for Ledoit-Wolf estimator.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns:
    --------
    float
        Cross-validation score for Ledoit-Wolf estimator.
    """
    return np.mean(cross_val_score(LedoitWolf(), X))


for X, title in [(X_homo, "Homoscedastic Noise"), (X_hetero, "Heteroscedastic Noise")]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    # 根据 fa_scores 数组找到具有最大值的元素索引，并从 n_components 数组中获取相应的组件数
    n_components_fa = n_components[np.argmax(fa_scores)]

    # 使用 PCA 方法，设置 svd_solver 为 "full"，并使用 "mle" 方法确定最佳的组件数量
    pca = PCA(svd_solver="full", n_components="mle")
    pca.fit(X)
    # 获取使用 PCA 方法确定的最佳组件数量
    n_components_pca_mle = pca.n_components_

    # 打印最佳 PCA 交叉验证得分对应的组件数量
    print("best n_components by PCA CV = %d" % n_components_pca)
    # 打印最佳因子分析交叉验证得分对应的组件数量
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    # 打印使用 PCA MLE 方法确定的最佳组件数量
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    # 创建一个新的图形窗口
    plt.figure()
    # 绘制 PCA 方法得分曲线，用蓝色表示
    plt.plot(n_components, pca_scores, "b", label="PCA scores")
    # 绘制因子分析方法得分曲线，用红色表示
    plt.plot(n_components, fa_scores, "r", label="FA scores")
    # 在图中插入真实的组件数，用绿色表示
    plt.axvline(rank, color="g", label="TRUTH: %d" % rank, linestyle="-")
    # 在图中插入 PCA 交叉验证最佳组件数，用蓝色虚线表示
    plt.axvline(
        n_components_pca,
        color="b",
        label="PCA CV: %d" % n_components_pca,
        linestyle="--",
    )
    # 在图中插入因子分析交叉验证最佳组件数，用红色虚线表示
    plt.axvline(
        n_components_fa,
        color="r",
        label="FactorAnalysis CV: %d" % n_components_fa,
        linestyle="--",
    )
    # 在图中插入使用 PCA MLE 方法确定的最佳组件数，用黑色虚线表示
    plt.axvline(
        n_components_pca_mle,
        color="k",
        label="PCA MLE: %d" % n_components_pca_mle,
        linestyle="--",
    )

    # 在图中插入使用收缩协方差 MLE 方法计算的得分，用紫色表示
    plt.axhline(
        shrunk_cov_score(X),
        color="violet",
        label="Shrunk Covariance MLE",
        linestyle="-.",
    )
    # 在图中插入使用 LedoitWolf MLE 方法计算的得分，用橙色表示
    plt.axhline(
        lw_score(X),
        color="orange",
        label="LedoitWolf MLE",
        linestyle="-.",
    )

    # 设置 x 轴标签为 "nb of components"
    plt.xlabel("nb of components")
    # 设置 y 轴标签为 "CV scores"
    plt.ylabel("CV scores")
    # 在图中设置图例位置为右下角
    plt.legend(loc="lower right")
    # 设置图的标题为给定的 title 变量的值
    plt.title(title)
# 显示 matplotlib 中当前的图形
plt.show()
```