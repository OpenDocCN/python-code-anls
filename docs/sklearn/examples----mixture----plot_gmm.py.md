# `D:\src\scipysrc\scikit-learn\examples\mixture\plot_gmm.py`

```
"""
=================================
Gaussian Mixture Model Ellipsoids
=================================

Plot the confidence ellipsoids of a mixture of two Gaussians
obtained with Expectation Maximisation (``GaussianMixture`` class) and
Variational Inference (``BayesianGaussianMixture`` class models with
a Dirichlet process prior).

Both models have access to five components with which to fit the data. Note
that the Expectation Maximisation model will necessarily use all five
components while the Variational Inference model will effectively only use as
many as are needed for a good fit. Here we can see that the Expectation
Maximisation model splits some components arbitrarily, because it is trying to
fit too many components, while the Dirichlet Process model adapts it number of
state automatically.

This example doesn't show it, as we're in a low-dimensional space, but
another advantage of the Dirichlet process model is that it can fit
full covariance matrices effectively even when there are less examples
per cluster than there are dimensions in the data, due to
regularization properties of the inference algorithm.

"""

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn import mixture

# Iterator for cycling through colors
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


# Function to plot the results of Gaussian Mixture Model
def plot_results(X, Y_, means, covariances, index, title):
    # Create a subplot for plotting
    splot = plt.subplot(2, 1, 1 + index)
    # Iterate through means, covariances, and colors
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # Compute eigenvalues and eigenvectors of covariance matrix
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # Skip plotting redundant components not used by Dirichlet Process
        if not np.any(Y_ == i):
            continue
        # Scatter plot of data points for each component
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # Set plot limits and labels
    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0.0, -0.1], [1.7, 0.4]])
X = np.r_[
    np.dot(np.random.randn(n_samples, 2), C),
    0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
]

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type="full").fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

# Fit a Dirichlet process Gaussian mixture using five components
# 使用贝叶斯高斯混合模型 (Bayesian Gaussian Mixture Model, BGMM) 进行拟合，其中包含5个高斯分量，并使用完全协方差矩阵类型
dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(X)

# 绘制贝叶斯高斯混合模型的聚类结果
plot_results(
    X,                               # 输入数据集
    dpgmm.predict(X),                # 预测数据点所属的类别
    dpgmm.means_,                    # 每个高斯分量的均值向量
    dpgmm.covariances_,              # 每个高斯分量的协方差矩阵
    1,                               # 图形显示参数，用于指定图形编号或其他配置
    "Bayesian Gaussian Mixture with a Dirichlet process prior",  # 图形标题
)

# 显示绘制的图形
plt.show()
```