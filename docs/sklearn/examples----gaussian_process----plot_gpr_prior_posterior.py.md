# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpr_prior_posterior.py`

```
"""
==========================================================================
Illustration of prior and posterior Gaussian process for different kernels
==========================================================================

This example illustrates the prior and posterior of a
:class:`~sklearn.gaussian_process.GaussianProcessRegressor` with different
kernels. Mean, standard deviation, and 5 samples are shown for both prior
and posterior distributions.

Here, we only give some illustration. To know more about kernels' formulation,
refer to the :ref:`User Guide <gp_kernels>`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Helper function
# ---------------
#
# Before presenting each individual kernel available for Gaussian processes,
# we will define an helper function allowing us plotting samples drawn from
# the Gaussian process.
#
# This function will take a
# :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model and will
# drawn sample from the Gaussian process. If the model was not fit, the samples
# are drawn from the prior distribution while after model fitting, the samples are
# drawn from the posterior distribution.
import matplotlib.pyplot as plt
import numpy as np

# Define a function to plot samples from a Gaussian process model
def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    # Generate a set of input points
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    # Predict mean and standard deviation of the Gaussian process at input points
    y_mean, y_std = gpr_model.predict(X, return_std=True)

    # Draw samples from the Gaussian process
    y_samples = gpr_model.sample_y(X, n_samples)

    # Plot each sampled function
    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )

    # Plot the mean of the Gaussian process
    ax.plot(x, y_mean, color="black", label="Mean")

    # Plot the 1 standard deviation interval around the mean
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )

    # Set labels for the axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set y-axis limits
    ax.set_ylim([-3, 3])


# %%
# Dataset and Gaussian process generation
# ---------------------------------------
# We will create a training dataset that we will use in the different sections.
rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5

# %%
# Kernel cookbook
# ---------------
#
# Radial Basis Function kernel
# ............................
# 导入所需的类和函数
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# 创建 RBF 核函数对象，并设置长度尺度及其范围
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

# 创建高斯过程回归器对象，使用上面定义的核函数
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 创建包含两个子图的图表对象
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# 绘制先验分布的样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# 绘制后验分布的样本
# 根据训练数据拟合高斯过程回归模型
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# 在后验分布图上绘制观测点
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

# 设置整体图的标题
fig.suptitle("Radial Basis Function kernel", fontsize=18)
# 调整布局以确保子图之间的适当间距
plt.tight_layout()

# %%
# 打印在拟合之前的核函数参数信息
print(f"Kernel parameters before fit:\n{kernel})")
# 打印拟合后的核函数参数信息以及对数似然值
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# Rational Quadradtic kernel
# ..........................
# 导入 RationalQuadratic 核函数类
from sklearn.gaussian_process.kernels import RationalQuadratic

# 创建 Rational Quadratic 核函数对象，并设置长度尺度、alpha 参数及其范围
kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))

# 创建新的高斯过程回归器对象，使用上述定义的核函数
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 创建包含两个子图的图表对象
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# 绘制先验分布的样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# 绘制后验分布的样本
# 根据训练数据拟合高斯过程回归模型
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# 在后验分布图上绘制观测点
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

# 设置整体图的标题
fig.suptitle("Rational Quadratic kernel", fontsize=18)
# 调整布局以确保子图之间的适当间距
plt.tight_layout()

# %%
# 打印在拟合之前的核函数参数信息
print(f"Kernel parameters before fit:\n{kernel})")
# 打印拟合后的核函数参数信息以及对数似然值
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# Exp-Sine-Squared kernel
# .......................
# 导入 ExpSineSquared 核函数类
from sklearn.gaussian_process.kernels import ExpSineSquared

# 创建 Exp-Sine-Squared 核函数对象，并设置长度尺度、周期性参数及其范围
kernel = 1.0 * ExpSineSquared(
    length_scale=1.0,
    periodicity=3.0,
    length_scale_bounds=(0.1, 10.0),
    periodicity_bounds=(1.0, 10.0),
)

# 创建新的高斯过程回归器对象，使用上述定义的核函数
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 创建包含两个子图的图表对象
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# 绘制先验分布的样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# 绘制后验分布的样本
# 根据训练数据拟合高斯过程回归模型
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# 在后验分布图上绘制观测点
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")

# %%
# 设置第二个子图的图例位置，以及相对于坐标轴的位置
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
# 设置第二个子图的标题
axs[1].set_title("Samples from posterior distribution")

# 设置整个图形的总标题
fig.suptitle("Exp-Sine-Squared kernel", fontsize=18)
# 调整子图之间的布局，使它们更加紧凑
plt.tight_layout()

# %%
# 打印拟合前的核函数参数
print(f"Kernel parameters before fit:\n{kernel})")
# 打印拟合后的核函数参数和对数似然值
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# Dot-product kernel
# ..................
# 导入所需的核函数模块
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct

# 定义核函数为常数核乘以点积核的平方
kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
    DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
)
# 创建高斯过程回归器对象，使用定义的核函数
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 创建包含两个子图的图形对象
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# 绘制先验分布样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
# 设置第一个子图的标题
axs[0].set_title("Samples from prior distribution")

# 拟合高斯过程回归模型
gpr.fit(X_train, y_train)
# 绘制后验分布样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# 绘制观测数据点
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
# 设置第二个子图的图例位置和标题
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

# 设置整个图形的总标题
fig.suptitle("Dot-product kernel", fontsize=18)
# 调整子图之间的布局，使它们更加紧凑
plt.tight_layout()

# %%
# 打印拟合前的核函数参数
print(f"Kernel parameters before fit:\n{kernel})")
# 打印拟合后的核函数参数和对数似然值
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# Matérn kernel
# ..............
# 导入所需的核函数模块
from sklearn.gaussian_process.kernels import Matern

# 定义核函数为 Matérn 核
kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
# 创建高斯过程回归器对象，使用定义的核函数
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 创建包含两个子图的图形对象
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# 绘制先验分布样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
# 设置第一个子图的标题
axs[0].set_title("Samples from prior distribution")

# 拟合高斯过程回归模型
gpr.fit(X_train, y_train)
# 绘制后验分布样本
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# 绘制观测数据点
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
# 设置第二个子图的图例位置和标题
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

# 设置整个图形的总标题
fig.suptitle("Matérn kernel", fontsize=18)
# 调整子图之间的布局，使它们更加紧凑
plt.tight_layout()

# %%
# 打印拟合前的核函数参数
print(f"Kernel parameters before fit:\n{kernel})")
# 打印拟合后的核函数参数和对数似然值
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)
```