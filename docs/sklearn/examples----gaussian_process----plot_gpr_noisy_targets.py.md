# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpr_noisy_targets.py`

```
"""
=========================================================
Gaussian Processes regression: basic introductory example
=========================================================

A simple one-dimensional regression example computed in two different ways:

1. A noise-free case
2. A noisy case with known noise-level per datapoint

In both cases, the kernel's parameters are estimated using the maximum
likelihood principle.

The figures illustrate the interpolating property of the Gaussian Process model
as well as its probabilistic nature in the form of a pointwise 95% confidence
interval.

Note that `alpha` is a parameter to control the strength of the Tikhonov
regularization on the assumed training points' covariance matrix.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Dataset generation
# ------------------
#
# We will start by generating a synthetic dataset. The true generative process
# is defined as :math:`f(x) = x \sin(x)`.
import numpy as np

# Generate evenly spaced values from 0 to 10 as input features X
X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)

# Compute the corresponding output values y using the generative process f(x) = x * sin(x)
y = np.squeeze(X * np.sin(X))

# %%
import matplotlib.pyplot as plt

# Plot the generated dataset
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")

# %%
# We will use this dataset in the next experiment to illustrate how Gaussian
# Process regression is working.
#
# Example with noise-free target
# ------------------------------
#
# In this first example, we will use the true generative process without
# adding any noise. For training the Gaussian Process regression, we will only
# select few samples.
rng = np.random.RandomState(1)
# Randomly choose 6 indices from the range of y.size without replacement
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
# Select corresponding X and y values for training
X_train, y_train = X[training_indices], y[training_indices]

# %%
# Now, we fit a Gaussian process on these few training data samples. We will
# use a radial basis function (RBF) kernel and a constant parameter to fit the
# amplitude.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Define the kernel for Gaussian Process regression using RBF kernel with specified parameters
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# Create a Gaussian Process Regressor object with the defined kernel and optimizer restarts
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# Fit the Gaussian Process model to the training data
gaussian_process.fit(X_train, y_train)
# Output the optimized kernel parameters
gaussian_process.kernel_

# %%
# After fitting our model, we see that the hyperparameters of the kernel have
# been optimized. Now, we will use our kernel to compute the mean prediction
# of the full dataset and plot the 95% confidence interval.
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# Plot the true generative process
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
# Plot the training observations
plt.scatter(X_train, y_train, label="Observations")
# Plot the mean prediction of the Gaussian Process
plt.plot(X, mean_prediction, label="Mean prediction")
# Plot the 95% confidence interval around the mean prediction
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
# 设置 y 轴的标签为 "$f(x)$"
plt.ylabel("$f(x)$")
# 设置图表标题为 "Gaussian process regression on noise-free dataset"
_ = plt.title("Gaussian process regression on noise-free dataset")

# %%
# 对于接近训练集中数据点的预测，95% 置信区间的振幅较小。当样本远离训练数据时，我们的模型预测不太准确，
# 并且模型的预测不太精确（不确定性较高）。
#
# Example with noisy targets
# --------------------------
#
# 我们可以重复类似的实验，这次给目标添加额外的噪声。这将展示噪声对拟合模型的影响。
#
# 我们给目标添加一个具有任意标准偏差的随机高斯噪声。
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

# %%
# 我们创建一个类似的高斯过程模型。除了核函数外，这次我们指定参数 `alpha`，它可以解释为高斯噪声的方差。
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %%
# 如前所述，绘制平均预测值和不确定性区域。
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on a noisy dataset")

# %%
# 噪声影响接近训练样本的预测：在训练样本附近，预测的不确定性较大，因为我们显式地建模了给定水平目标噪声，与输入变量无关。
```