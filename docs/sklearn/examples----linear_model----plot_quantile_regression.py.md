# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_quantile_regression.py`

```
"""
===================
Quantile regression
===================

This example illustrates how quantile regression can predict non-trivial
conditional quantiles.

The left figure shows the case when the error distribution is normal,
but has non-constant variance, i.e. with heteroscedasticity.

The right figure shows an example of an asymmetric error distribution,
namely the Pareto distribution.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Dataset generation
# ------------------
#
# To illustrate the behaviour of quantile regression, we will generate two
# synthetic datasets. The true generative random processes for both datasets
# will be composed by the same expected value with a linear relationship with a
# single feature `x`.
import numpy as np

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.RandomState(42)
# Generate evenly spaced values from 0 to 10 with 100 points
x = np.linspace(start=0, stop=10, num=100)
# Reshape `x` to be a column vector
X = x[:, np.newaxis]
# Calculate true values of `y` based on a linear relationship with `x`
y_true_mean = 10 + 0.5 * x

# %%
# We will create two subsequent problems by changing the distribution of the
# target `y` while keeping the same expected value:
#
# - in the first case, a heteroscedastic Normal noise is added;
# - in the second case, an asymmetric Pareto noise is added.
y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])
a = 5
y_pareto = y_true_mean + 10 * (rng.pareto(a, size=x.shape[0]) - 1 / (a - 1))

# %%
# Let's first visualize the datasets as well as the distribution of the
# residuals `y - mean(y)`.
import matplotlib.pyplot as plt

# Create subplots for visualization
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11), sharex="row", sharey="row")

# Plotting for the dataset with heteroscedastic Normal distributed targets
axs[0, 0].plot(x, y_true_mean, label="True mean")
axs[0, 0].scatter(x, y_normal, color="black", alpha=0.5, label="Observations")
axs[1, 0].hist(y_true_mean - y_normal, edgecolor="black")

# Plotting for the dataset with asymmetric Pareto distributed targets
axs[0, 1].plot(x, y_true_mean, label="True mean")
axs[0, 1].scatter(x, y_pareto, color="black", alpha=0.5, label="Observations")
axs[1, 1].hist(y_true_mean - y_pareto, edgecolor="black")

# Titles and labels for the subplots
axs[0, 0].set_title("Dataset with heteroscedastic Normal distributed targets")
axs[0, 1].set_title("Dataset with asymmetric Pareto distributed target")
axs[1, 0].set_title(
    "Residuals distribution for heteroscedastic Normal distributed targets"
)
axs[1, 1].set_title("Residuals distribution for asymmetric Pareto distributed target")
axs[0, 0].legend()
axs[0, 1].legend()
axs[0, 0].set_ylabel("y")
axs[1, 0].set_ylabel("Counts")
axs[0, 1].set_xlabel("x")
axs[0, 0].set_xlabel("x")
axs[1, 0].set_xlabel("Residuals")
_ = axs[1, 1].set_xlabel("Residuals")

# %%
# With the heteroscedastic Normal distributed target, we observe that the
# variance of the noise is increasing when the value of the feature `x` is
# increasing.
#
# With the asymmetric Pareto distributed target, we observe that the positive
# residuals are bounded.
#
# These types of noisy targets make the estimation via
# :class:`~sklearn.linear_model.LinearRegression` less efficient, i.e. we need
# more data to get stable results and, in addition, large outliers can have a
# huge impact on the fitted coefficients. (Stated otherwise: in a setting with
# constant variance, ordinary least squares estimators converge much faster to
# the *true* coefficients with increasing sample size.)
#
# In this asymmetric setting, the median or different quantiles give additional
# insights. On top of that, median estimation is much more robust to outliers
# and heavy tailed distributions. But note that extreme quantiles are estimated
# by very few data points. 95% quantile are more or less estimated by the 5%
# largest values and thus also a bit sensitive outliers.
#
# In the remainder of this tutorial, we will show how
# :class:`~sklearn.linear_model.QuantileRegressor` can be used in practice and
# give the intuition into the properties of the fitted models. Finally,
# we will compare the both :class:`~sklearn.linear_model.QuantileRegressor`
# and :class:`~sklearn.linear_model.LinearRegression`.
#
# Fitting a `QuantileRegressor`
# -----------------------------
#
# In this section, we want to estimate the conditional median as well as
# a low and high quantile fixed at 5% and 95%, respectively. Thus, we will get
# three linear models, one for each quantile.
#
# We will use the quantiles at 5% and 95% to find the outliers in the training
# sample beyond the central 90% interval.
from sklearn.utils.fixes import parse_version, sp_version

# This is line is to avoid incompatibility if older SciPy version.
# You should use `solver="highs"` with recent version of SciPy.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

# %%
from sklearn.linear_model import QuantileRegressor

# List of quantiles for which we want to estimate linear models
quantiles = [0.05, 0.5, 0.95]

# Dictionary to store predictions for each quantile
predictions = {}

# Boolean array to track predictions outside the central 90% interval
out_bounds_predictions = np.zeros_like(y_true_mean, dtype=np.bool_)

# Loop over each quantile
for quantile in quantiles:
    # Initialize QuantileRegressor for the current quantile
    qr = QuantileRegressor(quantile=quantile, alpha=0, solver=solver)
    
    # Fit the QuantileRegressor model and predict on the data
    y_pred = qr.fit(X, y_normal).predict(X)
    
    # Store the predictions in the dictionary
    predictions[quantile] = y_pred

    # Update the out_bounds_predictions based on quantile
    if quantile == min(quantiles):
        out_bounds_predictions = np.logical_or(
            out_bounds_predictions, y_pred >= y_normal
        )
    elif quantile == max(quantiles):
        out_bounds_predictions = np.logical_or(
            out_bounds_predictions, y_pred <= y_normal
        )

# %%
# Now, we can plot the three linear models and the distinguished samples that
# are within the central 90% interval from samples that are outside this
# interval.
plt.plot(X, y_true_mean, color="black", linestyle="dashed", label="True mean")

# Plot each quantile's linear model
for quantile, y_pred in predictions.items():
    plt.plot(X, y_pred, label=f"Quantile: {quantile}")

# Scatter plot for samples outside the central 90% interval
plt.scatter(
    x[out_bounds_predictions],
    y_normal[out_bounds_predictions],
    color="black",
    marker="+",
    alpha=0.5,
    label="Outside interval",
)

# Scatter plot for samples inside the central 90% interval
plt.scatter(
    x[~out_bounds_predictions],
    y_normal[~out_bounds_predictions],
    color="black",
    alpha=0.5,
    label="Inside interval",
)

# Add legend to the plot
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
_ = plt.title("Quantiles of heteroscedastic Normal distributed target")

# 设置横坐标标签为"x"
# 设置纵坐标标签为"y"
# 设置图表标题为"Quantiles of heteroscedastic Normal distributed target"

# %%
# 由于噪声仍然是正态分布的，特别是对称的，
# 真实条件均值和真实条件中位数是相等的。事实上，
# 我们看到估计的中位数几乎等于真实均值。我们观察到
# 噪声方差增加对于5%和95%分位数的影响：
# 这些分位数的斜率非常不同，并且它们之间的间隔随着"x"的增加而变宽。
#
# 为了进一步理解5%和95%分位数估计器的含义，可以统计落在
# 预测分位数上方和下方的样本数量（在上图中用十字表示），考虑到我们总共有100个样本。
#
# 我们可以使用非对称的Pareto分布目标重复相同的实验。
quantiles = [0.05, 0.5, 0.95]
predictions = {}
out_bounds_predictions = np.zeros_like(y_true_mean, dtype=np.bool_)
for quantile in quantiles:
    qr = QuantileRegressor(quantile=quantile, alpha=0, solver=solver)
    y_pred = qr.fit(X, y_pareto).predict(X)
    predictions[quantile] = y_pred

    if quantile == min(quantiles):
        out_bounds_predictions = np.logical_or(
            out_bounds_predictions, y_pred >= y_pareto
        )
    elif quantile == max(quantiles):
        out_bounds_predictions = np.logical_or(
            out_bounds_predictions, y_pred <= y_pareto
        )

# %%
# 在这里画出真实均值的黑色虚线
# 对于每个分位数，画出对应的预测曲线
# 标记超出区间的点为黑色十字，表示"Outside interval"
# 标记未超出区间的点为黑色圆点，表示"Inside interval"
plt.plot(X, y_true_mean, color="black", linestyle="dashed", label="True mean")

for quantile, y_pred in predictions.items():
    plt.plot(X, y_pred, label=f"Quantile: {quantile}")

plt.scatter(
    x[out_bounds_predictions],
    y_pareto[out_bounds_predictions],
    color="black",
    marker="+",
    alpha=0.5,
    label="Outside interval",
)
plt.scatter(
    x[~out_bounds_predictions],
    y_pareto[~out_bounds_predictions],
    color="black",
    alpha=0.5,
    label="Inside interval",
)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
_ = plt.title("Quantiles of asymmetric Pareto distributed target")


# %%
# 由于噪声分布的不对称性，我们观察到真实均值和估计的条件中位数是不同的。
# 我们还观察到每个分位数模型具有不同的参数，以更好地拟合所需的分位数。
# 注意，在这种情况下，理想情况下，所有分位数应该是平行的，
# 这在数据点更多或者分位数不那么极端（例如10%和90%）时会更为明显。
#
# 比较`QuantileRegressor`和`LinearRegression`
# ----------------------------------------------------
#
# 在本节中，我们将详细讨论:class:`~sklearn.linear_model.QuantileRegressor`
# 和:class:`~sklearn.linear_model.LinearRegression`在最小化的误差方面的差异。
#
# 实际上，:class:`~sklearn.linear_model.LinearRegression`是一种最小二乘方法，
# 最小化训练和
# 预测目标。相比之下，
# :class:`~sklearn.linear_model.QuantileRegressor` 使用 `quantile=0.5`
# 最小化平均绝对误差（MAE）。
#
# 让我们首先计算这些模型的训练误差，即均方误差和平均绝对误差。我们将使用不对称帕累托分布的目标，使其更有趣，因为平均值和中位数不相等。
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 创建线性回归模型对象
linear_regression = LinearRegression()
# 创建分位数回归模型对象，设置quantile=0.5，alpha=0，使用给定的solver
quantile_regression = QuantileRegressor(quantile=0.5, alpha=0, solver=solver)

# 使用线性回归模型在训练集上进行拟合和预测
y_pred_lr = linear_regression.fit(X, y_pareto).predict(X)
# 使用分位数回归模型在训练集上进行拟合和预测
y_pred_qr = quantile_regression.fit(X, y_pareto).predict(X)

# 打印训练误差（样本内表现）
print(
    f"""Training error (in-sample performance)
    {linear_regression.__class__.__name__}:
    MAE = {mean_absolute_error(y_pareto, y_pred_lr):.3f}
    MSE = {mean_squared_error(y_pareto, y_pred_lr):.3f}
    {quantile_regression.__class__.__name__}:
    MAE = {mean_absolute_error(y_pareto, y_pred_qr):.3f}
    MSE = {mean_squared_error(y_pareto, y_pred_qr):.3f}
    """
)

# %%
# 在训练集上，我们看到:class:`~sklearn.linear_model.QuantileRegressor`的MAE比
# :class:`~sklearn.linear_model.LinearRegression`低。相反，MSE对于
# :class:`~sklearn.linear_model.LinearRegression`而言更低。这些结果证实了
# :class:`~sklearn.linear_model.QuantileRegressor`最小化的是MAE，而
# :class:`~sklearn.linear_model.LinearRegression`最小化的是MSE。
#
# 我们可以通过观察交叉验证获得类似的评估结果来进行类似的评估。
from sklearn.model_selection import cross_validate

# 使用交叉验证计算线性回归模型的测试误差
cv_results_lr = cross_validate(
    linear_regression,
    X,
    y_pareto,
    cv=3,
    scoring=["neg_mean_absolute_error", "neg_mean_squared_error"],
)
# 使用交叉验证计算分位数回归模型的测试误差
cv_results_qr = cross_validate(
    quantile_regression,
    X,
    y_pareto,
    cv=3,
    scoring=["neg_mean_absolute_error", "neg_mean_squared_error"],
)
# 打印测试误差（交叉验证性能）
print(
    f"""Test error (cross-validated performance)
    {linear_regression.__class__.__name__}:
    MAE = {-cv_results_lr["test_neg_mean_absolute_error"].mean():.3f}
    MSE = {-cv_results_lr["test_neg_mean_squared_error"].mean():.3f}
    {quantile_regression.__class__.__name__}:
    MAE = {-cv_results_qr["test_neg_mean_absolute_error"].mean():.3f}
    MSE = {-cv_results_qr["test_neg_mean_squared_error"].mean():.3f}
    """
)

# %%
# 我们在样本外评估上得出类似的结论。
```