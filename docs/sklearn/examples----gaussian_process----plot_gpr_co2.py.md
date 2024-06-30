# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpr_co2.py`

```
"""
====================================================================================
Forecasting of CO2 level on Mona Loa dataset using Gaussian process regression (GPR)
====================================================================================

This example is based on Section 5.4.3 of "Gaussian Processes for Machine
Learning" [1]_. It illustrates an example of complex kernel engineering
and hyperparameter optimization using gradient ascent on the
log-marginal-likelihood. The data consists of the monthly average atmospheric
CO2 concentrations (in parts per million by volume (ppm)) collected at the
Mauna Loa Observatory in Hawaii, between 1958 and 2001. The objective is to
model the CO2 concentration as a function of the time :math:`t` and extrapolate
for years after 2001.

.. rubric:: References

.. [1] `Rasmussen, Carl Edward. "Gaussian processes in machine learning."
    Summer school on machine learning. Springer, Berlin, Heidelberg, 2003
    <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_.
"""

# 打印示例的文档字符串
print(__doc__)

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Build the dataset
# -----------------
#
# We will derive a dataset from the Mauna Loa Observatory that collected air
# samples. We are interested in estimating the concentration of CO2 and
# extrapolate it for further year. First, we load the original dataset available
# in OpenML as a pandas dataframe. This will be replaced with Polars
# once `fetch_openml` adds a native support for it.
from sklearn.datasets import fetch_openml

# 使用 fetch_openml 函数从 OpenML 加载数据集
co2 = fetch_openml(data_id=41187, as_frame=True)

# 显示数据集的前几行
co2.frame.head()

# %%
# First, we process the original dataframe to create a date column and select
# it along with the CO2 column.
import polars as pl

# 创建一个 Polars 的 DataFrame，并选取包含年份、月份、日期和 CO2 浓度的列
co2_data = pl.DataFrame(co2.frame[["year", "month", "day", "co2"]]).select(
    pl.date("year", "month", "day"), "co2"
)
co2_data.head()

# %%
# 找出日期列的最小值和最大值
co2_data["date"].min(), co2_data["date"].max()

# %%
# We see that we get CO2 concentration for some days from March, 1958 to
# December, 2001. We can plot these raw information to have a better
# understanding.
import matplotlib.pyplot as plt

# 绘制原始数据的 CO2 浓度随时间变化的图像
plt.plot(co2_data["date"], co2_data["co2"])
plt.xlabel("date")
plt.ylabel("CO$_2$ concentration (ppm)")
_ = plt.title("Raw air samples measurements from the Mauna Loa Observatory")

# %%
# We will preprocess the dataset by taking a monthly average and drop month
# for which no measurements were collected. Such a processing will have an
# smoothing effect on the data.

# 对数据集进行预处理，计算每月的 CO2 浓度平均值，并且删除没有收集测量数据的月份
co2_data = (
    co2_data.sort(by="date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("co2").mean())
    .drop_nulls()
)

# 绘制经过平均处理后的数据的 CO2 浓度随时间变化的图像
plt.plot(co2_data["date"], co2_data["co2"])
plt.xlabel("date")
plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
_ = plt.title(
    "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
)

# %%
# The idea in this example will be to predict the CO2 concentration in function
# 将日期数据转换为数值，作为特征向量 X
X = co2_data.select(
    pl.col("date").dt.year() + pl.col("date").dt.month / 12
).to_numpy()

# 提取二氧化碳浓度作为目标向量 y
y = co2_data["co2"].to_numpy()

# %%
# 设计适当的核函数
# ------------------------
#
# 为了设计与高斯过程配合使用的核函数，我们可以根据数据的特点进行一些假设。
# 我们观察到数据具有几个特征：长期上升趋势、显著的季节性变化和一些较小的不规则性。
# 我们可以使用不同的适当核函数来捕捉这些特征。
#
# 首先，长期上升趋势可以使用径向基函数（RBF）核，其具有较大的长度尺度参数。
# RBF 核的较大长度尺度使得这个组成部分变得平滑。不强制趋势增长，为我们的模型提供自由度。
# 具体的长度尺度和振幅是自由超参数。
from sklearn.gaussian_process.kernels import RBF

long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)

# %%
# 季节性变化由固定周期为1年的周期指数正弦平方核解释。控制其平滑度的周期性成分的长度尺度是自由参数。
# 为了允许远离精确周期性的衰减，与 RBF 核的乘积被采用。RBF 组件的长度尺度控制衰减时间，也是进一步的自由参数。
# 这种核函数也称为局部周期核。
from sklearn.gaussian_process.kernels import ExpSineSquared

seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)

# %%
# 小的不规则性可以通过有理二次核组件解释，其长度尺度和 alpha 参数（量化长度尺度的扩散性）需确定。
# 有理二次核等效于具有多个长度尺度的 RBF 核，将更好地适应不同的不规则性。
from sklearn.gaussian_process.kernels import RationalQuadratic

irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

# %%
# 最后，数据集中的噪声可以通过核函数来解释，该核函数包含 RBF 核贡献部分，用于解释诸如局部天气现象之类的相关噪声组成，
# 以及白噪声的白核贡献部分。相对振幅和 RBF 的长度尺度是进一步的自由参数。
from sklearn.gaussian_process.kernels import WhiteKernel

noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)

# %%
# Thus, our final kernel is an addition of all previous kernel.
co2_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
co2_kernel



# %%
# Model fitting and extrapolation
# -------------------------------
#
# Now, we are ready to use a Gaussian process regressor and fit the available
# data. To follow the example from the literature, we will subtract the mean
# from the target. We could have used `normalize_y=True`. However, doing so
# would have also scaled the target (dividing `y` by its standard deviation).
# Thus, the hyperparameters of the different kernel would have had different
# meaning since they would not have been expressed in ppm.
from sklearn.gaussian_process import GaussianProcessRegressor

# Calculate the mean of y values
y_mean = y.mean()

# Initialize Gaussian Process Regressor with the composed kernel
gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False)

# Fit the Gaussian Process model to X and y with mean-subtracted y
gaussian_process.fit(X, y - y_mean)



# %%
# Now, we will use the Gaussian process to predict on:
#
# - training data to inspect the goodness of fit;
# - future data to see the extrapolation done by the model.
#
# Thus, we create synthetic data from 1958 to the current month. In addition,
# we need to add the subtracted mean computed during training.
import datetime
import numpy as np

# Get current date and month
today = datetime.datetime.now()
current_month = today.year + today.month / 12

# Generate synthetic test data from 1958 to current month
X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)

# Predict mean and standard deviation using Gaussian process
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)

# Add back the mean that was subtracted during training
mean_y_pred += y_mean



# %%
# Plotting the results
import matplotlib.pyplot as plt

# Plot measured data
plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")

# Plot Gaussian process prediction with uncertainty
plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
plt.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2,
)

# Add legend and labels
plt.legend()
plt.xlabel("Year")
plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
_ = plt.title(
    "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
)



# %%
# Our fitted model is capable to fit previous data properly and extrapolate to
# future year with confidence.
#
# Interpretation of kernel hyperparameters
# ----------------------------------------
#
# Now, we can have a look at the hyperparameters of the kernel.
gaussian_process.kernel_



# %%
# Thus, most of the target signal, with the mean subtracted, is explained by a
# long-term rising trend for ~45 ppm and a length-scale of ~52 years. The
# periodic component has an amplitude of ~2.6ppm, a decay time of ~90 years and
# a length-scale of ~1.5. The long decay time indicates that we have a
# component very close to a seasonal periodicity. The correlated noise has an
# amplitude of ~0.2 ppm with a length scale of ~0.12 years and a white-noise
# contribution of ~0.04 ppm. Thus, the overall noise level is very small,
# indicating that the data can be very well explained by the model.
```