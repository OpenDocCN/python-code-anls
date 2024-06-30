# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_compare_gpr_krr.py`

```
"""
==========================================================
Comparison of kernel ridge and Gaussian process regression
==========================================================

This example illustrates differences between a kernel ridge regression and a
Gaussian process regression.

Both kernel ridge regression and Gaussian process regression are using a
so-called "kernel trick" to make their models expressive enough to fit
the training data. However, the machine learning problems solved by the two
methods are drastically different.

Kernel ridge regression will find the target function that minimizes a loss
function (the mean squared error).

Instead of finding a single target function, the Gaussian process regression
employs a probabilistic approach : a Gaussian posterior distribution over
target functions is defined based on the Bayes' theorem, Thus prior
probabilities on target functions are being combined with a likelihood function
defined by the observed training data to provide estimates of the posterior
distributions.

We will illustrate these differences with an example and we will also focus on
tuning the kernel hyperparameters.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generating a dataset
# --------------------
#
# We create a synthetic dataset. The true generative process will take a 1-D
# vector and compute its sine. Note that the period of this sine is thus
# :math:`2 \pi`. We will reuse this information later in this example.
import numpy as np

rng = np.random.RandomState(0)  # 创建一个伪随机数生成器对象rng，种子为0
data = np.linspace(0, 30, num=1_000).reshape(-1, 1)  # 生成包含1000个数据点的一维数据，并转为列向量
target = np.sin(data).ravel()  # 计算data中每个元素的正弦值，并将结果展平成一维数组

# %%
# Now, we can imagine a scenario where we get observations from this true
# process. However, we will add some challenges:
#
# - the measurements will be noisy;
# - only samples from the beginning of the signal will be available.
training_sample_indices = rng.choice(np.arange(0, 400), size=40, replace=False)  # 从0到399中随机选择40个不重复的索引作为训练样本索引
training_data = data[training_sample_indices]  # 根据训练样本索引从data中获取训练数据
training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(
    len(training_sample_indices)
)  # 对应训练数据添加高斯噪声，噪声水平为标准正态分布乘以0.5

# %%
# Let's plot the true signal and the noisy measurements available for training.
import matplotlib.pyplot as plt

plt.plot(data, target, label="True signal", linewidth=2)  # 绘制真实信号的图像
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)  # 绘制带有噪声的训练数据散点图
plt.legend()  # 添加图例
plt.xlabel("data")  # x轴标签
plt.ylabel("target")  # y轴标签
_ = plt.title(
    "Illustration of the true generative process and \n"
    "noisy measurements available during training"
)  # 设置图表标题

# %%
# Limitations of a simple linear model
# ------------------------------------
#
# First, we would like to highlight the limitations of a linear model given
# our dataset. We fit a :class:`~sklearn.linear_model.Ridge` and check the
# predictions of this model on our dataset.
from sklearn.linear_model import Ridge

ridge = Ridge().fit(training_data, training_noisy_target)  # 使用Ridge回归拟合训练数据和目标值
plt.plot(data, target, label="True signal", linewidth=2)
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
plt.plot(data, ridge.predict(data), label="Ridge regression")
plt.legend()
plt.xlabel("data")
plt.ylabel("target")
_ = plt.title("Limitation of a linear model such as ridge")



# %%
# Such a ridge regressor underfits data since it is not expressive enough.
#
# Kernel methods: kernel ridge and Gaussian process
# -------------------------------------------------
#
# Kernel ridge
# ............
#
# We can make the previous linear model more expressive by using a so-called
# kernel. A kernel is an embedding from the original feature space to another
# one. Simply put, it is used to map our original data into a newer and more
# complex feature space. This new space is explicitly defined by the choice of
# kernel.
#
# In our case, we know that the true generative process is a periodic function.
# We can use a :class:`~sklearn.gaussian_process.kernels.ExpSineSquared` kernel
# which allows recovering the periodicity. The class
# :class:`~sklearn.kernel_ridge.KernelRidge` will accept such a kernel.
#
# Using this model together with a kernel is equivalent to embed the data
# using the mapping function of the kernel and then apply a ridge regression.
# In practice, the data are not mapped explicitly; instead the dot product
# between samples in the higher dimensional feature space is computed using the
# "kernel trick".
#
# Thus, let's use such a :class:`~sklearn.kernel_ridge.KernelRidge`.
import time

from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge

kernel_ridge = KernelRidge(kernel=ExpSineSquared())

start_time = time.time()
kernel_ridge.fit(training_data, training_noisy_target)
print(
    f"Fitting KernelRidge with default kernel: {time.time() - start_time:.3f} seconds"
)



# %%
plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
plt.plot(
    data,
    kernel_ridge.predict(data),
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
plt.legend(loc="lower right")
plt.xlabel("data")
plt.ylabel("target")
_ = plt.title(
    "Kernel ridge regression with an exponential sine squared\n "
    "kernel using default hyperparameters"
)



# %%
# This fitted model is not accurate. Indeed, we did not set the parameters of
# the kernel and instead used the default ones. We can inspect them.
kernel_ridge.kernel



# %%
# Our kernel has two parameters: the length-scale and the periodicity. For our
# dataset, we use `sin` as the generative process, implying a
# :math:`2 \pi`-periodicity for the signal. The default value of the parameter
# being :math:`1`, it explains the high frequency observed in the predictions of
# our model.
# %%
# 导入所需的库和模块
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

# 定义参数分布字典，包括 alpha 参数和 kernel 的两个参数：length_scale 和 periodicity
param_distributions = {
    "alpha": loguniform(1e0, 1e3),                   # alpha 参数的对数均匀分布
    "kernel__length_scale": loguniform(1e-2, 1e2),   # kernel 的 length_scale 参数的对数均匀分布
    "kernel__periodicity": loguniform(1e0, 1e1),     # kernel 的 periodicity 参数的对数均匀分布
}

# 创建 RandomizedSearchCV 对象，用于随机搜索最佳的 kernel ridge 模型参数
kernel_ridge_tuned = RandomizedSearchCV(
    kernel_ridge,                    # kernel ridge 模型
    param_distributions=param_distributions,   # 参数分布
    n_iter=500,                      # 迭代次数
    random_state=0,                  # 随机数种子
)

# 记录开始时间
start_time = time.time()

# 在训练数据上拟合模型，用于降噪的目标数据
kernel_ridge_tuned.fit(training_data, training_noisy_target)

# 输出模型拟合时间
print(f"Time for KernelRidge fitting: {time.time() - start_time:.3f} seconds")

# %%
# 现在模型拟合更加耗时，因为需要尝试多个超参数组合。
# 我们可以查看找到的最佳超参数，以便获取一些直觉。
kernel_ridge_tuned.best_params_

# %%
# 查看最佳参数时发现，它们与默认值不同。
# 我们还发现，周期性接近预期值：math:`2 \pi`。
# 现在我们可以检查调整后的 kernel ridge 模型的预测结果。
start_time = time.time()

# 对数据进行预测
predictions_kr = kernel_ridge_tuned.predict(data)

# 输出预测时间
print(f"Time for KernelRidge predict: {time.time() - start_time:.3f} seconds")

# %%
# 绘制图形显示真实信号、噪声测量点和 kernel ridge 拟合结果。
plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
plt.plot(
    data,
    predictions_kr,
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
plt.legend(loc="lower right")
plt.xlabel("data")
plt.ylabel("target")
_ = plt.title(
    "Kernel ridge regression with an exponential sine squared\n "
    "kernel using tuned hyperparameters"
)

# %%
# 现在我们获得了一个更加精确的模型。尽管如此，仍然可以观察到一些误差，主要是由于数据集中添加的噪声。
#
# Gaussian process regression
# ...........................
#
# 现在，我们将使用 :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
# 对相同的数据集进行拟合。在训练高斯过程时，内核的超参数在拟合过程中进行优化。
# 不需要外部超参数搜索。这里，我们创建一个比 kernel ridge 回归更复杂的内核：
# 添加了一个 :class:`~sklearn.gaussian_process.kernels.WhiteKernel`，用于估计数据集中的噪声。
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

# 定义高斯过程的内核，包括 ExpSineSquared 和 WhiteKernel
kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(
    1e-1
)

# 创建 GaussianProcessRegressor 对象，使用定义的内核
gaussian_process = GaussianProcessRegressor(kernel=kernel)

# 记录开始时间
start_time = time.time()

# 在训练数据上拟合高斯过程模型，用于降噪的目标数据
gaussian_process.fit(training_data, training_noisy_target)
# 输出训练高斯过程回归器所需的时间
print(
    f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds"
)

# %%
# 训练高斯过程的计算成本远低于使用随机搜索的核岭回归。我们可以检查已计算的内核参数。
gaussian_process.kernel_

# %%
# 确实，我们看到参数已经优化。观察 `periodicity` 参数，我们发现找到的周期接近于理论值 :math:`2 \pi`。
# 现在我们可以查看模型的预测结果。
start_time = time.time()
mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
    data,
    return_std=True,
)
# 输出使用高斯过程回归器进行预测所需的时间
print(
    f"Time for GaussianProcessRegressor predict: {time.time() - start_time:.3f} seconds"
)

# %%
# 绘制真实信号数据
plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
# 绘制训练数据的散点图
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
# 绘制核岭回归的预测结果
plt.plot(
    data,
    predictions_kr,
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
# 绘制高斯过程回归器的预测结果
plt.plot(
    data,
    mean_predictions_gpr,
    label="Gaussian process regressor",
    linewidth=2,
    linestyle="dotted",
)
# 使用绿色填充高斯过程回归器预测的置信区间
plt.fill_between(
    data.ravel(),
    mean_predictions_gpr - std_predictions_gpr,
    mean_predictions_gpr + std_predictions_gpr,
    color="tab:green",
    alpha=0.2,
)
plt.legend(loc="lower right")
plt.xlabel("data")
plt.ylabel("target")
_ = plt.title("Comparison between kernel ridge and gaussian process regressor")

# %%
# 我们观察到核岭回归和高斯过程回归器的结果接近。然而，高斯过程回归器还提供了不可用于核岭回归的不确定性信息。
# 由于目标函数的概率形式，高斯过程可以输出均值预测以及标准差（或协方差）。
#
# 然而，这也带来了一个代价：使用高斯过程进行预测所需的时间更长。
#
# 最终结论
# ----------------
#
# 我们可以对这两种模型在外推能力方面给出最终结论。事实上，我们只提供了信号的起始部分作为训练集。
# 使用周期内核强制我们的模型重复训练集上找到的模式。利用这个内核信息以及两种模型的外推能力，我们观察到这些模型将继续预测正弦模式。
#
# 高斯过程允许将多个内核组合在一起。因此，我们可以将指数正弦平方内核与径向基函数内核结合起来。
from sklearn.gaussian_process.kernels import RBF

# 创建高斯过程回归器的内核，结合指数正弦平方内核与径向基函数内核
kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(
    length_scale=15, length_scale_bounds="fixed"
)
# 定义一个带有高斯核和白噪声的核函数
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(1e-1)
# 创建一个高斯过程回归器，使用上面定义的核函数
gaussian_process = GaussianProcessRegressor(kernel=kernel)
# 使用训练数据拟合高斯过程回归器
gaussian_process.fit(training_data, training_noisy_target)
# 对给定数据进行预测，返回预测均值和标准差
mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
    data,
    return_std=True,
)

# %%
# 绘制真实信号的图像，用虚线表示
plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
# 绘制训练数据点，用黑色散点表示
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
# 绘制核岭回归的预测结果，用长虚线表示
plt.plot(
    data,
    predictions_kr,
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
# 绘制高斯过程回归的预测结果，用点线表示
plt.plot(
    data,
    mean_predictions_gpr,
    label="Gaussian process regressor",
    linewidth=2,
    linestyle="dotted",
)
# 用绿色填充高斯过程回归的预测标准差范围
plt.fill_between(
    data.ravel(),
    mean_predictions_gpr - std_predictions_gpr,
    mean_predictions_gpr + std_predictions_gpr,
    color="tab:green",
    alpha=0.2,
)
# 添加图例，放置在右下角
plt.legend(loc="lower right")
# 添加 x 轴标签
plt.xlabel("data")
# 添加 y 轴标签
plt.ylabel("target")
# 设置图表标题
_ = plt.title("Effect of using a radial basis function kernel")

# %%
# 当训练样本不可用时，使用径向基函数核的效果会减弱周期性效应。
# 随着测试样本远离训练样本，预测结果逐渐收敛到它们的均值，同时它们的标准差也会增加。
```