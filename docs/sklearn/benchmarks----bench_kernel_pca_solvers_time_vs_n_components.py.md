# `D:\src\scipysrc\scikit-learn\benchmarks\bench_kernel_pca_solvers_time_vs_n_components.py`

```
# =============================================================
# Kernel PCA Solvers comparison benchmark: time vs n_components
# =============================================================

# This benchmark compares the performance of different solvers in Kernel PCA
# when varying the number of principal components (`n_components`). It demonstrates
# that approximate solvers can significantly speed up execution, especially
# when fewer principal components are sufficient.

# Description:
# ------------
# Generates a fixed number of training (default: 2000) and test (default: 1000)
# samples with 2 features using the `make_circles` helper method.

# KernelPCA models are trained on the training set with an increasing number of
# principal components, ranging from 1 to `max_n_compo` (default: 1999), at
# `n_compo_grid_size` positions (default: 10). For each `n_components` value,
# KernelPCA models are trained using various `eigen_solver` options. The execution
# times are then plotted at the end of the experiment.

# Observations:
# -------------
# Dense solver takes more time when the number of principal components is small,
# whereas the randomized method provides similar results with shorter execution times.

# Further Exploration:
# --------------------
# You can adjust `max_n_compo` and `n_compo_grid_size` to explore different
# ranges of `n_components`.

# Setting `arpack_all=True` activates the arpack solver for a large number of
# components, but this increases computation time.

# Authors: Sylvain MARIE, Schneider Electric

import time  # Importing the time module for measuring execution times

import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing numpy for numerical operations
from numpy.testing import assert_array_almost_equal  # Importing testing utilities from numpy

from sklearn.datasets import make_circles  # Importing make_circles dataset generator
from sklearn.decomposition import KernelPCA  # Importing KernelPCA from decomposition in sklearn

print(__doc__)  # Print the module-level docstring defined at the beginning

# 1- Design the Experiment
# ------------------------
n_train, n_test = 2000, 1000  # Define number of training and test samples
max_n_compo = 1999  # Maximum number of principal components to try
n_compo_grid_size = 10  # Number of positions in the grid to try

# Generate a grid of `n_compo_range` using exponential spacing
n_compo_range = [
    np.round(np.exp((x / (n_compo_grid_size - 1)) * np.log(max_n_compo)))
    for x in range(0, n_compo_grid_size)
]

n_iter = 3  # Number of times to repeat each experiment
arpack_all = False  # Whether to use arpack solver for all `n_compo` values

# 2- Generate random data
# -----------------------
n_features = 2
# Generate `make_circles` dataset with specified parameters
X, y = make_circles(
    n_samples=(n_train + n_test), factor=0.3, noise=0.05, random_state=0
)
X_train, X_test = X[:n_train, :], X[n_train:, :]  # Split data into training and test sets

# 3- Benchmark
# ------------
# Initialize arrays to store execution times for different solvers
ref_time = np.empty((len(n_compo_range), n_iter)) * np.nan
a_time = np.empty((len(n_compo_range), n_iter)) * np.nan
r_time = np.empty((len(n_compo_range), n_iter)) * np.nan

# Iterate over each `n_components` value in `n_compo_range` for benchmarking
for j, n_components in enumerate(n_compo_range):
    n_components = int(n_components)  # Convert `n_components` to integer
    print("Performing kPCA with n_components = %i" % n_components)

    # A- reference (dense)
    # 打印信息，指示正在使用密集求解器
    print("  - dense solver")
    # 对每个迭代次数执行以下操作
    for i in range(n_iter):
        # 记录开始时间
        start_time = time.perf_counter()
        # 使用密集求解器对训练数据进行 KernelPCA 拟合并对测试数据进行转换
        ref_pred = (
            KernelPCA(n_components, eigen_solver="dense").fit(X_train).transform(X_test)
        )
        # 记录运行时间并存储到数组中
        ref_time[j, i] = time.perf_counter() - start_time

    # 如果要求使用 arpack 求解器或者需要的主成分数量较少（小于100），则执行以下操作
    if arpack_all or n_components < 100:
        # 打印信息，指示正在使用 arpack 求解器
        print("  - arpack solver")
        # 对每个迭代次数执行以下操作
        for i in range(n_iter):
            # 记录开始时间
            start_time = time.perf_counter()
            # 使用 arpack 求解器对训练数据进行 KernelPCA 拟合并对测试数据进行转换
            a_pred = (
                KernelPCA(n_components, eigen_solver="arpack")
                .fit(X_train)
                .transform(X_test)
            )
            # 记录运行时间并存储到数组中
            a_time[j, i] = time.perf_counter() - start_time
            # 检查尽管近似，仍然确保结果是正确的
            assert_array_almost_equal(np.abs(a_pred), np.abs(ref_pred))

    # 打印信息，指示正在使用随机化求解器
    print("  - randomized solver")
    # 对每个迭代次数执行以下操作
    for i in range(n_iter):
        # 记录开始时间
        start_time = time.perf_counter()
        # 使用随机化求解器对训练数据进行 KernelPCA 拟合并对测试数据进行转换
        r_pred = (
            KernelPCA(n_components, eigen_solver="randomized")
            .fit(X_train)
            .transform(X_test)
        )
        # 记录运行时间并存储到数组中
        r_time[j, i] = time.perf_counter() - start_time
        # 检查尽管近似，仍然确保结果是正确的
        assert_array_almost_equal(np.abs(r_pred), np.abs(ref_pred))
# Compute statistics for the 3 methods
# 计算三种方法的统计数据

avg_ref_time = ref_time.mean(axis=1)
# 计算参考方法的平均执行时间

std_ref_time = ref_time.std(axis=1)
# 计算参考方法的执行时间标准差

avg_a_time = a_time.mean(axis=1)
# 计算arpack方法的平均执行时间

std_a_time = a_time.std(axis=1)
# 计算arpack方法的执行时间标准差

avg_r_time = r_time.mean(axis=1)
# 计算randomized方法的平均执行时间

std_r_time = r_time.std(axis=1)
# 计算randomized方法的执行时间标准差


# 4- Plots
# --------
# 创建一个图表对象，设置尺寸为12x8
fig, ax = plt.subplots(figsize=(12, 8))

# Display 1 plot with error bars per method
# 每种方法展示一个带有误差条的图表

ax.errorbar(
    n_compo_range,
    avg_ref_time,
    yerr=std_ref_time,
    marker="x",
    linestyle="",
    color="r",
    label="full",
)
# 绘制参考方法的图表，使用红色，带有"x"标记，无线条，带有误差条，标签为"full"

ax.errorbar(
    n_compo_range,
    avg_a_time,
    yerr=std_a_time,
    marker="x",
    linestyle="",
    color="g",
    label="arpack",
)
# 绘制arpack方法的图表，使用绿色，带有"x"标记，无线条，带有误差条，标签为"arpack"

ax.errorbar(
    n_compo_range,
    avg_r_time,
    yerr=std_r_time,
    marker="x",
    linestyle="",
    color="b",
    label="randomized",
)
# 绘制randomized方法的图表，使用蓝色，带有"x"标记，无线条，带有误差条，标签为"randomized"

ax.legend(loc="upper left")
# 设置图例位置为左上角

# customize axes
# 自定义坐标轴

ax.set_xscale("log")
# 设置x轴为对数尺度

ax.set_xlim(1, max(n_compo_range) * 1.1)
# 设置x轴的范围，从1到最大n_compo_range的值乘以1.1

ax.set_ylabel("Execution time (s)")
# 设置y轴标签为"Execution time (s)"

ax.set_xlabel("n_components")
# 设置x轴标签为"n_components"

ax.set_title(
    "kPCA Execution time comparison on %i samples with %i "
    "features, according to the choice of `eigen_solver`"
    "" % (n_train, n_features)
)
# 设置图表标题，显示样本数为n_train，特征数为n_features，根据'eigen_solver'选择比较kPCA执行时间

plt.show()
# 显示图表
```