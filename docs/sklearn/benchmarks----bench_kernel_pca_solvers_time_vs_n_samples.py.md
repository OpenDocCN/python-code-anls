# `D:\src\scipysrc\scikit-learn\benchmarks\bench_kernel_pca_solvers_time_vs_n_samples.py`

```
"""
==========================================================
Kernel PCA Solvers comparison benchmark: time vs n_samples
==========================================================

This benchmark shows that the approximate solvers provided in Kernel PCA can
help significantly improve its execution speed when an approximate solution
(small `n_components`) is acceptable. In many real-world datasets the number of
samples is very large, but a few hundreds of principal components are
sufficient enough to capture the underlying distribution.

Description:
------------
An increasing number of examples is used to train a KernelPCA, between
`min_n_samples` (default: 101) and `max_n_samples` (default: 4000) with
`n_samples_grid_size` positions (default: 4). Samples have 2 features, and are
generated using `make_circles`. For each training sample size, KernelPCA models
are trained for the various possible `eigen_solver` values. All of them are
trained to obtain `n_components` principal components (default: 100). The
execution times are displayed in a plot at the end of the experiment.

What you can observe:
---------------------
When the number of samples provided gets large, the dense solver takes a lot
of time to complete, while the randomized method returns similar results in
much shorter execution times.

Going further:
--------------
You can increase `max_n_samples` and `nb_n_samples_to_try` if you wish to
explore a wider range of values for `n_samples`.

You can also set `include_arpack=True` to add this other solver in the
experiments (much slower).

Finally you can have a look at the second example of this series, "Kernel PCA
Solvers comparison benchmark: time vs n_components", where this time the number
of examples is fixed, and the desired number of components varies.
"""

# Author: Sylvain MARIE, Schneider Electric

import time  # 导入时间模块，用于计时

import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_array_almost_equal  # 导入NumPy的测试工具，用于数组比较

from sklearn.datasets import make_circles  # 导入make_circles函数，用于生成环形数据集
from sklearn.decomposition import KernelPCA  # 导入KernelPCA类，用于核PCA分析

print(__doc__)  # 打印文档字符串（上面整段的注释说明）


# 1- Design the Experiment
# ------------------------
min_n_samples, max_n_samples = 101, 4000  # 定义最小和最大的样本数范围
n_samples_grid_size = 4  # 设置尝试的样本数的网格大小
# 生成样本数的网格
n_samples_range = [
    min_n_samples
    + np.floor((x / (n_samples_grid_size - 1)) * (max_n_samples - min_n_samples))
    for x in range(0, n_samples_grid_size)
]

n_components = 100  # 我们希望使用的主成分数目
n_iter = 3  # 每个实验重复运行的次数
include_arpack = False  # 设置为True以包括arpack求解器（更慢）


# 2- Generate random data
# -----------------------
n_features = 2  # 数据集特征数
X, y = make_circles(n_samples=max_n_samples, factor=0.3, noise=0.05, random_state=0)  # 生成环形数据集


# 3- Benchmark
# ------------
# 初始化
ref_time = np.empty((len(n_samples_range), n_iter)) * np.nan  # 初始化参考时间的数组
a_time = np.empty((len(n_samples_range), n_iter)) * np.nan  # 初始化近似解时间的数组
# 创建一个空的 NumPy 数组，用于存储每个 n_samples_range 和 n_iter 的计算时间
r_time = np.empty((len(n_samples_range), n_iter)) * np.nan

# 对 n_samples_range 中的每个 n_samples 进行循环
for j, n_samples in enumerate(n_samples_range):
    # 将 n_samples 转换为整数类型
    n_samples = int(n_samples)
    # 打印当前执行 kPCA 的 n_samples 值
    print("Performing kPCA with n_samples = %i" % n_samples)

    # 从数据集 X 中选择训练集和测试集（此处测试集即为整个训练集）
    X_train = X[:n_samples, :]
    X_test = X_train

    # A- 参考 (dense)
    print("  - dense")
    for i in range(n_iter):
        start_time = time.perf_counter()
        # 执行核PCA，使用稠密求解器，拟合训练集并转换测试集
        ref_pred = (
            KernelPCA(n_components, eigen_solver="dense").fit(X_train).transform(X_test)
        )
        # 计算执行时间并保存
        ref_time[j, i] = time.perf_counter() - start_time

    # B- arpack
    if include_arpack:
        print("  - arpack")
        for i in range(n_iter):
            start_time = time.perf_counter()
            # 执行核PCA，使用arpack求解器，拟合训练集并转换测试集
            a_pred = (
                KernelPCA(n_components, eigen_solver="arpack")
                .fit(X_train)
                .transform(X_test)
            )
            # 计算执行时间并保存
            a_time[j, i] = time.perf_counter() - start_time
            # 检查结果的正确性，尽管存在近似
            assert_array_almost_equal(np.abs(a_pred), np.abs(ref_pred))

    # C- randomized
    print("  - randomized")
    for i in range(n_iter):
        start_time = time.perf_counter()
        # 执行核PCA，使用随机化求解器，拟合训练集并转换测试集
        r_pred = (
            KernelPCA(n_components, eigen_solver="randomized")
            .fit(X_train)
            .transform(X_test)
        )
        # 计算执行时间并保存
        r_time[j, i] = time.perf_counter() - start_time
        # 检查结果的正确性，尽管存在近似
        assert_array_almost_equal(np.abs(r_pred), np.abs(ref_pred))

# 计算三种方法的统计信息
avg_ref_time = ref_time.mean(axis=1)
std_ref_time = ref_time.std(axis=1)
avg_a_time = a_time.mean(axis=1)
std_a_time = a_time.std(axis=1)
avg_r_time = r_time.mean(axis=1)
std_r_time = r_time.std(axis=1)


# 生成图表
# --------
fig, ax = plt.subplots(figsize=(12, 8))

# 每种方法生成一个带有误差条的图表
ax.errorbar(
    n_samples_range,
    avg_ref_time,
    yerr=std_ref_time,
    marker="x",
    linestyle="",
    color="r",
    label="full",
)
if include_arpack:
    ax.errorbar(
        n_samples_range,
        avg_a_time,
        yerr=std_a_time,
        marker="x",
        linestyle="",
        color="g",
        label="arpack",
    )
ax.errorbar(
    n_samples_range,
    avg_r_time,
    yerr=std_r_time,
    marker="x",
    linestyle="",
    color="b",
    label="randomized",
)
ax.legend(loc="upper left")

# 自定义坐标轴
ax.set_xlim(min(n_samples_range) * 0.9, max(n_samples_range) * 1.1)
ax.set_ylabel("Execution time (s)")
ax.set_xlabel("n_samples")

# 设置图表标题，显示kPCA在具有特定n_components和n_features特征数的样本上的执行时间比较，根据'eigen_solver'的选择
ax.set_title(
    "Execution time comparison of kPCA with %i components on samples "
    "with %i features, according to the choice of `eigen_solver`"
    "" % (n_components, n_features)
)

plt.show()
```