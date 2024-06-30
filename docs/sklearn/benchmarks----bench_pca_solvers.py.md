# `D:\src\scipysrc\scikit-learn\benchmarks\bench_pca_solvers.py`

```
# %%
# 该部分是一个分析 PCA 解算器在不同数据集大小上速度的基准测试，以确定通过 "auto" 策略选择默认解算器的最佳选择。
#
# 注意：我们没有控制解算器的准确性，假设所有解算器产生的转换数据具有类似的解释方差。这一假设通常成立，但随机解算器可能需要更多的幂迭代。
#
# 我们生成具有不同维度的合成数据，以便进行绘图：
# - 对于固定的 n_features，绘制时间 vs n_samples，
# - 对于固定的 n_samples 和 n_features，绘制时间 vs n_features。
import itertools
from math import log10
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import config_context
from sklearn.decomposition import PCA

# 参考维度，用于生成数据形状
REF_DIMS = [100, 1000, 10_000]
data_shapes = []

# 根据 REF_DIMS 生成多种数据形状
for ref_dim in REF_DIMS:
    data_shapes.extend([(ref_dim, 10**i) for i in range(1, 8 - int(log10(ref_dim)))])
    data_shapes.extend([(ref_dim, 3 * 10**i) for i in range(1, 8 - int(log10(ref_dim)))])
    data_shapes.extend([(10**i, ref_dim) for i in range(1, 8 - int(log10(ref_dim)))])
    data_shapes.extend([(3 * 10**i, ref_dim) for i in range(1, 8 - int(log10(ref_dim)))])

# 去除重复的数据形状
data_shapes = sorted(set(data_shapes))

# 打印生成测试数据集的信息
print("Generating test datasets...")

# 使用默认随机数生成器创建数据集
rng = np.random.default_rng(0)
datasets = [rng.normal(size=shape) for shape in data_shapes]


# %%
def measure_one(data, n_components, solver, method_name="fit"):
    # 输出当前测试的信息，包括解算器类型、主成分数量、方法名称和数据形状
    print(f"Benchmarking {solver=!r}, {n_components=}, {method_name=!r} on data with shape {data.shape}")
    
    # 创建 PCA 对象
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=0)
    timings = []
    elapsed = 0
    # 获取 PCA 对象中的方法，如 fit 或 fit_transform
    method = getattr(pca, method_name)
    
    # 使用 config_context 设置假设数据为有限
    with config_context(assume_finite=True):
        while elapsed < 0.5:
            tic = perf_counter()
            method(data)  # 执行方法，记录执行时间
            duration = perf_counter() - tic
            timings.append(duration)
            elapsed += duration
    
    return np.median(timings)


# 支持的解算器列表
SOLVERS = ["full", "covariance_eigh", "arpack", "randomized", "auto"]
measurements = []

# 对每个数据集、主成分数量和方法名称的组合进行测试
for data, n_components, method_name in itertools.product(
    datasets, [2, 50], ["fit", "fit_transform"]
):
    if n_components >= min(data.shape):
        continue
    # 遍历所有求解器列表 SOLVERS
    for solver in SOLVERS:
        # 如果当前求解器为 "covariance_eigh" 并且数据列数大于5000，则跳过当前循环
        if solver == "covariance_eigh" and data.shape[1] > 5000:
            # 内存占用过多且速度过慢，不适合当前情况
            continue
        # 如果当前求解器为 "arpack" 或者 "full" 并且数据元素的对数大于7，则跳过当前循环
        if solver in ["arpack", "full"] and log10(data.size) > 7:
            # 特别是对于完全求解器，速度过慢
            continue
        # 测量使用当前求解器求解的时间
        time = measure_one(data, n_components, solver, method_name=method_name)
        # 将测量结果作为字典添加到 measurements 列表中
        measurements.append(
            {
                "n_components": n_components,
                "n_samples": data.shape[0],
                "n_features": data.shape[1],
                "time": time,
                "solver": solver,
                "method_name": method_name,
            }
        )
# 将 measurements 转换为 Pandas DataFrame 格式
measurements = pd.DataFrame(measurements)

# 将 measurements 数据保存为 CSV 文件，排除行索引
measurements.to_csv("bench_pca_solvers.csv", index=False)

# %%
# 获取所有不同的 method_name 列值
all_method_names = measurements["method_name"].unique()

# 获取所有不同的 n_components 列值
all_n_components = measurements["n_components"].unique()

# 遍历每个 method_name
for method_name in all_method_names:
    # 创建子图和轴数组，布局为 len(REF_DIMS) 行 * len(all_n_components) 列
    fig, axes = plt.subplots(
        figsize=(16, 16),
        nrows=len(REF_DIMS),
        ncols=len(all_n_components),
        sharey=True,
        constrained_layout=True,
    )
    # 设置总标题，展示 PCA 方法名和变化的 n_samples
    fig.suptitle(f"Benchmarks for PCA.{method_name}, varying n_samples", fontsize=16)

    # 遍历 REF_DIMS 中的每个维度
    for row_idx, ref_dim in enumerate(REF_DIMS):
        # 遍历 all_n_components 中的每个 n_components
        for n_components, ax in zip(all_n_components, axes[row_idx]):
            # 遍历 SOLVERS 列表中的每个 solver
            for solver in SOLVERS:
                # 根据 solver 设置线条样式
                if solver == "auto":
                    style_kwargs = dict(linewidth=2, color="black", style="--")
                else:
                    style_kwargs = dict(style="o-")
                # 设置子图的标题和 y 轴标签
                ax.set(
                    title=f"n_components={n_components}, n_features={ref_dim}",
                    ylabel="time (s)",
                )
                # 根据条件查询数据并在子图上绘制线条图
                measurements.query(
                    "n_components == @n_components and n_features == @ref_dim"
                    " and solver == @solver and method_name == @method_name"
                ).plot.line(
                    x="n_samples",
                    y="time",
                    label=solver,
                    logx=True,
                    logy=True,
                    ax=ax,
                    **style_kwargs,
                )
# %%
# 针对每个 method_name 再次执行相同的操作
for method_name in all_method_names:
    # 创建子图和轴数组，布局为 len(REF_DIMS) 行 * len(all_n_components) 列
    fig, axes = plt.subplots(
        figsize=(16, 16),
        nrows=len(REF_DIMS),
        ncols=len(all_n_components),
        sharey=True,
    )
    # 设置总标题，展示 PCA 方法名和变化的 n_features
    fig.suptitle(f"Benchmarks for PCA.{method_name}, varying n_features", fontsize=16)

    # 遍历 REF_DIMS 中的每个维度
    for row_idx, ref_dim in enumerate(REF_DIMS):
        # 遍历 all_n_components 中的每个 n_components
        for n_components, ax in zip(all_n_components, axes[row_idx]):
            # 遍历 SOLVERS 列表中的每个 solver
            for solver in SOLVERS:
                # 根据 solver 设置线条样式
                if solver == "auto":
                    style_kwargs = dict(linewidth=2, color="black", style="--")
                else:
                    style_kwargs = dict(style="o-")
                # 设置子图的标题和 y 轴标签
                ax.set(
                    title=f"n_components={n_components}, n_samples={ref_dim}",
                    ylabel="time (s)",
                )
                # 根据条件查询数据并在子图上绘制线条图
                measurements.query(
                    "n_components == @n_components and n_samples == @ref_dim "
                    " and solver == @solver and method_name == @method_name"
                ).plot.line(
                    x="n_features",
                    y="time",
                    label=solver,
                    logx=True,
                    logy=True,
                    ax=ax,
                    **style_kwargs,
                )

# %%
```