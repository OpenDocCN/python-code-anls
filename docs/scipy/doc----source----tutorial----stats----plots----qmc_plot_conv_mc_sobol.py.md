# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_conv_mc_sobol.py`

```
"""Integration convergence comparison: MC vs Sobol'.

The function is a synthetic example specifically designed
to verify the correctness of the implementation [2]_.

References
----------

.. [1] I. M. Sobol. The distribution of points in a cube and the accurate
   evaluation of integrals. Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
   1967.
.. [2] Art B. Owen. On dropping the first Sobol' point. arXiv 2008.08051,
   2020.

"""
# 引入必要的库和模块
from collections import namedtuple  # 引入 namedtuple，用于创建命名元组

import numpy as np  # 引入 NumPy 库，用于科学计算
import matplotlib.pyplot as plt  # 引入 Matplotlib 库，用于绘图
from scipy.stats import qmc  # 从 SciPy 库的 stats 模块中引入 qmc

n_conv = 99  # 设定收敛次数为 99
ns_gen = 2 ** np.arange(4, 13)  # 创建一个 NumPy 数组，包含 4 到 12 的指数值（4, 8, 16, ..., 4096）


def art_2(sample):
    # 定义一个函数，计算样本的平方和
    # 维度为 5，真实值为 5/3 + 5*(5 - 1)/4
    return np.sum(sample, axis=1) ** 2


functions = namedtuple('functions', ['name', 'func', 'dim', 'ref'])
case = functions('Art 2', art_2, 5, 5 / 3 + 5 * (5 - 1) / 4)


def conv_method(sampler, func, n_samples, n_conv, ref):
    # 定义一个函数，评估不同采样方法下的收敛性
    samples = [sampler(n_samples) for _ in range(n_conv)]  # 生成 n_conv 个样本
    samples = np.array(samples)  # 转换为 NumPy 数组

    evals = [np.sum(func(sample)) / n_samples for sample in samples]  # 计算每个样本的函数值平均
    squared_errors = (ref - np.array(evals)) ** 2  # 计算平方误差
    rmse = (np.sum(squared_errors) / n_conv) ** 0.5  # 计算均方根误差

    return rmse


# Analysis
sample_mc_rmse = []  # 初始化 Monte Carlo 方法的 RMSE 列表
sample_sobol_rmse = []  # 初始化 Sobol' 方法的 RMSE 列表
rng = np.random.default_rng()  # 创建一个 NumPy 随机数生成器

def sampler_mc(x):
    # 定义一个函数，生成 Monte Carlo 方法的样本
    return rng.random((x, case.dim))  # 返回一个 x 行 case.dim 列的随机样本矩阵


for ns in ns_gen:
    # Monte Carlo
    conv_res = conv_method(sampler_mc, case.func, ns, n_conv, case.ref)  # 计算 Monte Carlo 方法的收敛性
    sample_mc_rmse.append(conv_res)  # 将结果添加到 Monte Carlo RMSE 列表中

    # Sobol'
    engine = qmc.Sobol(d=case.dim, scramble=False)  # 创建一个 Sobol' 序列生成引擎
    conv_res = conv_method(engine.random, case.func, ns, 1, case.ref)  # 计算 Sobol' 方法的收敛性
    sample_sobol_rmse.append(conv_res)  # 将结果添加到 Sobol' RMSE 列表中

sample_mc_rmse = np.array(sample_mc_rmse)  # 转换 Monte Carlo RMSE 列表为 NumPy 数组
sample_sobol_rmse = np.array(sample_sobol_rmse)  # 转换 Sobol' RMSE 列表为 NumPy 数组

# Plot
fig, ax = plt.subplots(figsize=(4, 4))  # 创建绘图对象和子图

ax.set_aspect('equal')  # 设置坐标轴纵横比为相等

# MC
ratio = sample_mc_rmse[0] / ns_gen[0] ** (-1 / 2)  # 计算 Monte Carlo 方法的比率
ax.plot(ns_gen, ns_gen ** (-1 / 2) * ratio, ls='-', c='k')  # 绘制 Monte Carlo 方法的收敛速度曲线

ax.scatter(ns_gen, sample_mc_rmse, label="MC")  # 绘制 Monte Carlo 方法的 RMSE 散点图

# Sobol'
ratio = sample_sobol_rmse[0] / ns_gen[0] ** (-2/2)  # 计算 Sobol' 方法的比率
ax.plot(ns_gen, ns_gen ** (-2/2) * ratio, ls='-.', c='k')  # 绘制 Sobol' 方法的收敛速度曲线

ax.scatter(ns_gen, sample_sobol_rmse, label="Sobol' unscrambled")  # 绘制 Sobol' 方法的 RMSE 散点图

ax.set_xlabel(r'$N_s$')  # 设置 x 轴标签
ax.set_xscale('log')  # 设置 x 轴为对数尺度
ax.set_xticks(ns_gen)  # 设置 x 轴刻度
ax.set_xticklabels([fr'$2^{{{ns}}}$' for ns in np.arange(4, 13)])  # 设置 x 轴刻度标签

ax.set_ylabel(r'$\log (\epsilon)$')  # 设置 y 轴标签
ax.set_yscale('log')  # 设置 y 轴为对数尺度

ax.legend(loc='upper right')  # 设置图例位置为右上角
fig.tight_layout()  # 调整布局
plt.show()  # 显示图形
```