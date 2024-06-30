# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_conv_mc.py`

```
"""Integration convergence.

The function is a synthetic example specifically designed
to verify the correctness of the implementation [1]_.

References
----------

.. [1] Art B. Owen. On dropping the first Sobol' point. arXiv 2008.08051,
   2020.

"""
# 导入必要的库
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

# 定义收敛的采样数量
n_conv = 99
# 生成采样数量的幂次方数组
ns_gen = 2 ** np.arange(4, 13)  # 13


def art_2(sample):
    # 定义一个示例函数，用于测试集成的收敛性能
    # 维度为3，真值为5/3 + 5*(5 - 1)/4
    return np.sum(sample, axis=1) ** 2


# 定义一个命名元组，包含函数名称、函数本身、维度和参考值
functions = namedtuple('functions', ['name', 'func', 'dim', 'ref'])
case = functions('Art 2', art_2, 5, 5 / 3 + 5 * (5 - 1) / 4)


def conv_method(sampler, func, n_samples, n_conv, ref):
    # 执行收敛性方法，采样多次以评估误差
    samples = [sampler(n_samples) for _ in range(n_conv)]
    samples = np.array(samples)

    # 计算每个样本的函数评估值
    evals = [np.sum(func(sample)) / n_samples for sample in samples]
    # 计算平方误差并计算均方根误差
    squared_errors = (ref - np.array(evals)) ** 2
    rmse = (np.sum(squared_errors) / n_conv) ** 0.5

    return rmse


# Analysis
sample_mc_rmse = []
rng = np.random.default_rng()

def sampler_mc(x):
    # 定义Monte Carlo采样函数
    return rng.random((x, case.dim))

# 对每个采样数量进行Monte Carlo方法的收敛性评估
for ns in ns_gen:
    conv_res = conv_method(sampler_mc, case.func, ns, n_conv, case.ref)
    sample_mc_rmse.append(conv_res)

sample_mc_rmse = np.array(sample_mc_rmse)

# Plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_aspect('equal')

# 计算比例和绘制直线
ratio = sample_mc_rmse[0] / ns_gen[0] ** (-1 / 2)
ax.plot(ns_gen, ns_gen ** (-1 / 2) * ratio, ls='-', c='k')

# 绘制散点图
ax.scatter(ns_gen, sample_mc_rmse)

# 设置图形标签和比例尺
ax.set_xlabel(r'$N_s$')
ax.set_xscale('log')
ax.set_xticks(ns_gen)
ax.set_xticklabels([fr'$2^{{{ns}}}$' for ns in np.arange(4, 13)])

ax.set_ylabel(r'$\log (\epsilon)$')
ax.set_yscale('log')

# 调整布局并显示图形
fig.tight_layout()
plt.show()
```