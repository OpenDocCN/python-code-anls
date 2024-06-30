# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_sobol_halton.py`

```
"""Sobol' and Halton sequences."""
# 导入所需的库函数
from scipy.stats import qmc
import numpy as np

# 导入绘图库
import matplotlib.pyplot as plt

# 创建一个随机数生成器对象
rng = np.random.default_rng()

# 设置样本数量和维度
n_sample = 256
dim = 2

# 创建一个空字典来存储样本数据
sample = {}

# 使用 Sobol' 序列生成器创建引擎对象并生成随机样本
engine = qmc.Sobol(d=dim, seed=rng)
sample["Sobol'"] = engine.random(n_sample)

# 使用 Halton 序列生成器创建引擎对象并生成随机样本
engine = qmc.Halton(d=dim, seed=rng)
sample["Halton"] = engine.random(n_sample)

# 创建一个包含两个子图的画布
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# 遍历样本字典，绘制散点图
for i, kind in enumerate(sample):
    axs[i].scatter(sample[kind][:, 0], sample[kind][:, 1])  # 绘制散点图

    axs[i].set_aspect('equal')  # 设置图形纵横比为1:1
    axs[i].set_xlabel(r'$x_1$')  # 设置x轴标签
    axs[i].set_ylabel(r'$x_2$')  # 设置y轴标签
    axs[i].set_title(f'{kind}—$C^2 = ${qmc.discrepancy(sample[kind]):.2}')  # 设置子图标题，包括序列类型和偏差度量值

# 调整子图布局，使其紧凑显示
plt.tight_layout()
# 显示绘制的图形
plt.show()
```