# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_mc_qmc.py`

```
"""MC vs QMC in terms of space filling."""
# 导入需要的库函数
from scipy.stats import qmc
import numpy as np

# 导入绘图库
import matplotlib.pyplot as plt

# 使用默认的随机数生成器创建一个随机数生成器对象
rng = np.random.default_rng()

# 设定样本点数量和维度
n_sample = 256
dim = 2

# 创建一个空字典用于存储不同方法生成的样本点
sample = {}

# 使用 Monte Carlo (MC) 方法生成样本点
sample['MC'] = rng.random((n_sample, dim))

# 使用 Sobol' 序列生成器创建 QMC 方法的样本点
engine = qmc.Sobol(d=dim, seed=rng)
sample["Sobol'"] = engine.random(n_sample)

# 创建一个包含两个子图的图像窗口
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# 遍历样本字典中的不同生成方法
for i, kind in enumerate(sample):
    # 在当前子图中绘制散点图
    axs[i].scatter(sample[kind][:, 0], sample[kind][:, 1])

    # 设置子图纵横比为相等
    axs[i].set_aspect('equal')
    # 设置坐标轴标签
    axs[i].set_xlabel(r'$x_1$')
    axs[i].set_ylabel(r'$x_2$')
    # 设置子图标题，显示生成方法和样本点的 $C^2$ 分散度
    axs[i].set_title(f'{kind}—$C^2 = ${qmc.discrepancy(sample[kind]):.2}')

# 调整子图的布局，使其紧凑显示
plt.tight_layout()
# 显示图像
plt.show()
```