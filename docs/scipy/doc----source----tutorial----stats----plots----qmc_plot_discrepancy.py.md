# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_discrepancy.py`

```
"""Calculate the discrepancy of 2 designs and compare them."""
# 导入所需库
import numpy as np  # 导入 NumPy 库用于数值计算
from scipy.stats import qmc  # 导入 QMC（Quasi-Monte Carlo）相关函数
import matplotlib.pyplot as plt  # 导入 Matplotlib 库用于绘图

# 第一个设计空间
space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
# 第二个设计空间
space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])

# 设计空间边界
l_bounds = [0.5, 0.5]
u_bounds = [6.5, 6.5]

# 将设计空间进行缩放，使其落在指定边界内，reverse=True 表示反向缩放
space_1 = qmc.scale(space_1, l_bounds, u_bounds, reverse=True)
space_2 = qmc.scale(space_2, l_bounds, u_bounds, reverse=True)

# 构建样本字典，包含两个设计空间
sample = {'space_1': space_1, 'space_2': space_2}

# 创建包含两个子图的图形窗口
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# 遍历样本字典中的每个设计空间
for i, kind in enumerate(sample):
    # 在子图 i 上绘制散点图，显示设计空间的分布
    axs[i].scatter(sample[kind][:, 0], sample[kind][:, 1])

    # 设置图像比例为等比例
    axs[i].set_aspect('equal')
    # 设置 x 轴标签
    axs[i].set_xlabel(r'$x_1$')
    # 设置 y 轴标签
    axs[i].set_ylabel(r'$x_2$')
    # 设置子图标题，显示设计空间名称和其差异度的值
    axs[i].set_title(f'{kind}—$C^2 = ${qmc.discrepancy(sample[kind]):.5}')

# 调整子图布局，使其紧凑显示
plt.tight_layout()
# 显示图形
plt.show()
```