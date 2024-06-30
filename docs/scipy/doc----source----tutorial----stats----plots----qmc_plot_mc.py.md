# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_mc.py`

```
"""Multiple MC to show how it can be bad."""
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt

# 创建一个随机数生成器对象
rng = np.random.default_rng()

# 设置样本数量和维度
n_sample = 256
dim = 2

# 创建一个空字典用于存储样本数据
sample = {}

# 生成第一个蒙特卡洛样本数据
sample['MC 1'] = rng.random((n_sample, dim))
# 生成第二个蒙特卡洛样本数据
sample["MC 2"] = rng.random((n_sample, dim))

# 创建包含两个子图的图形窗口
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# 遍历样本数据字典，并在每个子图上绘制散点图
for i, kind in enumerate(sample):
    axs[i].scatter(sample[kind][:, 0], sample[kind][:, 1])  # 绘制散点图

    axs[i].set_aspect('equal')  # 设置纵横比相等
    axs[i].set_xlabel(r'$x_1$')  # 设置 x 轴标签
    axs[i].set_ylabel(r'$x_2$')  # 设置 y 轴标签

# 调整布局使得子图之间不重叠
plt.tight_layout()
# 显示图形
plt.show()
```