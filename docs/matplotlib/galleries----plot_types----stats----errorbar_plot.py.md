# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\errorbar_plot.py`

```
"""
==========================
errorbar(x, y, yerr, xerr)
==========================
Plot y versus x as lines and/or markers with attached errorbars.

See `~matplotlib.axes.Axes.errorbar`.
"""
# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用 '_mpl-gallery' 样式
plt.style.use('_mpl-gallery')

# 创建随机数种子以确保可复现性
np.random.seed(1)

# 定义数据
x = [2, 4, 6]
y = [3.6, 5, 4.2]
yerr = [0.9, 1.2, 0.5]

# 创建图形和坐标系
fig, ax = plt.subplots()

# 绘制带误差线的散点图
ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)

# 设置坐标轴范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```