# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\bar.py`

```py
"""
==============
bar(x, height)
==============

See `~matplotlib.axes.Axes.bar`.
"""

# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 创建数据：
# x 轴数据，从 0.5 开始，步长为 1，共有 8 个数据点
x = 0.5 + np.arange(8)
# y 轴数据，包含 8 个高度值
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

# 创建图形和子图
fig, ax = plt.subplots()

# 绘制柱状图
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

# 设置 x 轴和 y 轴的范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```