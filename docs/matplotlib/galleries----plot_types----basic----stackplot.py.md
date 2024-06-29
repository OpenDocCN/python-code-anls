# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\stackplot.py`

```py
"""
===============
stackplot(x, y)
===============
Draw a stacked area plot or a streamgraph.

See `~matplotlib.axes.Axes.stackplot`
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于处理数组和数值计算
import numpy as np

# 使用指定的样式风格 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 创建数据
# 创建 x 轴数据，范围从 0 到 10，步长为 2
x = np.arange(0, 10, 2)
# 创建三组 y 轴数据
ay = [1, 1.25, 2, 2.75, 3]
by = [1, 1, 1, 1, 1]
cy = [2, 1, 2, 1, 2]
# 将三组 y 轴数据垂直堆叠成一个二维数组
y = np.vstack([ay, by, cy])

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 使用 stackplot 方法绘制堆叠区域图
ax.stackplot(x, y)

# 设置 x 轴和 y 轴的范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```