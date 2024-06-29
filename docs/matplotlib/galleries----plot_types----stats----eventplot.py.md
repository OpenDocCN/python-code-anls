# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\eventplot.py`

```py
"""
============
eventplot(D)
============
Plot identical parallel lines at the given positions.

See `~matplotlib.axes.Axes.eventplot`.
"""
# 导入 matplotlib.pyplot 库，并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 使用 '_mpl-gallery' 样式
plt.style.use('_mpl-gallery')

# 设置随机种子为 1，以便结果可重复
np.random.seed(1)
# 设置 x 轴偏移位置
x = [2, 4, 6]
# 生成 Gamma 分布的随机数据，形状为 (3, 50)
D = np.random.gamma(4, size=(3, 50))

# 创建图表和轴对象
fig, ax = plt.subplots()

# 绘制事件图，设置为垂直方向，使用 x 轴偏移和线宽度为 0.75
ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidth=0.75)

# 设置 x 轴范围和刻度，限定在 0 到 8 之间，刻度从 1 到 7
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       # 设置 y 轴范围和刻度，限定在 0 到 8 之间，刻度从 1 到 7
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图表
plt.show()
```