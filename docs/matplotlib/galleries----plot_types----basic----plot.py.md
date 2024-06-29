# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\plot.py`

```
"""
==========
plot(x, y)
==========
Plot y versus x as lines and/or markers.

See `~matplotlib.axes.Axes.plot`.
"""

# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 生成数据
# 生成从 0 到 10 的 100 个等间距数列作为 x 值
x = np.linspace(0, 10, 100)
# 根据公式生成对应的 y 值，y = 4 + sin(2x)
y = 4 + 1 * np.sin(2 * x)
# 生成从 0 到 10 的 25 个等间距数列作为 x2 值
x2 = np.linspace(0, 10, 25)
# 根据相同的公式生成对应的 y2 值
y2 = 4 + 1 * np.sin(2 * x2)

# 绘图
# 创建一个图形窗口和一个轴对象
fig, ax = plt.subplots()

# 在轴对象上绘制 x2 vs y2+2.5 的散点图，用 'x' 符号表示，边界宽度为2
ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)
# 在轴对象上绘制 x vs y 的线图，线宽为2.0
ax.plot(x, y, linewidth=2.0)
# 在轴对象上绘制 x2 vs y2-2.5 的线图，用实心圆点连接，线宽为2
ax.plot(x2, y2 - 2.5, 'o-', linewidth=2)

# 设置轴对象的属性：x 轴范围为 0 到 8，刻度为从 1 到 8
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```