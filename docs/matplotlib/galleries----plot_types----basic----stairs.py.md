# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\stairs.py`

```
"""
==============
stairs(values)
==============
Draw a stepwise constant function as a line or a filled plot.

See `~matplotlib.axes.Axes.stairs` when plotting :math:`y` between
:math:`(x_i, x_{i+1})`. For plotting :math:`y` at :math:`x`, see
`~matplotlib.axes.Axes.step`.

.. redirect-from:: /plot_types/basic/step
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数据
import numpy as np

# 使用指定的 matplotlib 样式 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 生成数据
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

# 创建图形和轴对象
fig, ax = plt.subplots()

# 使用 ax.stairs 方法绘制阶梯图，设置线宽为 2.5
ax.stairs(y, linewidth=2.5)

# 设置 x 轴的范围为 (0, 8)，设置 x 轴刻度为 [1, 2, ..., 7]
# 设置 y 轴的范围为 (0, 8)，设置 y 轴刻度为 [1, 2, ..., 7]
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```