# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axes_props.py`

```py
"""
==========
Axes Props
==========

You can control the axis tick and grid properties
"""

# 导入 matplotlib 的 pyplot 模块，简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并简称为 np
import numpy as np

# 生成一个包含 t 值的 numpy 数组，范围是从 0 到 2（不包括），步长为 0.01
t = np.arange(0.0, 2.0, 0.01)
# 根据 t 的值，生成对应的正弦值数组 s
s = np.sin(2 * np.pi * t)

# 创建一个新的图形和一个子图对象 ax
fig, ax = plt.subplots()
# 在 ax 上绘制 t 对应的正弦值 s 的图形
ax.plot(t, s)

# 设置 ax 的网格显示为 True，网格线的线型为 '-.'
ax.grid(True, linestyle='-.')
# 设置刻度标签的颜色为红色 ('r')，标签大小为 'medium'，刻度线的宽度为 3
ax.tick_params(labelcolor='r', labelsize='medium', width=3)

# 显示图形
plt.show()
```