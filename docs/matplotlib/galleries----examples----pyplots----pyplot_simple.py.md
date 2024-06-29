# `D:\src\scipysrc\matplotlib\galleries\examples\pyplots\pyplot_simple.py`

```py
"""
===========
Simple plot
===========

A simple plot where a list of numbers are plotted against their index,
resulting in a straight line. Use a format string (here, 'o-r') to set the
markers (circles), linestyle (solid line) and color (red).

.. redirect-from:: /gallery/pyplots/fig_axes_labels_simple
.. redirect-from:: /gallery/pyplots/pyplot_formatstr
"""

# 导入 matplotlib.pyplot 模块，简称为 plt
import matplotlib.pyplot as plt

# 使用 plot 函数绘制图形，将给定的列表 [1, 2, 3, 4] 作为纵坐标数据，横坐标默认为索引
# 'o-r' 是格式字符串，设置了标记（圆圈'o'），线型（实线'-'），颜色（红色'r'）
plt.plot([1, 2, 3, 4], 'o-r')

# 设置纵坐标标签
plt.ylabel('some numbers')

# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.show`
```