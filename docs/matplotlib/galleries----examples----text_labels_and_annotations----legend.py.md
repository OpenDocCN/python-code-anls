# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\legend.py`

```
"""
===============================
Legend using pre-defined labels
===============================

Defining legend labels with plots.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，并简写为 plt
import numpy as np  # 导入 numpy 库，并简写为 np

# Make some fake data.
a = b = np.arange(0, 3, .02)  # 创建一个从0到3，步长为0.02的数组，并赋值给 a 和 b
c = np.exp(a)  # 计算数组 a 中每个元素的指数值，赋值给 c
d = c[::-1]  # 将数组 c 中的元素逆序排列，赋值给 d

# Create plots with pre-defined labels.
fig, ax = plt.subplots()  # 创建一个图形和一个轴对象
ax.plot(a, c, 'k--', label='Model length')  # 在轴对象上绘制线条，使用黑色虚线，设置标签为 'Model length'
ax.plot(a, d, 'k:', label='Data length')  # 在轴对象上绘制线条，使用黑色点线，设置标签为 'Data length'
ax.plot(a, c + d, 'k', label='Total message length')  # 在轴对象上绘制线条，使用黑色实线，设置标签为 'Total message length'

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')  # 在轴对象上创建图例，位于上中位置，带阴影效果，字体大小为 'x-large'

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')  # 设置图例的背景颜色为 C0 色

plt.show()  # 显示绘制的图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
```