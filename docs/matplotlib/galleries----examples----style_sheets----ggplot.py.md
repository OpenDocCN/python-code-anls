# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\ggplot.py`

```py
"""
==================
ggplot style sheet
==================

This example demonstrates the "ggplot" style, which adjusts the style to
emulate ggplot_ (a popular plotting package for R_).

These settings were shamelessly stolen from [1]_ (with permission).

.. [1] https://everyhue.me/posts/sane-color-scheme-for-matplotlib/

.. _ggplot: https://ggplot2.tidyverse.org/
.. _R: https://www.r-project.org/

"""
# 导入 matplotlib.pyplot 库，简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，简称为 np
import numpy as np

# 使用 'ggplot' 样式
plt.style.use('ggplot')

# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 创建一个包含 2x2 子图的图形
fig, axs = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axs.flat

# 绘制散点图（注意：plt.scatter 不使用默认颜色）
x, y = np.random.normal(size=(2, 200))
ax1.plot(x, y, 'o')

# 绘制带有默认颜色循环的正弦曲线
L = 2*np.pi
x = np.linspace(0, L)
ncolors = len(plt.rcParams['axes.prop_cycle'])
shift = np.linspace(0, L, ncolors, endpoint=False)
for s in shift:
    ax2.plot(x, np.sin(x + s), '-')
ax2.margins(0)

# 绘制条形图
x = np.arange(5)
y1, y2 = np.random.randint(1, 25, size=(2, 5))
width = 0.25
ax3.bar(x, y1, width)
ax3.bar(x + width, y2, width,
        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
ax3.set_xticks(x + width, labels=['a', 'b', 'c', 'd', 'e'])

# 绘制带有默认颜色循环的圆圈
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    xy = np.random.normal(size=2)
    ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))
ax4.axis('equal')
ax4.margins(0)

# 显示图形
plt.show()
```