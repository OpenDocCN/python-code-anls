# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\quiver_simple_demo.py`

```py
"""
==================
Quiver Simple Demo
==================

A simple example of a `~.axes.Axes.quiver` plot with a `~.axes.Axes.quiverkey`.

For more advanced options refer to
:doc:`/gallery/images_contours_and_fields/quiver_demo`.
"""
# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# 创建 X 和 Y 数组，范围从 -10 到 9，步长为 1
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
# 使用 meshgrid 函数创建 U 和 V 矩阵
U, V = np.meshgrid(X, Y)

# 创建图形和子图对象
fig, ax = plt.subplots()
# 绘制矢量场图，使用 X, Y, U, V 参数
q = ax.quiver(X, Y, U, V)
# 添加矢量场的标尺，位置在图中的坐标 (0.3, 1.1)，矢量长度为 10，标签为 'Quiver key, length = 10'，标签位置为东方 ('E')
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.quiver` / `matplotlib.pyplot.quiver`
#    - `matplotlib.axes.Axes.quiverkey` / `matplotlib.pyplot.quiverkey`
```