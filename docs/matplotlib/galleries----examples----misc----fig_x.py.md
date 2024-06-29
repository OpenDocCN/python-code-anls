# `D:\src\scipysrc\matplotlib\galleries\examples\misc\fig_x.py`

```
"""
=======================
Adding lines to figures
=======================

Adding lines to a figure without any Axes.

.. redirect-from:: /gallery/pyplots/fig_x
"""

# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt

# 导入 matplotlib.lines 库
import matplotlib.lines as lines

# 创建一个新的空白图形对象
fig = plt.figure()

# 在图形对象中添加一条线，起点(0, 0)，终点(1, 1)
fig.add_artist(lines.Line2D([0, 1], [0, 1]))

# 在图形对象中添加另一条线，起点(0, 1)，终点(1, 0)
fig.add_artist(lines.Line2D([0, 1], [1, 0]))

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.figure`: 创建新的图形对象
#    - `matplotlib.lines`: 导入线条相关模块
#    - `matplotlib.lines.Line2D`: 添加线条到图形对象中
```