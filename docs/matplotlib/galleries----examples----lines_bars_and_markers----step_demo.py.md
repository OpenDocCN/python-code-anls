# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\step_demo.py`

```py
"""
=========
Step Demo
=========

This example demonstrates the use of `.pyplot.step` for piece-wise constant
curves. In particular, it illustrates the effect of the parameter *where*
on the step position.

.. note::

    For the common case that you know the edge positions, use `.pyplot.stairs`
    instead.

The circular markers created with `.pyplot.plot` show the actual data
positions so that it's easier to see the effect of *where*.

"""
# 导入 matplotlib.pyplot 模块，简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，简称为 np
import numpy as np

# 创建一个包含 14 个元素的数组 x，表示从 0 到 13 的整数
x = np.arange(14)
# 创建数组 y，其中每个元素是 x/2 的正弦值
y = np.sin(x / 2)

# 绘制步阶曲线，使用默认的 where='pre' 参数
plt.step(x, y + 2, label='pre (default)')
# 绘制数据点，使用灰色圆圈和虚线
plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

# 绘制步阶曲线，使用 where='mid' 参数
plt.step(x, y + 1, where='mid', label='mid')
# 绘制数据点，使用灰色圆圈和虚线
plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

# 绘制步阶曲线，使用 where='post' 参数
plt.step(x, y, where='post', label='post')
# 绘制数据点，使用灰色圆圈和虚线
plt.plot(x, y, 'o--', color='grey', alpha=0.3)

# 设置 x 轴的网格线颜色为浅灰色
plt.grid(axis='x', color='0.95')
# 添加图例，标题为 'Parameter where:'
plt.legend(title='Parameter where:')
# 设置图的标题为 'plt.step(where=...)'
plt.title('plt.step(where=...)')
# 显示图形
plt.show()

# %%
# The same behavior can be achieved by using the ``drawstyle`` parameter of
# `.pyplot.plot`.

# 使用 `drawstyle` 参数绘制类似的步阶曲线效果
plt.plot(x, y + 2, drawstyle='steps', label='steps (=steps-pre)')
# 绘制数据点，使用灰色圆圈和虚线
plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

plt.plot(x, y + 1, drawstyle='steps-mid', label='steps-mid')
plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

plt.plot(x, y, drawstyle='steps-post', label='steps-post')
plt.plot(x, y, 'o--', color='grey', alpha=0.3)

# 设置 x 轴的网格线颜色为浅灰色
plt.grid(axis='x', color='0.95')
# 添加图例，标题为 'Parameter drawstyle:'
plt.legend(title='Parameter drawstyle:')
# 设置图的标题为 'plt.plot(drawstyle=...)'
plt.title('plt.plot(drawstyle=...)')
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.step` / `matplotlib.pyplot.step`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
```