# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\ticklabels_rotation.py`

```
"""
====================
Rotating tick labels
====================
"""

# 导入 matplotlib.pyplot 库
import matplotlib.pyplot as plt

# 创建一些示例数据
x = [1, 2, 3, 4]
y = [1, 4, 9, 6]
labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']

# 创建图形和轴对象
fig, ax = plt.subplots()
# 在轴上绘制数据图
ax.plot(x, y)

# 使用 Axes.tick_params 方法设置 y 轴刻度标签的旋转角度为 45 度
ax.tick_params("y", rotation=45)

# 使用 Axes.set_xticks 方法设置 x 轴刻度位置和标签，并设置标签的垂直方向旋转
ax.set_xticks(x, labels, rotation='vertical')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.tick_params` / `matplotlib.pyplot.tick_params`
#    - `matplotlib.axes.Axes.set_xticks` / `matplotlib.pyplot.xticks`
```