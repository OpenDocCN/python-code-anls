# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\annotate_transform.py`

```py
"""
==================
Annotate Transform
==================

This example shows how to use different coordinate systems for annotations.
For a complete overview of the annotation capabilities, also see the
:ref:`annotation tutorial<annotations>`.

.. redirect-from:: /gallery/pyplots/annotate_transform
"""

# 导入 matplotlib 库中的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 创建一个包含数据点的数组 x 和对应的函数值数组 y
x = np.arange(0, 10, 0.005)
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

# 创建一个图形和坐标系对象
fig, ax = plt.subplots()
# 在坐标系中绘制曲线
ax.plot(x, y)
# 设置 x 轴的显示范围
ax.set_xlim(0, 10)
# 设置 y 轴的显示范围
ax.set_ylim(-1, 1)

# 设定数据点的坐标和要显示的位置的坐标的转换
xdata, ydata = 5, 0
xdisplay, ydisplay = ax.transData.transform((xdata, ydata))

# 定义标注框的样式
bbox = dict(boxstyle="round", fc="0.8")
# 定义箭头的样式
arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

# 设置第一个注释的偏移量
offset = 72
# 在坐标系中添加第一个注释，显示数据点的坐标
ax.annotate(
    f'data = ({xdata:.1f}, {ydata:.1f})',  # 注释的文本内容
    (xdata, ydata),  # 注释的位置
    xytext=(-2*offset, offset), textcoords='offset points',  # 文本的偏移量和坐标系
    bbox=bbox, arrowprops=arrowprops  # 注释框和箭头的样式
)

# 设置第二个注释的偏移量
ax.annotate(
    f'display = ({xdisplay:.1f}, {ydisplay:.1f})',  # 注释的文本内容
    xy=(xdisplay, ydisplay), xycoords='figure pixels',  # 注释的位置和坐标系
    xytext=(0.5*offset, -offset), textcoords='offset points',  # 文本的偏移量和坐标系
    bbox=bbox, arrowprops=arrowprops  # 注释框和箭头的样式
)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.transforms.Transform.transform`
#    - `matplotlib.axes.Axes.annotate` / `matplotlib.pyplot.annotate`
```