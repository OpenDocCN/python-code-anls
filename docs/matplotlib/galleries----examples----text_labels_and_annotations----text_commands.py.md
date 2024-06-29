# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\text_commands.py`

```py
"""
=============
Text Commands
=============

Plotting text of many different kinds.

.. redirect-from:: /gallery/pyplots/text_commands
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 创建一个新的图形对象
fig = plt.figure()
# 设置图形的总标题
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

# 添加一个子图到图形中
ax = fig.add_subplot()
# 调整子图的参数，使得标题不会被子图内容遮挡
fig.subplots_adjust(top=0.85)
# 设置子图的标题
ax.set_title('axes title')

# 设置子图的 x 轴标签
ax.set_xlabel('xlabel')
# 设置子图的 y 轴标签
ax.set_ylabel('ylabel')

# 在子图中添加带框的斜体文本，以数据坐标为基准
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

# 在子图中添加包含数学公式的文本
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

# 在子图中添加包含 Unicode 字符的文本
ax.text(3, 2, 'Unicode: Institut f\374r Festk\366rperphysik')

# 在子图中添加使用坐标轴比例的彩色文本
ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)

# 在子图中绘制一个点
ax.plot([2], [1], 'o')
# 在子图中添加注释
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

# 设置子图的 x 和 y 轴的范围
ax.set(xlim=(0, 10), ylim=(0, 10))

# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.suptitle`
#    - `matplotlib.figure.Figure.add_subplot`
#    - `matplotlib.figure.Figure.subplots_adjust`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_xlabel`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.text`
#    - `matplotlib.axes.Axes.annotate`
```