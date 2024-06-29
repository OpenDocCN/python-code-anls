# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\legend_picking.py`

```py
"""
==============
Legend picking
==============

Enable picking on the legend to toggle the original line on and off

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

t = np.linspace(0, 1)  # 生成从 0 到 1 的等间隔数字的数组
y1 = 2 * np.sin(2 * np.pi * t)  # 计算第一个波形的数据
y2 = 4 * np.sin(2 * np.pi * 2 * t)  # 计算第二个波形的数据

fig, ax = plt.subplots()  # 创建一个新的图形和子图
ax.set_title('Click on legend line to toggle line on/off')  # 设置子图标题

(line1, ) = ax.plot(t, y1, lw=2, label='1 Hz')  # 绘制第一条线并添加到子图，设置线宽和标签
(line2, ) = ax.plot(t, y2, lw=2, label='2 Hz')  # 绘制第二条线并添加到子图，设置线宽和标签
leg = ax.legend(fancybox=True, shadow=True)  # 创建图例，并设置图例的样式为圆角和阴影效果

lines = [line1, line2]  # 将所有的线条放入列表中
map_legend_to_ax = {}  # 创建空字典，用于将图例线条映射到原始线条

pickradius = 5  # 点的半径，用于触发事件的点击范围

for legend_line, ax_line in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)  # 设置图例线条可被点击选取
    map_legend_to_ax[legend_line] = ax_line  # 将图例线条映射到对应的子图线条


def on_pick(event):
    # 在点击事件中，找到对应于图例线条的原始线条，并切换其可见性
    legend_line = event.artist

    # 如果事件源不是图例线条，则不执行任何操作
    if legend_line not in map_legend_to_ax:
        return

    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()  # 切换原始线条的可见性状态
    ax_line.set_visible(visible)  # 设置原始线条的可见性
    # 修改图例中线条的透明度，以显示已切换的线条
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()  # 重新绘制图形


fig.canvas.mpl_connect('pick_event', on_pick)  # 将点击事件连接到处理函数 on_pick 上

# 即使图例可以拖动，也可以正常工作。这与选择图例线条是独立的。
leg.set_draggable(True)  # 设置图例可拖动

plt.show()  # 显示绘制的图形
```