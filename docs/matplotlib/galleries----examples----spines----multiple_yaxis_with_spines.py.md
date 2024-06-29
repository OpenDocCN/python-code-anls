# `D:\src\scipysrc\matplotlib\galleries\examples\spines\multiple_yaxis_with_spines.py`

```py
r"""
===========================
Multiple y-axis with Spines
===========================

Create multiple y axes with a shared x-axis. This is done by creating
a `~.axes.Axes.twinx` Axes, turning all spines but the right one invisible
and offset its position using `~.spines.Spine.set_position`.

Note that this approach uses `matplotlib.axes.Axes` and their
`~matplotlib.spines.Spine`\s.  Alternative approaches using non-standard Axes
are shown in the :doc:`/gallery/axisartist/demo_parasite_axes` and
:doc:`/gallery/axisartist/demo_parasite_axes2` examples.
"""

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 创建一个包含单个子图的图形fig和坐标轴ax
fig, ax = plt.subplots()

# 调整图形的右边距，使得有足够的空间放置额外的y轴
fig.subplots_adjust(right=0.75)

# 创建第一个与ax共享x轴的y轴，即ax的孪生y轴
twin1 = ax.twinx()

# 创建第二个与ax共享x轴的y轴，即ax的另一个孪生y轴
twin2 = ax.twinx()

# 调整twin2的右边脊柱位置，使其偏移相对于坐标轴1.2倍的宽度
twin2.spines.right.set_position(("axes", 1.2))

# 绘制三条曲线，并分别将其赋给变量p1、p2、p3
p1, = ax.plot([0, 1, 2], [0, 1, 2], "C0", label="Density")  # 密度曲线
p2, = twin1.plot([0, 1, 2], [0, 3, 2], "C1", label="Temperature")  # 温度曲线
p3, = twin2.plot([0, 1, 2], [50, 30, 15], "C2", label="Velocity")  # 速度曲线

# 设置主坐标轴ax的x轴和y轴的范围，以及标签
ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Distance", ylabel="Density")

# 设置twin1的y轴范围和标签
twin1.set(ylim=(0, 4), ylabel="Temperature")

# 设置twin2的y轴范围和标签
twin2.set(ylim=(1, 65), ylabel="Velocity")

# 设置主坐标轴ax的y轴标签颜色与p1曲线颜色相同
ax.yaxis.label.set_color(p1.get_color())

# 设置twin1的y轴标签颜色与p2曲线颜色相同
twin1.yaxis.label.set_color(p2.get_color())

# 设置twin2的y轴标签颜色与p3曲线颜色相同
twin2.yaxis.label.set_color(p3.get_color())

# 设置主坐标轴ax的y轴刻度颜色与p1曲线颜色相同
ax.tick_params(axis='y', colors=p1.get_color())

# 设置twin1的y轴刻度颜色与p2曲线颜色相同
twin1.tick_params(axis='y', colors=p2.get_color())

# 设置twin2的y轴刻度颜色与p3曲线颜色相同
twin2.tick_params(axis='y', colors=p3.get_color())

# 在图上添加图例，包括p1、p2、p3三条曲线的标签
ax.legend(handles=[p1, p2, p3])

# 显示图形
plt.show()
```