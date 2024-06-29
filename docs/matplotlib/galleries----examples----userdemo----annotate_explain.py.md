# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\annotate_explain.py`

```py
"""
================
Annotate Explain
================

"""

# 导入 matplotlib 库中的 pyplot 模块，用于创建绘图和图形
import matplotlib.pyplot as plt

# 导入 matplotlib 库中的 patches 模块，用于创建图形对象，如椭圆等
import matplotlib.patches as mpatches

# 创建一个包含四个子图的图形对象和子图数组
fig, axs = plt.subplots(2, 2)

# 设置起始点和终点的坐标
x1, y1 = 0.3, 0.3
x2, y2 = 0.7, 0.7

# 在第一个子图上绘制起点和终点的连线
ax = axs.flat[0]
ax.plot([x1, x2], [y1, y2], ".")
# 创建一个椭圆对象并添加到第一个子图中
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)
# 在第一个子图上添加箭头注释，指示连接
ax.annotate("",
            xy=(x1, y1), xycoords='data',  # 箭头起点坐标
            xytext=(x2, y2), textcoords='data',  # 箭头终点坐标
            arrowprops=dict(arrowstyle="-",  # 箭头样式
                            color="0.5",  # 箭头颜色
                            patchB=None,  # 不用于连接的对象
                            shrinkB=0,  # 缩小连接区域的比例
                            connectionstyle="arc3,rad=0.3",  # 连接样式
                            ),
            )
# 在第一个子图上添加文本标签
ax.text(.05, .95, "connect", transform=ax.transAxes, ha="left", va="top")

# 重复以上步骤，为第二个子图到第四个子图添加相同的注释和图形绘制
ax = axs.flat[1]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchB=el,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )
ax.text(.05, .95, "clip", transform=ax.transAxes, ha="left", va="top")

ax = axs.flat[2]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )
ax.text(.05, .95, "shrink", transform=ax.transAxes, ha="left", va="top")

ax = axs.flat[3]
ax.plot([x1, x2], [y1, y2], ".")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
ax.add_artist(el)
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="fancy",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ),
            )
ax.text(.05, .95, "mutate", transform=ax.transAxes, ha="left", va="top")

# 对所有子图设置相同的坐标轴范围、刻度及纵横比
for ax in axs.flat:
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect=1)

# 显示图形
plt.show()
```