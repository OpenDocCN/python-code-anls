# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\connectionstyle_demo.py`

```
"""
=================================
Connection styles for annotations
=================================

When creating an annotation using `~.Axes.annotate`, the arrow shape can be
controlled via the *connectionstyle* parameter of *arrowprops*. For further
details see the description of `.FancyArrowPatch`.
"""

# 导入 matplotlib 的 pyplot 模块，简称 plt
import matplotlib.pyplot as plt


# 定义演示连接样式的函数 demo_con_style
def demo_con_style(ax, connectionstyle):
    # 定义起始点和终止点的坐标
    x1, y1 = 0.3, 0.2
    x2, y2 = 0.8, 0.6

    # 在 Axes 对象 ax 上绘制起始点和终止点的散点图
    ax.plot([x1, x2], [y1, y2], ".")

    # 创建箭头注释，指定起始点、终止点和箭头样式等参数
    ax.annotate("",
                xy=(x1, y1), xycoords='data',  # 箭头起始点的数据坐标
                xytext=(x2, y2), textcoords='data',  # 箭头终止点的数据坐标
                arrowprops=dict(arrowstyle="->", color="0.5",  # 箭头的样式和颜色
                                shrinkA=5, shrinkB=5,  # 起始点和终止点的缩进距离
                                patchA=None, patchB=None,  # 起始点和终止点的修补
                                connectionstyle=connectionstyle,  # 指定连接样式
                                ),
                )

    # 在 Axes 对象 ax 上添加文字注释，显示连接样式的具体参数
    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")


# 创建 3x5 的子图布局
fig, axs = plt.subplots(3, 5, figsize=(7, 6.3), layout="constrained")

# 在子图中调用 demo_con_style 函数演示不同的连接样式
demo_con_style(axs[0, 0], "angle3,angleA=90,angleB=0")
demo_con_style(axs[1, 0], "angle3,angleA=0,angleB=90")
demo_con_style(axs[0, 1], "arc3,rad=0.")
demo_con_style(axs[1, 1], "arc3,rad=0.3")
demo_con_style(axs[2, 1], "arc3,rad=-0.3")
demo_con_style(axs[0, 2], "angle,angleA=-90,angleB=180,rad=0")
demo_con_style(axs[1, 2], "angle,angleA=-90,angleB=180,rad=5")
demo_con_style(axs[2, 2], "angle,angleA=-90,angleB=10,rad=5")
demo_con_style(axs[0, 3], "arc,angleA=-90,angleB=0,armA=30,armB=30,rad=0")
demo_con_style(axs[1, 3], "arc,angleA=-90,angleB=0,armA=30,armB=30,rad=5")
demo_con_style(axs[2, 3], "arc,angleA=-90,angleB=0,armA=0,armB=40,rad=0")
demo_con_style(axs[0, 4], "bar,fraction=0.3")
demo_con_style(axs[1, 4], "bar,fraction=-0.3")
demo_con_style(axs[2, 4], "bar,angle=180,fraction=-0.2")

# 遍历所有子图对象，设置坐标轴范围、刻度等属性
for ax in axs.flat:
    ax.set(xlim=(0, 1), ylim=(0, 1.25), xticks=[], yticks=[], aspect=1.25)

# 调整子图布局引擎的参数，设置子图之间的间距
fig.get_layout_engine().set(wspace=0, hspace=0, w_pad=0, h_pad=0)

# 显示图形
plt.show()
```