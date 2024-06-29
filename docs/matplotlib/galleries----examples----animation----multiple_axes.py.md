# `D:\src\scipysrc\matplotlib\galleries\examples\animation\multiple_axes.py`

```py
"""
=======================
Multiple Axes animation
=======================

This example showcases:

- how animation across multiple subplots works,
- using a figure artist in the animation.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib.animation as animation  # 导入 matplotlib 的 animation 模块
from matplotlib.patches import ConnectionPatch  # 导入 ConnectionPatch 类

# 创建包含两个子图的 Figure 对象
fig, (axl, axr) = plt.subplots(
    ncols=2,
    sharey=True,  # 共享 y 轴
    figsize=(6, 2),  # 设置图形大小
    gridspec_kw=dict(width_ratios=[1, 3], wspace=0),  # 设置子图比例和间距
)
axl.set_aspect(1)  # 设置左侧子图的纵横比为1
axr.set_box_aspect(1 / 3)  # 设置右侧子图的纵横比为1/3
axr.yaxis.set_visible(False)  # 隐藏右侧子图的 y 轴
axr.xaxis.set_ticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])  # 设置右侧子图的 x 轴刻度

# 在左侧子图中绘制圆形
x = np.linspace(0, 2 * np.pi, 50)
axl.plot(np.cos(x), np.sin(x), "k", lw=0.3)  # 绘制圆周
point, = axl.plot(0, 0, "o")  # 绘制初始点

# 在右侧子图中绘制完整的正弦曲线以设定视图限制
sine, = axr.plot(x, np.sin(x))  # 绘制正弦曲线

# 在两个图之间绘制连接线
con = ConnectionPatch(
    (1, 0),  # 起点在左侧子图的 (1, 0)
    (0, 0),  # 终点在右侧子图的 (0, 0)
    "data",  # 使用数据坐标系
    "data",  # 使用数据坐标系
    axesA=axl,  # 连接线起点所在的 Axes 对象
    axesB=axr,  # 连接线终点所在的 Axes 对象
    color="C0",  # 连接线颜色
    ls="dotted",  # 连接线线型
)
fig.add_artist(con)  # 将连接线添加到图形对象中


def animate(i):
    # 动画函数，更新正弦曲线和移动点的位置
    x = np.linspace(0, i, int(i * 25 / np.pi))  # 更新 x 数据范围
    sine.set_data(x, np.sin(x))  # 更新正弦曲线数据
    x, y = np.cos(i), np.sin(i)  # 计算点的新位置
    point.set_data([x], [y])  # 更新点的位置
    con.xy1 = x, y  # 更新连接线起点坐标
    con.xy2 = i, y  # 更新连接线终点坐标
    return point, sine, con  # 返回更新后的对象


# 创建动画对象
ani = animation.FuncAnimation(
    fig,
    animate,  # 调用的动画函数
    interval=50,  # 每帧之间的间隔时间（毫秒）
    blit=False,  # 禁用 blitting，因为 Figure artist 不能用于 blitting
    frames=x,  # 动画的帧数
    repeat_delay=100,  # 动画重复的延迟时间（毫秒）
)

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.ConnectionPatch`
#    - `matplotlib.animation.FuncAnimation`
#
# .. tags:: component: axes, animation
```