# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\stem3d_demo.py`

```
"""
=======
3D stem
=======

Demonstration of a stem plot in 3D, which plots vertical lines from a baseline
to the *z*-coordinate and places a marker at the tip.
"""

# 导入 matplotlib 的 pyplot 模块，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并命名为 np
import numpy as np

# 生成角度数组，范围从 0 到 2π
theta = np.linspace(0, 2*np.pi)
# 根据角度数组计算 x 坐标，cos 函数做相应变换
x = np.cos(theta - np.pi/2)
# 根据角度数组计算 y 坐标，sin 函数做相应变换
y = np.sin(theta - np.pi/2)
# z 坐标直接使用角度数组
z = theta

# 创建一个带有 3D 投影的子图
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# 在 3D 图中绘制 stem plot，显示垂直线从基线到 z 坐标，并在顶端放置一个标记
ax.stem(x, y, z)

plt.show()

# %%
#
# The position of the baseline can be adapted using *bottom*. The parameters
# *linefmt*, *markerfmt*, and *basefmt* control basic format properties of the
# plot. However, in contrast to `~.axes3d.Axes3D.plot` not all properties are
# configurable via keyword arguments. For more advanced control adapt the line
# objects returned by `~mpl_toolkits.mplot3d.axes3d.Axes3D.stem`.

# 创建一个新的带有 3D 投影的子图
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# 绘制 stem plot，定制线条格式为灰色，标记格式为菱形，并调整基线位置为 π
markerline, stemlines, baseline = ax.stem(
    x, y, z, linefmt='grey', markerfmt='D', bottom=np.pi)
# 设置标记的面颜色为空（即无填充）

plt.show()

# %%
#
# The orientation of the stems and baseline can be changed using *orientation*.
# This determines in which direction the stems are projected from the head
# points, towards the *bottom* baseline.
#
# For examples, by setting ``orientation='x'``, the stems are projected along
# the *x*-direction, and the baseline is in the *yz*-plane.

# 创建一个新的带有 3D 投影的子图
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# 绘制 stem plot，将基线位置设置为 -1，并指定投影方向为 'x'
markerline, stemlines, baseline = ax.stem(x, y, z, bottom=-1, orientation='x')
# 设置 x、y、z 轴的标签

plt.show()
```