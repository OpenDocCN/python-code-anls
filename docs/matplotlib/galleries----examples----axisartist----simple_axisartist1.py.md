# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axisartist1.py`

```
"""
=============================
Custom spines with axisartist
=============================

This example showcases the use of :mod:`.axisartist` to draw spines at custom
positions (here, at ``y = 0``).

Note, however, that it is simpler to achieve this effect using standard
`.Spine` methods, as demonstrated in
:doc:`/gallery/spines/centered_spines_with_arrows`.

.. redirect-from:: /gallery/axisartist/simple_axisline2
"""

# 导入 matplotlib 的 pyplot 模块并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块并命名为 np
import numpy as np

# 从 mpl_toolkits 中导入 axisartist 模块
from mpl_toolkits import axisartist

# 创建一个新的 Figure 对象，并设置大小为 6x3，布局为 "constrained"
fig = plt.figure(figsize=(6, 3), layout="constrained")
# 使用 gridspec 添加一个 1x2 的子图网格
gs = fig.add_gridspec(1, 2)

# 在子图网格的第一格中添加一个 axisartist.Axes 类型的子图 ax0
ax0 = fig.add_subplot(gs[0, 0], axes_class=axisartist.Axes)
# 在第一个子图 ax0 上创建一个新的浮动坐标轴 "y=0"，它沿着 y=0 的方向
ax0.axis["y=0"] = ax0.new_floating_axis(nth_coord=0, value=0,
                                        axis_direction="bottom")
# 将新创建的坐标轴 "y=0" 设为可见状态
ax0.axis["y=0"].toggle(all=True)
# 设置坐标轴 "y=0" 的标签文本为 "y = 0"
ax0.axis["y=0"].label.set_text("y = 0")
# 将底部、顶部和右侧的默认坐标轴设为不可见
ax0.axis["bottom", "top", "right"].set_visible(False)

# 在子图网格的第二格中添加一个 axisartist.axislines.AxesZero 类型的子图 ax1
ax1 = fig.add_subplot(gs[0, 1], axes_class=axisartist.axislines.AxesZero)
# 将默认不可见的 "xzero" 坐标轴设为可见状态
ax1.axis["xzero"].set_visible(True)
# 设置坐标轴 "xzero" 的标签文本为 "Axis Zero"
ax1.axis["xzero"].label.set_text("Axis Zero")
# 将底部、顶部和右侧的默认坐标轴设为不可见
ax1.axis["bottom", "top", "right"].set_visible(False)

# 绘制一些示例数据，使用 np.arange 生成 x 值范围为 [0, 2*pi)，步长为 0.01
x = np.arange(0, 2*np.pi, 0.01)
# 在子图 ax0 上绘制 sin(x) 曲线
ax0.plot(x, np.sin(x))
# 在子图 ax1 上绘制 sin(x) 曲线
ax1.plot(x, np.sin(x))

# 显示绘制的图形
plt.show()
```