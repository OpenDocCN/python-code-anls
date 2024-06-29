# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\ellipse_demo.py`

```
"""
============
Ellipse Demo
============

Draw many ellipses. Here individual ellipses are drawn. Compare this
to the :doc:`Ellipse collection example
</gallery/shapes_and_collections/ellipse_collection>`.
"""
# 导入 matplotlib 的 pyplot 模块并命名为 plt，用于绘图操作
import matplotlib.pyplot as plt
# 导入 numpy 并命名为 np，用于数值计算
import numpy as np

# 从 matplotlib.patches 模块导入 Ellipse 类
from matplotlib.patches import Ellipse

# 设定随机数种子以便结果可重复
np.random.seed(19680801)

# 定义要绘制的椭圆数量
NUM = 250

# 创建包含多个 Ellipse 对象的列表
# 每个 Ellipse 对象具有随机位置、宽度、高度和角度
ells = [Ellipse(xy=np.random.rand(2) * 10,
                width=np.random.rand(), height=np.random.rand(),
                angle=np.random.rand() * 360)
        for i in range(NUM)]

# 创建图形和坐标轴对象
fig, ax = plt.subplots()
# 设置坐标轴的范围和纵横比为相等
ax.set(xlim=(0, 10), ylim=(0, 10), aspect="equal")

# 将每个椭圆对象添加到坐标轴中，并设置剪切框和透明度、填充颜色
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(np.random.rand())
    e.set_facecolor(np.random.rand(3))

# 显示图形
plt.show()

# %%
# ===============
# Ellipse Rotated
# ===============
#
# Draw many ellipses with different angles.
#

# 设置椭圆的旋转角度步进为 45 度
angle_step = 45  # degrees
# 生成一系列角度值，从 0 度到 180 度，步进为 angle_step
angles = np.arange(0, 180, angle_step)

# 创建新的图形和坐标轴对象
fig, ax = plt.subplots()
# 设置坐标轴的范围和纵横比为相等
ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal")

# 根据每个角度创建一个椭圆对象，并添加到坐标轴中显示
for angle in angles:
    ellipse = Ellipse((0, 0), 4, 2, angle=angle, alpha=0.1)
    ax.add_artist(ellipse)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Ellipse`
#    - `matplotlib.axes.Axes.add_artist`
#    - `matplotlib.artist.Artist.set_clip_box`
#    - `matplotlib.artist.Artist.set_alpha`
#    - `matplotlib.patches.Patch.set_facecolor`
```