# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\patch_collection.py`

```py
"""
============================
Circles, Wedges and Polygons
============================

This example demonstrates how to use `.collections.PatchCollection`.

See also :doc:`/gallery/shapes_and_collections/artist_reference`, which instead
adds each artist separately to its own Axes.
"""

# 导入所需的库
import matplotlib.pyplot as plt
import numpy as np

# 导入图形集合相关的类
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Wedge

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 定义顶点数量
resolution = 50  # the number of vertices
N = 3

# 随机生成圆的位置和半径
x = np.random.rand(N)
y = np.random.rand(N)
radii = 0.1*np.random.rand(N)
patches = []

# 创建圆形并添加到patches列表中
for x1, y1, r in zip(x, y, radii):
    circle = Circle((x1, y1), r)
    patches.append(circle)

# 再次生成随机位置和半径
x = np.random.rand(N)
y = np.random.rand(N)
radii = 0.1*np.random.rand(N)
theta1 = 360.0*np.random.rand(N)
theta2 = 360.0*np.random.rand(N)

# 创建楔形并添加到patches列表中
for x1, y1, r, t1, t2 in zip(x, y, radii, theta1, theta2):
    wedge = Wedge((x1, y1), r, t1, t2)
    patches.append(wedge)

# 添加一些楔形的限制条件
patches += [
    Wedge((.3, .7), .1, 0, 360),             # Full circle
    Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
    Wedge((.8, .3), .2, 0, 45),              # Full sector
    Wedge((.8, .3), .2, 45, 90, width=0.10),  # Ring sector
]

# 随机生成多边形的顶点，并创建多边形并添加到patches列表中
for i in range(N):
    polygon = Polygon(np.random.rand(N, 2), closed=True)
    patches.append(polygon)

# 生成随机颜色
colors = 100 * np.random.rand(len(patches))

# 创建PatchCollection对象并设置颜色和透明度，将其添加到坐标轴中
p = PatchCollection(patches, alpha=0.4)
p.set_array(colors)
ax.add_collection(p)

# 添加颜色条
fig.colorbar(p, ax=ax)

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
#    - `matplotlib.patches.Circle`
#    - `matplotlib.patches.Wedge`
#    - `matplotlib.patches.Polygon`
#    - `matplotlib.collections.PatchCollection`
#    - `matplotlib.collections.Collection.set_array`
#    - `matplotlib.axes.Axes.add_collection`
#    - `matplotlib.figure.Figure.colorbar`
```