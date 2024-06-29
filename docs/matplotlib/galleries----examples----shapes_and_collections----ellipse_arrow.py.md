# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\ellipse_arrow.py`

```
"""
===================================
Ellipse with orientation arrow demo
===================================

This demo shows how to draw an ellipse with
an orientation arrow (clockwise or counterclockwise).
Compare this to the :doc:`Ellipse collection example
</gallery/shapes_and_collections/ellipse_collection>`.
"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt

# 导入需要使用的类和函数
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

# 创建一个包含等宽比例的子图的 figure 和 axis 对象
fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

# 创建一个椭圆对象，定义位置、宽度、高度、角度、无填充颜色和边框颜色
ellipse = Ellipse(
    xy=(2, 4),
    width=30,
    height=20,
    angle=35,
    facecolor="none",
    edgecolor="b"
)
# 将椭圆对象添加到图中
ax.add_patch(ellipse)

# 获取椭圆边界顶点坐标，并应用旋转变换矩阵
vertices = ellipse.get_verts()
t = Affine2D().rotate_deg(ellipse.angle)

# 在椭圆的次要轴末端绘制一个箭头标记
ax.plot(
    vertices[0][0],
    vertices[0][1],
    color="b",
    marker=MarkerStyle(">", "full", t),
    markersize=10
)
# 注意：若要反转箭头方向，将标记类型从 > 改为 <。

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
```