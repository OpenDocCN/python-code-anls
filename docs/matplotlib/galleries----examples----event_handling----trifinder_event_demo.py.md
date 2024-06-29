# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\trifinder_event_demo.py`

```
"""
====================
Trifinder Event Demo
====================

Example showing the use of a TriFinder object.  As the mouse is moved over the
triangulation, the triangle under the cursor is highlighted and the index of
the triangle is displayed in the plot title.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入所需的库
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

# 定义更新多边形的函数，根据三角形的索引来更新
def update_polygon(tri):
    # 如果 tri 为 -1，则显示一个单点
    if tri == -1:
        points = [0, 0, 0]
    else:
        # 否则获取对应三角形的顶点索引
        points = triang.triangles[tri]
    # 根据顶点索引获取 x 和 y 坐标
    xs = triang.x[points]
    ys = triang.y[points]
    # 更新多边形的顶点坐标
    polygon.set_xy(np.column_stack([xs, ys]))

# 定义鼠标移动事件处理函数
def on_mouse_move(event):
    # 如果鼠标不在图形区域内，则 tri 设置为 -1
    if event.inaxes is None:
        tri = -1
    else:
        # 否则使用 trifinder 函数获取鼠标位置处的三角形索引
        tri = trifinder(event.xdata, event.ydata)
    # 更新多边形显示的三角形
    update_polygon(tri)
    # 设置图表的标题，显示当前所在三角形的索引
    ax.set_title(f'In triangle {tri}')
    # 重新绘制图形
    event.canvas.draw()

# 创建一个三角剖分对象
n_angles = 16
n_radii = 5
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles
x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
triang = Triangulation(x, y)
# 根据条件设置三角剖分的掩码
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

# 获取默认的 TriFinder 对象
trifinder = triang.get_trifinder()

# 设置图形和回调函数
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
ax.triplot(triang, 'bo-')  # 绘制三角剖分的边界
polygon = Polygon([[0, 0], [0, 0]], facecolor='y')  # 初始化多边形对象，用于显示
update_polygon(-1)  # 初始时显示一个点
ax.add_patch(polygon)  # 将多边形对象添加到图形中
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)  # 绑定鼠标移动事件处理函数
plt.show()  # 显示图形
```