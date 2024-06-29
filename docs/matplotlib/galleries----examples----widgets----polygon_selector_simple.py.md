# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\polygon_selector_simple.py`

```
"""
================
Polygon Selector
================

Shows how to create a polygon programmatically or interactively
"""

# 导入 matplotlib 库中的 pyplot 和 PolygonSelector 类
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

# %%
#
# To create the polygon programmatically
#
# 创建一个包含子图的图形窗口
fig, ax = plt.subplots()
# 显示图形窗口
fig.show()

# 创建一个 PolygonSelector 对象，关联到当前的子图 ax，并设置回调函数为 lambda *args: None
selector = PolygonSelector(ax, lambda *args: None)

# 设置初始的多边形顶点列表，这里包含三个顶点
selector.verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]


# %%
#
# To create the polygon interactively
#
# 创建另一个包含子图的图形窗口
fig2, ax2 = plt.subplots()
# 显示图形窗口
fig2.show()

# 创建一个 PolygonSelector 对象，关联到当前的子图 ax2，并设置回调函数为 lambda *args: None
selector2 = PolygonSelector(ax2, lambda *args: None)

# 打印交互提示信息
print("Click on the figure to create a polygon.")
print("Press the 'esc' key to start a new polygon.")
print("Try holding the 'shift' key to move all of the vertices.")
print("Try holding the 'ctrl' key to move a single vertex.")


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.PolygonSelector`
```