# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\polygon_selector_demo.py`

```
"""
=======================================================
Select indices from a collection using polygon selector
=======================================================

Shows how one can select indices of a polygon interactively.
"""

import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        # 获取图形画布对象
        self.canvas = ax.figure.canvas
        # 设置要选择的集合对象
        self.collection = collection
        # 设置非选定点的透明度
        self.alpha_other = alpha_other

        # 获取集合对象中的点的坐标
        self.xys = collection.get_offsets()
        # 确定点的数量
        self.Npts = len(self.xys)

        # 确保每个对象有单独的颜色
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # 创建多边形选择器，连接到当前的轴上，并指定回调函数
        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        # 用于存储被选择点的索引
        self.ind = []

    def onselect(self, verts):
        # 根据多边形的顶点创建路径对象
        path = Path(verts)
        # 根据路径判断哪些点在多边形内部
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        # 将所有点的透明度设为非选中状态的透明度
        self.fc[:, -1] = self.alpha_other
        # 将被选中的点设为完全不透明
        self.fc[self.ind, -1] = 1
        # 更新集合对象的颜色
        self.collection.set_facecolors(self.fc)
        # 绘制更新后的画布
        self.canvas.draw_idle()

    def disconnect(self):
        # 断开多边形选择器的事件连接
        self.poly.disconnect_events()
        # 将所有点的透明度恢复为完全不透明
        self.fc[:, -1] = 1
        # 更新集合对象的颜色
        self.collection.set_facecolors(self.fc)
        # 绘制更新后的画布
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    grid_size = 5
    grid_x = np.tile(np.arange(grid_size), grid_size)
    grid_y = np.repeat(np.arange(grid_size), grid_size)
    pts = ax.scatter(grid_x, grid_y)

    selector = SelectFromCollection(ax, pts)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    # 输出换行字符和文本 'Selected points:'，用于显示选定的点
    print('\nSelected points:')
    # 输出选择器对象中索引为 selector.ind 的点的坐标
    print(selector.xys[selector.ind])
# %%
#
# .. admonition:: References
#    
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#    
#    - `matplotlib.widgets.PolygonSelector`
#    - `matplotlib.path.Path`
```