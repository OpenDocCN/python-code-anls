# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\lasso_selector_demo_sgskip.py`

```
"""
==============
Lasso Selector
==============

Interactively selecting data points with the lasso tool.

This examples plots a scatter plot. You can then select a few points by drawing
a lasso loop around the points on the graph. To draw, just click
on the graph, hold, and drag it around the points you need to select.
"""


import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

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
        """
        Initialize the selection tool.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to interact with.
        collection : `matplotlib.collections.Collection` subclass
            Collection you want to select from.
        alpha_other : float, optional
            Transparency value for non-selected points (0 <= alpha_other <= 1).
        """
        self.canvas = ax.figure.canvas  # 获取图表的画布对象
        self.collection = collection  # 保存传入的绘图集合对象
        self.alpha_other = alpha_other  # 设置非选中点的透明度

        self.xys = collection.get_offsets()  # 获取集合对象的偏移量（坐标点）
        self.Npts = len(self.xys)  # 记录坐标点数量

        # 确保每个对象有单独的颜色
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # 创建 LassoSelector 对象，绑定选择事件到 self.onselect 方法
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []  # 保存选中点的索引列表

    def onselect(self, verts):
        """
        Callback function when points are selected with Lasso.

        Parameters
        ----------
        verts : list
            List of vertices of the lasso polygon.
        """
        path = Path(verts)  # 创建 Path 对象表示 lasso 区域的路径
        self.ind = np.nonzero(path.contains_points(self.xys))[0]  # 找到落在 lasso 区域内的点的索引
        self.fc[:, -1] = self.alpha_other  # 设置所有点的透明度为 alpha_other
        self.fc[self.ind, -1] = 1  # 设置选中点的透明度为 1
        self.collection.set_facecolors(self.fc)  # 更新集合对象的颜色
        self.canvas.draw_idle()  # 更新画布显示

    def disconnect(self):
        """
        Disconnect the lasso selector from the plot.
        """
        self.lasso.disconnect_events()  # 断开 LassoSelector 的事件绑定
        self.fc[:, -1] = 1  # 恢复所有点的透明度为 1
        self.collection.set_facecolors(self.fc)  # 更新集合对象的颜色
        self.canvas.draw_idle()  # 更新画布显示


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)  # 绘制散点图
    selector = SelectFromCollection(ax, pts)  # 创建 SelectFromCollection 实例

    def accept(event):
        """
        Callback function to accept the selection and disconnect lasso.

        Parameters
        ----------
        event : key press event
            Matplotlib event object.
        """
        if event.key == "enter":  # 检测是否按下回车键
            print("Selected points:")
            print(selector.xys[selector.ind])  # 打印选中点的坐标
            selector.disconnect()  # 断开选择工具
            ax.set_title("")  # 清空图表标题
            fig.canvas.draw()  # 更新画布显示

    fig.canvas.mpl_connect("key_press_event", accept)  # 连接按键事件

    plt.show()  # 显示图表
    # 给图形对象的画布添加一个按键按下事件的监听器，事件发生时调用 accept 函数
    fig.canvas.mpl_connect("key_press_event", accept)
    # 设置当前图形对象的子图的标题为指定的文本
    ax.set_title("Press enter to accept selected points.")
    # 显示当前所有已创建的图形
    plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.LassoSelector`
#    - `matplotlib.path.Path`
```