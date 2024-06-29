# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\poly_editor.py`

```py
"""
===========
Poly Editor
===========

This is an example to show how to build cross-GUI applications using
Matplotlib event handling to interact with objects on the canvas.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入所需的库
import numpy as np  # 导入 NumPy 库

from matplotlib.artist import Artist  # 导入 Artist 类
from matplotlib.lines import Line2D  # 导入 Line2D 类


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance from the point *p* to the segment (*s0*, *s1*), where
    *p*, *s0*, *s1* are ``[x, y]`` arrays.
    """
    s01 = s1 - s0  # 计算线段 s0 到 s1 的向量差
    s0p = p - s0   # 计算点 p 到 s0 的向量差
    if (s01 == 0).all():  # 如果 s01 向量全为零向量
        return np.hypot(*s0p)  # 返回 s0 到 p 的欧几里得距离
    # 投影到线段上，不超过线段端点
    p1 = s0 + np.clip((s0p @ s01) / (s01 @ s01), 0, 1) * s01
    return np.hypot(*(p - p1))  # 返回 p 到投影点 p1 的欧几里得距离


class PolygonInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True  # 控制是否显示顶点标记
    epsilon = 5       # 视为顶点命中的最大像素距离

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax  # 设置绘图区域
        canvas = poly.figure.canvas
        self.poly = poly  # 多边形对象

        x, y = zip(*self.poly.xy)  # 提取多边形的顶点坐标
        self.line = Line2D(x, y,
                           marker='o', markerfacecolor='r',
                           animated=True)  # 创建一个连接多边形顶点的线段
        self.ax.add_line(self.line)  # 将线段添加到绘图区域中

        self.cid = self.poly.add_callback(self.poly_changed)  # 注册多边形变化时的回调函数
        self._ind = None  # 当前活动的顶点索引

        # 绑定事件处理函数
        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas  # 保存画布对象

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)  # 备份绘图区域的背景
        self.ax.draw_artist(self.poly)  # 绘制多边形
        self.ax.draw_artist(self.line)  # 绘制顶点连接线段
        # 这里不需要执行 blit 操作，在更新屏幕之前会自动执行
    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is changed."""
        # 获取当前线条的可见性状态
        vis = self.line.get_visible()
        # 将 poly 的属性更新到线条中（除了可见性）
        Artist.update_from(self.line, poly)
        # 恢复线条的可见性状态（不使用 poly 的可见性状态）
        self.line.set_visible(vis)

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # 将多边形顶点坐标转换为显示坐标系
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        # 计算事件位置与各顶点距离的欧几里得距离
        d = np.hypot(xt - event.x, yt - event.y)
        # 找到距离最近的顶点的索引
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        # 如果最近顶点与事件位置的距离超过阈值 epsilon，则返回 None
        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        # 如果不显示顶点，则返回
        if not self.showverts:
            return
        # 如果事件不在坐标轴内，则返回
        if event.inaxes is None:
            return
        # 如果按下的不是左键，则返回
        if event.button != 1:
            return
        # 获取最近的顶点索引并存储在 self._ind 中
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        # 如果不显示顶点，则返回
        if not self.showverts:
            return
        # 如果释放的不是左键，则返回
        if event.button != 1:
            return
        # 清空顶点索引 self._ind
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        # 如果事件不在坐标轴内，则返回
        if not event.inaxes:
            return
        # 根据按键进行不同的操作
        if event.key == 't':
            # 切换顶点显示状态
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            # 如果隐藏顶点，则清空顶点索引 self._ind
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            # 删除最近的顶点
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            # 在最近线段中插入新顶点
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        # 如果线条已经过时，则重新绘制画布
        if self.line.stale:
            self.canvas.draw_idle()
    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        # 如果不显示顶点，则退出
        if not self.showverts:
            return
        # 如果没有选定顶点，则退出
        if self._ind is None:
            return
        # 如果鼠标事件不在坐标系内，则退出
        if event.inaxes is None:
            return
        # 如果鼠标按键不是左键，则退出
        if event.button != 1:
            return
        # 获取鼠标当前的数据坐标
        x, y = event.xdata, event.ydata

        # 更新多边形顶点的坐标
        self.poly.xy[self._ind] = x, y
        # 如果更新的是第一个顶点，同时更新闭合多边形的最后一个顶点
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        # 如果更新的是最后一个顶点，同时更新闭合多边形的第一个顶点
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        # 重新绘制多边形和连接线
        self.line.set_data(zip(*self.poly.xy))

        # 恢复背景
        self.canvas.restore_region(self.background)
        # 重新绘制多边形
        self.ax.draw_artist(self.poly)
        # 重新绘制连接线
        self.ax.draw_artist(self.line)
        # 在画布上部分更新被修改的区域
        self.canvas.blit(self.ax.bbox)
if __name__ == '__main__':
    # 检查是否在主程序中执行该脚本

    import matplotlib.pyplot as plt
    # 导入 matplotlib 库中的 pyplot 模块，并重命名为 plt

    from matplotlib.patches import Polygon
    # 从 matplotlib 库的 patches 模块中导入 Polygon 类

    theta = np.arange(0, 2*np.pi, 0.1)
    # 创建一个包含从 0 到 2π 的角度值的数组，步长为 0.1
    r = 1.5
    # 设置半径为 1.5

    xs = r * np.cos(theta)
    # 计算每个角度对应的 x 坐标，使用余弦函数
    ys = r * np.sin(theta)
    # 计算每个角度对应的 y 坐标，使用正弦函数

    poly = Polygon(np.column_stack([xs, ys]), animated=True)
    # 创建一个多边形对象 Polygon，使用 np.column_stack 将 xs 和 ys 数组堆叠起来作为多边形的顶点

    fig, ax = plt.subplots()
    # 创建一个新的图形窗口和一个子图 axes 对象
    ax.add_patch(poly)
    # 将多边形对象 poly 添加到子图 ax 上作为一个补丁

    p = PolygonInteractor(ax, poly)
    # 创建一个交互式多边形对象 PolygonInteractor，传入 ax 和 poly 对象

    ax.set_title('Click and drag a point to move it')
    # 设置子图 ax 的标题
    ax.set_xlim((-2, 2))
    # 设置子图 ax 的 x 轴显示范围
    ax.set_ylim((-2, 2))
    # 设置子图 ax 的 y 轴显示范围

    plt.show()
    # 显示图形窗口和其中的子图
```