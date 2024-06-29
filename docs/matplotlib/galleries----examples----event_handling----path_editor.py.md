# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\path_editor.py`

```
"""
===========
Path editor
===========

Sharing events across GUIs.

This example demonstrates a cross-GUI application using Matplotlib event
handling to interact with and modify objects on the canvas.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块

from matplotlib.backend_bases import MouseButton  # 从 matplotlib.backend_bases 导入 MouseButton 类
from matplotlib.patches import PathPatch  # 从 matplotlib.patches 导入 PathPatch 类
from matplotlib.path import Path  # 从 matplotlib.path 导入 Path 类

fig, ax = plt.subplots()  # 创建一个新的图形和子图

# 定义路径数据，包括顶点和绘制代码
pathdata = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
]

codes, verts = zip(*pathdata)  # 将路径数据解压为代码和顶点
path = Path(verts, codes)  # 创建路径对象
patch = PathPatch(
    path, facecolor='green', edgecolor='yellow', alpha=0.5)  # 创建路径补丁对象
ax.add_patch(patch)  # 将路径补丁添加到子图中


class PathInteractor:
    """
    A path editor.

    Press 't' to toggle vertex markers on and off.  When vertex markers are on,
    they can be dragged with the mouse.
    """

    showverts = True  # 是否显示顶点标记
    epsilon = 5  # 最大像素距离，被认为是顶点点击的距离阈值

    def __init__(self, pathpatch):
        """
        Initialize the path interactor object.

        Parameters:
        - pathpatch : PathPatch
            The PathPatch object representing the path to interact with.
        """

        self.ax = pathpatch.axes  # 获取路径补丁所在的坐标系
        canvas = self.ax.figure.canvas  # 获取坐标系所在的画布
        self.pathpatch = pathpatch  # 设置路径补丁对象
        self.pathpatch.set_animated(True)  # 设置路径补丁对象可动画化

        x, y = zip(*self.pathpatch.get_path().vertices)  # 获取路径补丁的顶点坐标
        self.line, = ax.plot(
            x, y, marker='o', markerfacecolor='r', animated=True)  # 创建连接顶点的线条

        self._ind = None  # 当前活动顶点的索引

        # 绑定事件处理函数到画布的不同事件
        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas  # 保存画布对象

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        xy = self.pathpatch.get_path().vertices  # 获取路径补丁的顶点坐标
        xyt = self.pathpatch.get_transform().transform(xy)  # 转换为显示坐标系
        xt, yt = xyt[:, 0], xyt[:, 1]  # 提取转换后的 x 和 y 坐标
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)  # 计算到事件位置的距离
        ind = d.argmin()  # 找到最近的顶点索引
        return ind if d[ind] < self.epsilon else None  # 如果最近的距离小于阈值则返回索引，否则返回 None

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)  # 备份当前画布区域
        self.ax.draw_artist(self.pathpatch)  # 绘制路径补丁
        self.ax.draw_artist(self.line)  # 绘制连接顶点的线条
    # 鼠标按键按下的回调函数
    def on_button_press(self, event):
        """Callback for mouse button presses."""
        # 如果没有在坐标系中点击或者点击的不是左键或者不需要显示顶点，则退出
        if (event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # 获取点击位置最接近的顶点索引
        self._ind = self.get_ind_under_point(event)

    # 鼠标按键释放的回调函数
    def on_button_release(self, event):
        """Callback for mouse button releases."""
        # 如果释放的不是左键或者不需要显示顶点，则退出
        if (event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # 清空顶点索引
        self._ind = None

    # 键盘按键按下的回调函数
    def on_key_press(self, event):
        """Callback for key presses."""
        # 如果不在坐标系内按键按下，则退出
        if not event.inaxes:
            return
        # 如果按下的是 't' 键，则切换顶点显示状态，并更新图形显示
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        # 重新绘制画布
        self.canvas.draw()

    # 鼠标移动的回调函数
    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        # 如果没有选中顶点或者不在坐标系中移动鼠标或者不是左键按下或者不需要显示顶点，则退出
        if (self._ind is None
                or event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return

        # 获取路径补丁的顶点坐标
        vertices = self.pathpatch.get_path().vertices

        # 更新选中顶点的坐标为当前鼠标位置
        vertices[self._ind] = event.xdata, event.ydata

        # 更新线条数据
        self.line.set_data(zip(*vertices))

        # 恢复画布背景
        self.canvas.restore_region(self.background)
        
        # 重新绘制路径补丁和线条
        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.line)
        
        # 刷新画布部分区域
        self.canvas.blit(self.ax.bbox)
# 创建路径交互器对象，用于管理路径操作
interactor = PathInteractor(patch)
# 设置图表的标题为'drag vertices to update path'
ax.set_title('drag vertices to update path')
# 设置图表的 x 轴范围从 -3 到 4
ax.set_xlim(-3, 4)
# 设置图表的 y 轴范围从 -3 到 4
ax.set_ylim(-3, 4)

# 显示图表
plt.show()
```