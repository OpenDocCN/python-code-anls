# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\cursor_demo.py`

```
# 导入 matplotlib 库用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库用于数值计算
import numpy as np

# 导入鼠标事件相关模块
from matplotlib.backend_bases import MouseEvent

# 定义一个光标类，用于创建十字光标
class Cursor:
    """
    A cross hair cursor.
    """
    # 初始化方法，接收一个轴对象 ax
    def __init__(self, ax):
        # 设置轴对象
        self.ax = ax
        # 创建水平线对象，颜色为黑色，线宽为0.8，线型为虚线
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        # 创建垂直线对象，颜色为黑色，线宽为0.8，线型为虚线
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # 创建文本对象，位置在轴坐标系的指定位置
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    # 设置十字光标的可见性
    def set_cross_hair_visible(self, visible):
        # 检查是否需要重绘
        need_redraw = self.horizontal_line.get_visible() != visible
        # 设置水平线、垂直线和文本的可见性
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        # 返回是否需要重绘的标志
        return need_redraw

    # 鼠标移动事件的处理方法
    def on_mouse_move(self, event):
        # 如果鼠标不在轴内，将十字光标设为不可见，并检查是否需要重绘
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                # 如果需要重绘，重新绘制整个图形
                self.ax.figure.canvas.draw()
        else:
            # 如果鼠标在轴内，设置十字光标为可见状态
            self.set_cross_hair_visible(True)
            # 获取鼠标位置的数据坐标
            x, y = event.xdata, event.ydata
            # 更新水平线和垂直线的位置
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            # 设置文本内容显示当前坐标位置
            self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
            # 重新绘制图形
            self.ax.figure.canvas.draw()

# 生成数据
x = np.arange(0, 1, 0.01)
y = np.sin(2 * 2 * np.pi * x)

# 创建图形和轴对象
fig, ax = plt.subplots()
# 设置图形标题
ax.set_title('Simple cursor')
# 绘制数据点图
ax.plot(x, y, 'o')
# 创建 Cursor 类的实例对象
cursor = Cursor(ax)
# 将鼠标移动事件连接到 Cursor 类的处理方法上
fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

# 模拟鼠标移动事件，位置在 (0.5, 0.5)，用于在线文档展示
t = ax.transData
MouseEvent(
    "motion_notify_event", ax.figure.canvas, *t.transform((0.5, 0.5))
)._process()

# %%
# 使用 blitting 加速重绘
# """""""""""""""""""""""""""""""
# 这种技术将渲染的图形保存为背景图像。只有变化的艺术家（十字光标线和文本）会重新渲染。
# 它们与背景图像结合使用 blitting 技术。
#
# 这种技术明显更快。但需要更多的设置，因为背景必须在没有十字光标线的情况下保存（见下文
#```
示例中的代码已经被完整地注释了。每一行代码都有相应的注释，解释了其作用和功能。
# ``create_new_background()``). Additionally, a new background has to be
# created whenever the figure changes. This is achieved by connecting to the
# ``'draw_event'``.

class BlittedCursor:
    """
    A cross-hair cursor using blitting for faster redraw.
    """
    def __init__(self, ax):
        self.ax = ax
        self.background = None
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
        self._creating_background = False
        # Connect to the draw_event to call on_draw whenever the figure is drawn
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        """
        Event handler for draw_event, triggers creation of a new background.
        """
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        """
        Sets the visibility of the cross-hair cursor elements.

        Parameters:
        - visible (bool): Visibility flag

        Returns:
        - need_redraw (bool): Whether a redraw is needed after setting visibility
        """
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def create_new_background(self):
        """
        Creates a new background image for blitting.

        Uses copy_from_bbox and blit methods for efficient redrawing.
        """
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        """
        Event handler for mouse movement.

        Updates the position of the cross-hair cursor and redraws if necessary.
        """
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.restore_region(self.background)
                self.ax.figure.canvas.blit(self.ax.bbox)
        else:
            self.set_cross_hair_visible(True)
            # update the line positions
            x, y = event.xdata, event.ydata
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')

            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.horizontal_line)
            self.ax.draw_artist(self.vertical_line)
            self.ax.draw_artist(self.text)
            self.ax.figure.canvas.blit(self.ax.bbox)


x = np.arange(0, 1, 0.01)
y = np.sin(2 * 2 * np.pi * x)

fig, ax = plt.subplots()
ax.set_title('Blitted cursor')
ax.plot(x, y, 'o')
blitted_cursor = BlittedCursor(ax)
fig.canvas.mpl_connect('motion_notify_event', blitted_cursor.on_mouse_move)

# Simulate a mouse move to (0.5, 0.5), needed for online docs
t = ax.transData
MouseEvent(
    "motion_notify_event", ax.figure.canvas, *t.transform((0.5, 0.5))
)._process()

# %%
# Snapping to data points
# """""""""""""""""""""""
# The following cursor snaps its position to the data points of a `.Line2D`
# object.
#
# 为了避免不必要的重绘，上一次指示的数据点的索引保存在 ``self._last_index`` 中。
# 只有当鼠标移动到足够远的位置时，才会触发重新绘制，因为这时需要选择另一个数据点。
# 这样可以减少因多次重绘而导致的延迟。当然，还可以在此基础上添加位块传输（blitting）以进一步加快速度。

class SnappingCursor:
    """
    一个十字光标，它会捕捉最接近鼠标 *x* 位置的数据点所在的线。

    简单起见，这假设数据的 *x* 值是排序过的。
    """
    def __init__(self, ax, line):
        self.ax = ax
        # 水平线，表示跨度
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        # 垂直线，表示选定点
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # 获取线的数据
        self.x, self.y = line.get_data()
        # 上一次指示数据点的索引，初始化为 None
        self._last_index = None
        # 文本位置在坐标系内
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        # 检查是否需要重新绘制
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            # 如果不在坐标系内，重置上次索引为 None，并且隐藏十字光标
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            # 如果在坐标系内，显示十字光标
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # 在 self.x 中找到最接近 x 的索引
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # 仍然在同一个数据点上，无需操作
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # 更新线的位置
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            # 更新文本显示
            self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
            self.ax.figure.canvas.draw()


x = np.arange(0, 1, 0.01)
y = np.sin(2 * 2 * np.pi * x)

fig, ax = plt.subplots()
ax.set_title('Snapping cursor')
line, = ax.plot(x, y, 'o')
snap_cursor = SnappingCursor(ax, line)
fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)

# 模拟鼠标移动到 (0.5, 0.5)，文档演示需要
t = ax.transData
MouseEvent(
    "motion_notify_event", ax.figure.canvas, *t.transform((0.5, 0.5))
)._process()

plt.show()
```