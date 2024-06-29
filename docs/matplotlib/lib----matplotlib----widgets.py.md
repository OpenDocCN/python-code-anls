# `D:\src\scipysrc\matplotlib\lib\matplotlib\widgets.py`

```
"""
GUI neutral widgets
===================

Widgets that are designed to work for any of the GUI backends.
All of these widgets require you to predefine an `~.axes.Axes`
instance and pass that as the first parameter.  Matplotlib doesn't try to
be too smart with respect to layout -- you will have to figure out how
wide and tall you want your Axes to be to accommodate your widget.
"""

# 引入上下文管理工具
from contextlib import ExitStack
# 引入用于复制对象的模块
import copy
# 引入用于生成迭代器的模块
import itertools
# 引入用于处理整数和浮点数的模块
from numbers import Integral, Number

# 引入循环器，用于控制属性的迭代
from cycler import cycler
# 引入 numpy 库
import numpy as np

# 引入 matplotlib 库
import matplotlib as mpl
# 从当前目录中引入多个模块
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
               text as mtext, ticker, transforms)
# 从 lines 模块引入 Line2D 类
from .lines import Line2D
# 从 patches 模块引入 Rectangle, Ellipse, Polygon 类
from .patches import Rectangle, Ellipse, Polygon
# 从 transforms 模块引入 TransformedPatchPath, Affine2D 类
from .transforms import TransformedPatchPath, Affine2D


class LockDraw:
    """
    Some widgets, like the cursor, draw onto the canvas, and this is not
    desirable under all circumstances, like when the toolbar is in zoom-to-rect
    mode and drawing a rectangle.  To avoid this, a widget can acquire a
    canvas' lock with ``canvas.widgetlock(widget)`` before drawing on the
    canvas; this will prevent other widgets from doing so at the same time (if
    they also try to acquire the lock first).
    """

    def __init__(self):
        self._owner = None

    def __call__(self, o):
        """Reserve the lock for *o*."""
        # 检查是否已经被锁定，如果是则抛出异常
        if not self.available(o):
            raise ValueError('already locked')
        # 将当前对象设置为拥有者
        self._owner = o

    def release(self, o):
        """Release the lock from *o*."""
        # 检查是否拥有该锁，如果没有则抛出异常
        if not self.available(o):
            raise ValueError('you do not own this lock')
        # 释放当前对象的锁定状态
        self._owner = None

    def available(self, o):
        """Return whether drawing is available to *o*."""
        # 如果没有被锁定或者当前对象是拥有者，则返回 True，否则返回 False
        return not self.locked() or self.isowner(o)

    def isowner(self, o):
        """Return whether *o* owns this lock."""
        # 返回当前对象是否是指定对象 o 的拥有者
        return self._owner is o

    def locked(self):
        """Return whether the lock is currently held by an owner."""
        # 返回当前锁是否被任何对象拥有
        return self._owner is not None


class Widget:
    """
    Abstract base class for GUI neutral widgets.
    """
    # 默认允许绘制
    drawon = True
    # 默认允许事件处理
    eventson = True
    # 默认激活状态为 True
    _active = True

    def set_active(self, active):
        """Set whether the widget is active."""
        # 设置当前控件的激活状态
        self._active = active

    def get_active(self):
        """Get whether the widget is active."""
        # 获取当前控件的激活状态
        return self._active

    # 设置属性 active 的属性
    active = property(get_active, set_active, doc="Is the widget active?")

    def ignore(self, event):
        """
        Return whether *event* should be ignored.

        This method should be called at the beginning of any event callback.
        """
        # 如果控件不活跃，则忽略事件
        return not self.active
    # 定义一个方法 `_changed_canvas`，用于检查画布是否被更改
    def _changed_canvas(self):
        """
        Someone has switched the canvas on us!

        This happens if `savefig` needs to save to a format the previous
        backend did not support (e.g. saving a figure using an Agg based
        backend saved to a vector format).

        Returns
        -------
        bool
           True if the canvas has been changed.

        """
        # 返回一个布尔值，指示画布是否被更改
        return self.canvas is not self.ax.figure.canvas
class AxesWidget(Widget):
    """
    Widget connected to a single `~matplotlib.axes.Axes`.

    To guarantee that the widget remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the widget.  If there are no references to the widget
    object it may be garbage collected which will disconnect the callbacks.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    canvas : `~matplotlib.backend_bases.FigureCanvasBase`
        The parent figure canvas for the widget.
    active : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, ax):
        # 初始化方法，传入一个 Axes 对象作为参数
        self.ax = ax
        # 获取该 Axes 对象所属的 figure 的 canvas
        self.canvas = ax.figure.canvas
        # 用于存储回调函数的 ID 的列表
        self._cids = []

    def connect_event(self, event, callback):
        """
        Connect a callback function with an event.

        This should be used in lieu of ``figure.canvas.mpl_connect`` since this
        function stores callback ids for later clean up.
        """
        # 连接事件和回调函数，并将回调函数的 ID 存储起来
        cid = self.canvas.mpl_connect(event, callback)
        self._cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        # 断开所有由该 widget 创建的事件的连接
        for c in self._cids:
            self.canvas.mpl_disconnect(c)

    def _get_data_coords(self, event):
        """Return *event*'s data coordinates in this widget's Axes."""
        # 返回事件在当前 widget 的 Axes 中的数据坐标
        # 处理 event.inaxes 不等于 self.ax 的情况，避免数据坐标转换错误
        return ((event.xdata, event.ydata) if event.inaxes is self.ax
                else self.ax.transData.inverted().transform((event.x, event.y)))
    def __init__(self, ax, label, image=None,
                 color='0.85', hovercolor='0.95', *, useblit=True):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            将按钮放置到的 `~.axes.Axes` 实例。
        label : str
            按钮的文本内容。
        image : array-like or PIL Image
            要放置在按钮中的图像，如果不为 *None*。该参数直接传递给 `~.axes.Axes.imshow`。
        color : :mpltype:`color`
            按钮未激活时的颜色。
        hovercolor : :mpltype:`color`
            鼠标悬停在按钮上时的颜色。
        useblit : bool, default: True
            如果后端支持，则使用 blitting 进行更快的绘制。
            详细信息请参见教程 :ref:`blitting`。

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        if image is not None:
            # 如果提供了图像，则在 Axes 上显示该图像
            ax.imshow(image)
        
        # 在指定位置创建文本标签
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        # 根据后端是否支持 blitting 来决定是否使用 blitting
        self._useblit = useblit and self.canvas.supports_blit

        # 创建一个事件回调注册表，用于处理 'clicked' 信号
        self._observers = cbook.CallbackRegistry(signals=["clicked"])

        # 连接按钮事件处理函数
        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        
        # 禁用 Axes 的导航功能
        ax.set_navigate(False)
        
        # 设置 Axes 的背景颜色
        ax.set_facecolor(color)
        
        # 清空 X 和 Y 轴的刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 记录按钮的颜色和悬停颜色
        self.color = color
        self.hovercolor = hovercolor

    def _click(self, event):
        # 如果事件未启用或忽略事件或事件不在 Axes 区域内，则返回
        if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0]:
            return
        # 如果鼠标未捕获 Axes，捕获之
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)

    def _release(self, event):
        # 如果忽略事件或鼠标未捕获 Axes，则返回
        if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
            return
        # 释放鼠标捕获，并且如果事件启用且在 Axes 区域内，则处理 'clicked' 信号
        event.canvas.release_mouse(self.ax)
        if self.eventson and self.ax.contains(event)[0]:
            self._observers.process('clicked', event)

    def _motion(self, event):
        # 如果忽略事件，则返回
        if self.ignore(event):
            return
        # 根据鼠标是否在 Axes 区域内决定按钮颜色为悬停颜色或普通颜色
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        # 如果按钮颜色与当前背景色不同，则更新按钮颜色
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            # 如果允许绘制，则根据使用 blitting 的选择来更新绘图
            if self.drawon:
                if self._useblit:
                    self.ax.draw_artist(self.ax)
                    self.canvas.blit(self.ax.bbox)
                else:
                    self.canvas.draw()

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
        # 将回调函数与 'clicked' 信号连接起来
        return self._observers.connect('clicked', lambda event: func(event))
    # 定义一个方法 disconnect，用于移除具有连接 ID *cid* 的回调函数。
    def disconnect(self, cid):
        """Remove the callback function with connection id *cid*."""
        # 调用内部属性 _observers 的 disconnect 方法，传入连接 ID *cid*。
        self._observers.disconnect(cid)
class SliderBase(AxesWidget):
    """
    The base class for constructing Slider widgets. Not intended for direct
    usage.

    For the slider to remain responsive you must maintain a reference to it.
    """
    def __init__(self, ax, orientation, closedmin, closedmax,
                 valmin, valmax, valfmt, dragging, valstep):
        # 检查是否尝试将滑块添加到 3D 坐标轴，若是则引发异常
        if ax.name == '3d':
            raise ValueError('Sliders cannot be added to 3D Axes')

        super().__init__(ax)
        # 检查滑块方向参数是否有效
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)

        # 设置实例变量
        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        self.drag_active = False
        self.valfmt = valfmt

        # 根据方向设置坐标轴的界限
        if orientation == "vertical":
            ax.set_ylim((valmin, valmax))
            axis = ax.yaxis
        else:
            ax.set_xlim((valmin, valmax))
            axis = ax.xaxis

        # 获取主要刻度格式化对象并设置其属性
        self._fmt = axis.get_major_formatter()
        if not isinstance(self._fmt, ticker.ScalarFormatter):
            self._fmt = ticker.ScalarFormatter()
            self._fmt.set_axis(axis)
        self._fmt.set_useOffset(False)  # 禁用额外的偏移
        self._fmt.set_useMathText(True)  # 在乘法偏移前显示 x 符号

        # 关闭坐标轴和导航功能
        ax.set_axis_off()
        ax.set_navigate(False)

        # 连接事件处理器
        self.connect_event("button_press_event", self._update)
        self.connect_event("button_release_event", self._update)
        if dragging:
            self.connect_event("motion_notify_event", self._update)
        self._observers = cbook.CallbackRegistry(signals=["changed"])

    def _stepped_value(self, val):
        """Return *val* coerced to closest number in the ``valstep`` grid."""
        # 将值舍入到最接近的 valstep 网格点
        if isinstance(self.valstep, Number):
            val = (self.valmin
                   + round((val - self.valmin) / self.valstep) * self.valstep)
        elif self.valstep is not None:
            valstep = np.asanyarray(self.valstep)
            if valstep.ndim != 1:
                raise ValueError(
                    f"valstep must have 1 dimension but has {valstep.ndim}"
                )
            val = valstep[np.argmin(np.abs(valstep - val))]
        return val

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*.

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed.
        """
        # 断开指定连接 ID 的观察者
        self._observers.disconnect(cid)

    def reset(self):
        """Reset the slider to the initial value."""
        # 将滑块重置为初始值
        if np.any(self.val != self.valinit):
            self.set_val(self.valinit)
    def _value_in_bounds(self, val):
        """确保 *val* 在指定的范围内。"""
        # 使用步进函数处理值
        val = self._stepped_value(val)

        # 检查是否小于最小值
        if val <= self.valmin:
            # 如果不允许超出最小值，则返回
            if not self.closedmin:
                return
            val = self.valmin
        # 检查是否大于最大值
        elif val >= self.valmax:
            # 如果不允许超出最大值，则返回
            if not self.closedmax:
                return
            val = self.valmax

        # 如果定义了 slidermin，并且 val 小于或等于 slidermin 的值
        if self.slidermin is not None and val <= self.slidermin.val:
            # 如果不允许超出 slidermin 的值，则返回
            if not self.closedmin:
                return
            val = self.slidermin.val

        # 如果定义了 slidermax，并且 val 大于或等于 slidermax 的值
        if self.slidermax is not None and val >= self.slidermax.val:
            # 如果不允许超出 slidermax 的值，则返回
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _update(self, event):
        """更新滑块的位置。"""
        # 如果事件被忽略或者不是左键点击事件，则返回
        if self.ignore(event) or event.button != 1:
            return

        # 如果是按下鼠标左键事件，并且在滑块上点击
        if event.name == 'button_press_event' and self.ax.contains(event)[0]:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        # 如果不是在滑块上按下鼠标左键事件，则返回
        if not self.drag_active:
            return

        # 如果是释放鼠标左键事件或者按下鼠标左键事件但不在滑块上
        if (event.name == 'button_release_event'
              or event.name == 'button_press_event' and not self.ax.contains(event)[0]):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        # 获取事件的数据坐标
        xdata, ydata = self._get_data_coords(event)
        # 根据方向选择 xdata 或 ydata 进行值的边界检查
        val = self._value_in_bounds(
            xdata if self.orientation == 'horizontal' else ydata)
        # 如果 val 不是 None 或者当前值 self.val，则设置新的值
        if val not in [None, self.val]:
            self.set_val(val)

    def _format(self, val):
        """对 *val* 进行美化打印。"""
        # 如果指定了 valfmt，则使用其格式化 val
        if self.valfmt is not None:
            return self.valfmt % val
        else:
            _, s, _ = self._fmt.format_ticks([self.valmin, val, self.valmax])
            # _fmt.get_offset 实际上是乘法因子（如果有的话）
            return s + self._fmt.get_offset()

    def set_val(self, val):
        """
        设置滑块的值为 *val*。

        Parameters
        ----------
        val : float
            新的滑块值。
        """
        # 如果是垂直方向滑块
        if self.orientation == 'vertical':
            # 设置滑块的高度
            self.poly.set_height(val - self.poly.get_y())
            self._handle.set_ydata([val])
        else:
            # 设置滑块的宽度
            self.poly.set_width(val - self.poly.get_x())
            self._handle.set_xdata([val])
        # 更新显示滑块值的文本
        self.valtext.set_text(self._format(val))
        # 如果设置了绘制标志，则在画布上绘制更新
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        # 更新当前滑块值
        self.val = val
        # 如果设置了事件处理标志，则处理观察者的 'changed' 事件
        if self.eventson:
            self._observers.process('changed', val)
    def on_changed(self, func):
        """
        将 *func* 连接作为滑块数值变化的回调函数。

        Parameters
        ----------
        func : callable
            当滑块数值变化时调用的函数。
            该函数必须接受一个单独的浮点数作为其参数。

        Returns
        -------
        int
            连接的 ID（可用于断开 *func* 的连接）。
        """
        # 使用 self._observers 对象连接 'changed' 事件，传入 lambda 表达式作为回调函数
        return self._observers.connect('changed', lambda val: func(val))
class RangeSlider(SliderBase):
    """
    A slider representing a range of floating point values. Defines the min and
    max of the range via the *val* attribute as a tuple of (min, max).

    Create a slider that defines a range contained within [*valmin*, *valmax*]
    in Axes *ax*. For the slider to remain responsive you must maintain a
    reference to it. Call :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tuple of float
        Slider value.
    """

    def __init__(
        self,
        ax,
        label,
        valmin,
        valmax,
        *,
        valinit=None,
        valfmt=None,
        closedmin=True,
        closedmax=True,
        dragging=True,
        valstep=None,
        orientation="horizontal",
        track_color='lightgrey',
        handle_style=None,
        **kwargs,
    ):
        # 初始化函数，用于创建 RangeSlider 对象
        super().__init__(ax=ax, valmin=valmin, valmax=valmax, valinit=valinit,
                         valfmt=valfmt, dragging=dragging, valstep=valstep,
                         orientation=orientation, **kwargs)
        self.val = (valmin, valmax)  # 设置初始的滑块值为给定的最小和最大值
        self.valmin = valmin  # 设置最小值
        self.valmax = valmax  # 设置最大值
        self.closedmin = closedmin  # 设置是否闭合最小值
        self.closedmax = closedmax  # 设置是否闭合最大值

        # 创建滑块的多边形表示
        self.poly = ax.fill([0, 0, 0, 0], [0, 1, 1, 0], closed=True,
                            facecolor=track_color, edgecolor='w', alpha=0.5)[0]
        self._update_selection_poly(valmin, valmax)  # 更新滑块的形状以反映当前的值范围

        if handle_style is None:
            handle_style = dict(facecolor='blue', edgecolor='white')

        # 创建滑块的可移动手柄
        self.handles = [ax.plot([0], [0], marker='o', markersize=10,
                                visible=False, **handle_style)[0],
                        ax.plot([0], [0], marker='o', markersize=10,
                                visible=False, **handle_style)[0]]
        self.set_label(label)  # 设置滑块的标签文字

    def _update_selection_poly(self, vmin, vmax):
        """
        Update the vertices of the *self.poly* slider in-place
        to cover the data range *vmin*, *vmax*.
        """
        # 更新多边形顶点位置以覆盖给定的数据范围vmin, vmax
        verts = self.poly.xy
        if self.orientation == "vertical":
            verts[0] = verts[4] = .25, vmin
            verts[1] = .25, vmax
            verts[2] = .75, vmax
            verts[3] = .75, vmin
        else:
            verts[0] = verts[4] = vmin, .25
            verts[1] = vmin, .75
            verts[2] = vmax, .75
            verts[3] = vmax, .25

    def _min_in_bounds(self, min):
        """Ensure the new min value is between valmin and self.val[1]."""
        if min <= self.valmin:
            if not self.closedmin:
                return self.val[0]
            min = self.valmin

        if min > self.val[1]:
            min = self.val[1]
        return self._stepped_value(min)

    def _max_in_bounds(self, max):
        """Ensure the new max value is between valmax and self.val[0]."""
        if max >= self.valmax:
            if not self.closedmax:
                return self.val[1]
            max = self.valmax

        if max <= self.val[0]:
            max = self.val[0]
        return self._stepped_value(max)

    def _value_in_bounds(self, vals):
        """Clip min, max values to the bounds."""
        return (self._min_in_bounds(vals[0]), self._max_in_bounds(vals[1]))

    def _update_val_from_pos(self, pos):
        """Update the slider value based on a given position."""
        idx = np.argmin(np.abs(self.val - pos))
        if idx == 0:
            val = self._min_in_bounds(pos)
            self.set_min(val)
        else:
            val = self._max_in_bounds(pos)
            self.set_max(val)
        if self._active_handle:
            if self.orientation == "vertical":
                self._active_handle.set_ydata([val])
            else:
                self._active_handle.set_xdata([val])
    def _update(self, event):
        """Update the slider position."""
        # 检查是否忽略事件或者不是左键点击事件，如果是，则返回
        if self.ignore(event) or event.button != 1:
            return

        # 如果是鼠标按下事件并且在轴内点击了，则标记拖动状态为活动，并抓取鼠标
        if event.name == "button_press_event" and self.ax.contains(event)[0]:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        # 如果未处于拖动状态，则返回
        if not self.drag_active:
            return

        # 如果是鼠标释放事件或者是鼠标按下事件但不在轴内点击，则标记拖动状态为非活动，
        # 释放鼠标，并重置活动的手柄为 None，然后返回
        if (event.name == "button_release_event"
              or event.name == "button_press_event" and not self.ax.contains(event)[0]):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            self._active_handle = None
            return

        # 确定哪个手柄被抓取
        xdata, ydata = self._get_data_coords(event)
        handle_index = np.argmin(np.abs(
            [h.get_xdata()[0] - xdata for h in self._handles]
            if self.orientation == "horizontal" else
            [h.get_ydata()[0] - ydata for h in self._handles]))
        handle = self._handles[handle_index]

        # 如果抓取的手柄不是当前活动的手柄，则更新当前活动的手柄
        if handle is not self._active_handle:
            self._active_handle = handle

        # 根据当前位置更新值
        self._update_val_from_pos(xdata if self.orientation == "horizontal" else ydata)

    def _format(self, val):
        """Pretty-print *val*."""
        # 如果定义了 valfmt，则使用其格式化 val；否则，使用默认格式化方式
        if self.valfmt is not None:
            return f"({self.valfmt % val[0]}, {self.valfmt % val[1]})"
        else:
            # 调用 _fmt.format_ticks 格式化 ticks，处理偏移并返回格式化后的字符串
            _, s1, s2, _ = self._fmt.format_ticks(
                [self.valmin, *val, self.valmax]
            )
            s1 += self._fmt.get_offset()
            s2 += self._fmt.get_offset()
            return f"({s1}, {s2})"

    def set_min(self, min):
        """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
            设置滑块的下限值为 min。
        """
        self.set_val((min, self.val[1]))

    def set_max(self, max):
        """
        Set the upper value of the slider to *max*.

        Parameters
        ----------
        max : float
            设置滑块的上限值为 max。
        """
        self.set_val((self.val[0], max))
    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
            要设置的滑块值，可以是包含浮点数的元组或类数组对象
        """
        # 对传入的值进行排序
        val = np.sort(val)
        # 检查值的形状是否符合预期（必须是长度为2的元组）
        _api.check_shape((2,), val=val)
        # 重置当前值，以便 _value_in_bounds() 函数可以正常工作
        self.val = (self.valmin, self.valmax)
        # 根据给定的值计算在界限内的最小值和最大值
        vmin, vmax = self._value_in_bounds(val)
        # 更新选择范围的多边形表示
        self._update_selection_poly(vmin, vmax)
        
        # 根据滑块的方向更新对应的句柄位置
        if self.orientation == "vertical":
            self._handles[0].set_ydata([vmin])
            self._handles[1].set_ydata([vmax])
        else:
            self._handles[0].set_xdata([vmin])
            self._handles[1].set_xdata([vmax])

        # 更新显示当前值的文本
        self.valtext.set_text(self._format((vmin, vmax)))

        # 如果设置为实时绘制，则立即刷新绘图
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        
        # 更新当前的值为计算得到的最小值和最大值
        self.val = (vmin, vmax)
        
        # 如果启用事件处理，则调用相应的回调函数
        if self.eventson:
            self._observers.process("changed", (vmin, vmax))

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            当滑块值发生变化时调用的回调函数。该函数必须接受一个包含两个浮点数的元组作为参数。

        Returns
        -------
        int
            连接的 ID（可用于断开与 *func* 的连接）。
        """
        # 将给定的函数作为回调函数连接到 'changed' 事件
        return self._observers.connect('changed', lambda val: func(val))
def _expand_text_props(props):
    # 使用 cbook.normalize_kwargs 函数将 props 规范化为适用于 Text 类的关键字参数
    props = cbook.normalize_kwargs(props, mtext.Text)
    # 如果 props 非空，则使用其创建一个 cycler 对象，否则返回一个无限重复空字典的迭代器
    return cycler(**props)() if props else itertools.repeat({})


class CheckButtons(AxesWidget):
    r"""
    A GUI neutral set of check buttons.

    For the check buttons to remain responsive you must keep a
    reference to this object.

    Connect to the CheckButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    labels : list of `~matplotlib.text.Text`
        The text label objects of the check buttons.
    """

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        # 如果事件被忽略或者画布发生变化，则直接返回
        if self.ignore(event) or self._changed_canvas():
            return
        # 复制当前 Axes 区域的背景，并在画布上绘制按钮
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._checks)

    def _clicked(self, event):
        # 如果事件被忽略、按键不是鼠标左键、或者事件不在 Axes 区域内，则直接返回
        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        # 获取包含事件的帧和文本的索引
        idxs = [
            *self._frames.contains(event)[1]["ind"],  # 帧的索引
            *[i for i, text in enumerate(self.labels) if text.contains(event)[0]]  # 包含事件的文本的索引
        ]
        # 如果有匹配的索引
        if idxs:
            # 获取帧的偏移转换并计算坐标
            coords = self._frames.get_offset_transform().transform(
                self._frames.get_offsets())
            # 设置活动状态为距离事件最近的索引
            self.set_active(
                idxs[(((event.x, event.y) - coords[idxs]) ** 2).sum(-1).argmin()])

    def set_label_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels.
        """
        # 检查 props 的类型是否为 dict
        _api.check_isinstance(dict, props=props)
        # 根据 props 设置文本标签的属性
        props = _expand_text_props(props)
        for text, prop in zip(self.labels, props):
            text.update(prop)

    def set_frame_props(self, props):
        """
        Set properties of the check button frames.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button frames.
        """
        # 检查 props 的类型是否为 dict
        _api.check_isinstance(dict, props=props)
        # 如果 props 中包含 's' 键，将其转换为 'sizes' 键以保持构造函数的一致性
        if 's' in props:
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        # 更新帧的属性
        self._frames.update(props)
    def set_check_props(self, props):
        """
        Set properties of the check button checks.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button check.
        """
        # 检查 props 是否为字典类型
        _api.check_isinstance(dict, props=props)
        
        # 如果 props 中有 's' 键，将其值广播到与 self.labels 长度相同的数组，并用 'sizes' 键替换 's' 键
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        
        # 获取当前的按钮状态
        actives = self.get_status()
        
        # 更新 self._checks 的属性
        self._checks.update(props)
        
        # 如果提供了新的颜色，重新应用状态
        self._init_status(actives)

    def set_active(self, index, state=None):
        """
        Modify the state of a check button by index.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            Index of the check button to toggle.

        state : bool, optional
            If a boolean value, set the state explicitly. If no value is
            provided, the state is toggled.

        Raises
        ------
        ValueError
            If *index* is invalid.
        TypeError
            If *state* is not boolean.
        """
        # 检查 index 是否在有效范围内
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid CheckButton index: {index}')
        
        # 检查 state 是否为布尔类型或 None
        _api.check_isinstance((bool, None), state=state)

        # 定义透明颜色
        invisible = colors.to_rgba('none')

        # 获取当前按钮的面颜色
        facecolors = self._checks.get_facecolor()
        
        # 如果 state 为 None，则根据当前状态和透明颜色来设置新的状态
        if state is None:
            state = colors.same_color(facecolors[index], invisible)
        
        # 根据 state 设置按钮的面颜色
        facecolors[index] = self._active_check_colors[index] if state else invisible
        
        # 更新按钮的面颜色
        self._checks.set_facecolor(facecolors)

        # 如果 drawon 为 True，则绘制按钮
        if self.drawon:
            if self._useblit:
                # 使用 blitting 技术绘制按钮
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                self.ax.draw_artist(self._checks)
                self.canvas.blit(self.ax.bbox)
            else:
                # 直接绘制到画布上
                self.canvas.draw()

        # 如果 eventson 为 True，则处理触发点击事件
        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    def _init_status(self, actives):
        """
        Initialize properties to match active status.

        The user may have passed custom colours in *check_props* to the
        constructor, or to `.set_check_props`, so we need to modify the
        visibility after getting whatever the user set.
        """
        # 获取当前按钮的面颜色
        self._active_check_colors = self._checks.get_facecolor()
        
        # 如果只有一个颜色，将其复制多次以匹配 actives 的长度
        if len(self._active_check_colors) == 1:
            self._active_check_colors = np.repeat(self._active_check_colors,
                                                  len(actives), axis=0)
        
        # 根据 actives 设置按钮的面颜色
        self._checks.set_facecolor(
            [ec if active else "none"
             for ec, active in zip(self._active_check_colors, actives)])
    # 清除所有复选框的选中状态
    def clear(self):
        """Uncheck all checkboxes."""

        # 将所有复选框的背景颜色设置为无色
        self._checks.set_facecolor(['none'] * len(self._active_check_colors))

        # 如果存在属性 '_lines'
        if hasattr(self, '_lines'):
            # 对于每一对 l1, l2 在 self._lines 中
            for l1, l2 in self._lines:
                # 设置 l1 和 l2 不可见
                l1.set_visible(False)
                l2.set_visible(False)

        # 如果需要绘制
        if self.drawon:
            # 重新绘制画布
            self.canvas.draw()

        # 如果事件开启
        if self.eventson:
            # 调用处理器，没有标签参数，因为所有复选框都被清除了
            self._observers.process('clicked', None)

    # 获取所有复选框的状态（选中为 True，未选中为 False）的列表
    def get_status(self):
        """
        Return a list of the status (True/False) of all of the check buttons.
        """
        return [not colors.same_color(color, colors.to_rgba("none"))
                for color in self._checks.get_facecolors()]

    # 返回当前由用户选中的标签的列表
    def get_checked_labels(self):
        """Return a list of labels currently checked by user."""

        return [l.get_text() for l, box_checked in
                zip(self.labels, self.get_status())
                if box_checked]

    # 将回调函数 *func* 连接到按钮点击事件
    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
        # 连接到观察者对象的 'clicked' 事件，传递标签文本给 func 函数
        return self._observers.connect('clicked', lambda text: func(text))

    # 断开与给定连接 id *cid* 相关的观察者
    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)
    """
    A GUI neutral text input box.

    For the text box to remain responsive you must keep a reference to it.

    Call `.on_text_change` to be updated whenever the text changes.

    Call `.on_submit` to be updated whenever the user hits enter or
    leaves the text entry field.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    label : `~matplotlib.text.Text`
        The text label associated with the text box.

    color : :mpltype:`color`
        The color of the text box when not hovering.
    hovercolor : :mpltype:`color`
        The color of the text box when hovering.
    """

    def __init__(self, ax, label, initial='', *,
                 color='.95', hovercolor='1', label_pad=.01,
                 textalignment="left"):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            Label for this text box.
        initial : str
            Initial value in the text box.
        color : :mpltype:`color`
            The color of the box.
        hovercolor : :mpltype:`color`
            The color of the box when the mouse is over it.
        label_pad : float
            The distance between the label and the right side of the textbox.
        textalignment : {'left', 'center', 'right'}
            The horizontal location of the text.
        """
        super().__init__(ax)

        self._text_position = _api.check_getitem(
            {"left": 0.05, "center": 0.5, "right": 0.95},
            textalignment=textalignment)  # 设置文本的水平对齐位置

        self.label = ax.text(
            -label_pad, 0.5, label, transform=ax.transAxes,
            verticalalignment='center', horizontalalignment='right')  # 创建文本标签对象

        # TextBox's text object should not parse mathtext at all.
        self.text_disp = self.ax.text(
            self._text_position, 0.5, initial, transform=self.ax.transAxes,
            verticalalignment='center', horizontalalignment=textalignment,
            parse_math=False)  # 创建用于显示文本内容的对象，并设置水平对齐和是否解析数学表达式

        self._observers = cbook.CallbackRegistry(signals=["change", "submit"])  # 创建回调函数注册器

        ax.set(
            xlim=(0, 1), ylim=(0, 1),  # 设置坐标轴范围
            navigate=False, facecolor=color,  # 设置导航、背景颜色
            xticks=[], yticks=[])  # 隐藏坐标轴刻度

        self.cursor_index = 0  # 初始化光标位置索引

        self.cursor = ax.vlines(0, 0, 0, visible=False, color="k", lw=1,
                                transform=mpl.transforms.IdentityTransform())  # 创建光标对象

        self.connect_event('button_press_event', self._click)  # 连接鼠标按下事件
        self.connect_event('button_release_event', self._release)  # 连接鼠标释放事件
        self.connect_event('motion_notify_event', self._motion)  # 连接鼠标移动事件
        self.connect_event('key_press_event', self._keypress)  # 连接键盘按下事件
        self.connect_event('resize_event', self._resize)  # 连接窗口大小调整事件

        self.color = color  # 设置文本框的颜色
        self.hovercolor = hovercolor  # 设置鼠标悬停时文本框的颜色

        self.capturekeystrokes = False  # 设置是否捕获键盘输入
    def text(self):
        return self.text_disp.get_text()

    def _rendercursor(self):
        # this is a hack to figure out where the cursor should go.
        # 我们通过这种方式来确定光标应该放在哪里。
        # we draw the text up to where the cursor should go, measure
        # 我们绘制文本直到光标应该停留的位置，测量
        # and save its dimensions, draw the real text, then put the cursor
        # 并保存其尺寸，绘制真实的文本，然后将光标放在保存的尺寸处
        # at the saved dimensions
        # 在保存的尺寸处放置光标

        # This causes a single extra draw if the figure has never been rendered
        # yet, which should be fine as we're going to repeatedly re-render the
        # figure later anyways.
        # 如果图形尚未渲染，这会导致额外的一次绘制，但这应该没问题，
        # 因为我们接下来会反复重新渲染这个图形。
        if self.ax.figure._get_renderer() is None:
            self.ax.figure.canvas.draw()

        text = self.text_disp.get_text()  # Save value before overwriting it.
        # Save the current text content before modifying it.

        widthtext = text[:self.cursor_index]
        # Get the text content up to the cursor index.

        bb_text = self.text_disp.get_window_extent()
        # Get the bounding box (extent) of the text display.

        self.text_disp.set_text(widthtext or ",")
        # Set the text display to the truncated text or a comma if empty.

        bb_widthtext = self.text_disp.get_window_extent()
        # Get the bounding box of the truncated text.

        if bb_text.y0 == bb_text.y1:  # Restoring the height if no text.
            bb_text.y0 -= bb_widthtext.height / 2
            bb_text.y1 += bb_widthtext.height / 2
        elif not widthtext:  # Keep width to 0.
            bb_text.x1 = bb_text.x0
        else:  # Move the cursor using width of bb_widthtext.
            bb_text.x1 = bb_text.x0 + bb_widthtext.width

        self.cursor.set(
            segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]],
            visible=True)
        # Set the position and visibility of the cursor.

        self.text_disp.set_text(text)
        # Restore the original text content.

        self.ax.figure.canvas.draw()
        # Redraw the figure canvas.

    def _release(self, event):
        if self.ignore(event):
            return
        # If the event should be ignored, return without further action.

        if event.canvas.mouse_grabber != self.ax:
            return
        # If the mouse grabber is not the current axis, return.

        event.canvas.release_mouse(self.ax)
        # Release the mouse grabber from the current axis.
    # 处理键盘按键事件，响应用户输入
    def _keypress(self, event):
        # 如果事件需要被忽略，则直接返回
        if self.ignore(event):
            return
        # 如果捕获键盘输入开启
        if self.capturekeystrokes:
            key = event.key  # 获取按下的键
            text = self.text  # 获取当前文本内容
            # 如果按下的是单个字符
            if len(key) == 1:
                # 在光标位置插入该字符到文本中
                text = (text[:self.cursor_index] + key +
                        text[self.cursor_index:])
                self.cursor_index += 1  # 更新光标位置
            elif key == "right":
                # 如果按下右箭头键，将光标向右移动一位
                if self.cursor_index != len(text):
                    self.cursor_index += 1
            elif key == "left":
                # 如果按下左箭头键，将光标向左移动一位
                if self.cursor_index != 0:
                    self.cursor_index -= 1
            elif key == "home":
                # 如果按下Home键，将光标移到文本起始位置
                self.cursor_index = 0
            elif key == "end":
                # 如果按下End键，将光标移到文本末尾位置
                self.cursor_index = len(text)
            elif key == "backspace":
                # 如果按下退格键，删除光标前的一个字符
                if self.cursor_index != 0:
                    text = (text[:self.cursor_index - 1] +
                            text[self.cursor_index:])
                    self.cursor_index -= 1
            elif key == "delete":
                # 如果按下删除键，删除光标后的一个字符
                if self.cursor_index != len(self.text):
                    text = (text[:self.cursor_index] +
                            text[self.cursor_index + 1:])
            # 更新显示文本
            self.text_disp.set_text(text)
            self._rendercursor()  # 重新渲染光标
            # 如果事件响应开启，则处理文本变化事件
            if self.eventson:
                self._observers.process('change', self.text)
                # 如果按下回车键，则处理提交事件
                if key in ["enter", "return"]:
                    self._observers.process('submit', self.text)

    # 设置文本框的值
    def set_val(self, val):
        newval = str(val)  # 将输入值转换为字符串
        # 如果新值与当前文本相同，则直接返回
        if self.text == newval:
            return
        # 更新显示文本为新值
        self.text_disp.set_text(newval)
        self._rendercursor()  # 重新渲染光标
        # 如果事件响应开启，则处理文本变化事件和提交事件
        if self.eventson:
            self._observers.process('change', self.text)
            self._observers.process('submit', self.text)

    # 开始允许用户输入文本
    def begin_typing(self):
        self.capturekeystrokes = True  # 开启捕获键盘输入
        # 创建ExitStack实例来注册在用户停止输入时执行的清理操作
        stack = ExitStack()
        self._on_stop_typing = stack.close  # 设置停止输入时执行的动作
        toolmanager = getattr(
            self.ax.figure.canvas.manager, "toolmanager", None)
        if toolmanager is not None:
            # 如果使用toolmanager，锁定键盘输入，并在停止输入时释放锁定
            toolmanager.keypresslock(self)
            stack.callback(toolmanager.keypresslock.release, self)
        else:
            # 如果不使用toolmanager，禁用所有与按键相关的rcParams设置，以避免不必要的警告
            with _api.suppress_matplotlib_deprecation_warning():
                stack.enter_context(mpl.rc_context(
                    {k: [] for k in mpl.rcParams if k.startswith("keymap.")}))
    # 如果允许捕获按键输入
    if self.capturekeystrokes:
        # 调用停止输入的回调函数
        self._on_stop_typing()
        # 清空停止输入的回调函数引用
        self._on_stop_typing = None
        # 提示可以提交操作
        notifysubmit = True
    else:
        # 不允许捕获按键输入时，不提示提交操作
        notifysubmit = False
    
    # 停止捕获按键输入
    self.capturekeystrokes = False
    # 隐藏光标
    self.cursor.set_visible(False)
    # 刷新绘图区域
    self.ax.figure.canvas.draw()
    
    # 如果允许事件处理，并且已经提示可以提交操作
    if notifysubmit and self.eventson:
        # 由于 process() 可能会在用户代码中抛出错误，因此只有在清理完成后才调用它
        self._observers.process('submit', self.text)

def _click(self, event):
    # 如果事件需要被忽略，则直接返回
    if self.ignore(event):
        return
    # 如果事件发生在绘图区域之外，则停止输入
    if not self.ax.contains(event)[0]:
        self.stop_typing()
        return
    # 如果事件处理被禁用，则直接返回
    if not self.eventson:
        return
    # 如果鼠标事件发生在其他地方，将鼠标事件捕获到当前绘图区域
    if event.canvas.mouse_grabber != self.ax:
        event.canvas.grab_mouse(self.ax)
    # 如果尚未捕获按键输入，则开始捕获
    if not self.capturekeystrokes:
        self.begin_typing()
    # 计算光标位置
    self.cursor_index = self.text_disp._char_index_at(event.x)
    # 重新渲染光标
    self._rendercursor()

def _resize(self, event):
    # 调用停止输入方法
    self.stop_typing()

def _motion(self, event):
    # 如果事件需要被忽略，则直接返回
    if self.ignore(event):
        return
    # 如果鼠标悬停在绘图区域上，则设置颜色为悬停颜色；否则使用默认颜色
    c = self.hovercolor if self.ax.contains(event)[0] else self.color
    # 如果设置的颜色与当前绘图区域的背景色不同，则更新绘图区域的背景色
    if not colors.same_color(c, self.ax.get_facecolor()):
        self.ax.set_facecolor(c)
        # 如果绘图区域需要绘制，则刷新绘图区域
        if self.drawon:
            self.ax.figure.canvas.draw()

def on_text_change(self, func):
    """
    当文本内容变化时，调用指定的函数 *func*。

    返回连接 ID，可以用于断开连接。
    """
    return self._observers.connect('change', lambda text: func(text))

def on_submit(self, func):
    """
    当用户按下回车或离开提交框时，调用指定的函数 *func*。

    返回连接 ID，可以用于断开连接。
    """
    return self._observers.connect('submit', lambda text: func(text))

def disconnect(self, cid):
    """移除具有连接 ID *cid* 的观察者。"""
    self._observers.disconnect(cid)
    """
    A GUI neutral radio button.

    For the buttons to remain responsive you must keep a reference to this
    object.

    Connect to the RadioButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    activecolor : :mpltype:`color`
        The color of the selected button.
    labels : list of `.Text`
        The button labels.
    value_selected : str
        The label text of the currently selected button.
    index_selected : int
        The index of the selected button.
    """

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        # 如果事件被忽略或者画布已更改，则返回
        if self.ignore(event) or self._changed_canvas():
            return
        # 复制画布上当前轴范围的内容，并存储为背景
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        # 重新绘制按钮
        self.ax.draw_artist(self._buttons)

    def _clicked(self, event):
        # 如果事件被忽略、不是左键点击或者事件不在轴内，则返回
        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        # 找到包含事件的按钮和文本的索引
        idxs = [
            *self._buttons.contains(event)[1]["ind"],
            *[i for i, text in enumerate(self.labels) if text.contains(event)[0]]
        ]
        # 如果找到了索引
        if idxs:
            # 获取按钮偏移的坐标变换
            coords = self._buttons.get_offset_transform().transform(
                self._buttons.get_offsets())
            # 设置活动状态为距离事件位置最近的按钮索引
            self.set_active(
                idxs[(((event.x, event.y) - coords[idxs]) ** 2).sum(-1).argmin()]
            )

    def set_label_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels.
        """
        # 检查props参数是否为字典类型
        _api.check_isinstance(dict, props=props)
        # 扩展文本属性
        props = _expand_text_props(props)
        # 更新每个标签的属性
        for text, prop in zip(self.labels, props):
            text.update(prop)

    def set_radio_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the radio
            buttons.
        """
        # 检查props参数是否为字典类型
        _api.check_isinstance(dict, props=props)
        # 如果props中包含's'键，则保持与构造函数一致的API
        if 's' in props:
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        # 更新按钮的属性
        self._buttons.update(props)
        # 获取活动颜色
        self._active_colors = self._buttons.get_facecolor()
        # 如果只有一个活动颜色，则重复以匹配标签数目
        if len(self._active_colors) == 1:
            self._active_colors = np.repeat(self._active_colors,
                                            len(self.labels), axis=0)
        # 设置按钮的面颜色
        self._buttons.set_facecolor(
            [activecolor if text.get_text() == self.value_selected else "none"
             for text, activecolor in zip(self.labels, self._active_colors)])

    @property
    # 返回当前对象的_activecolor属性值
    def activecolor(self):
        return self._activecolor

    # 设置对象的_activecolor属性，并验证颜色合法性
    @activecolor.setter
    def activecolor(self, activecolor):
        # 使用colors._check_color_like函数验证activecolor的合法性
        colors._check_color_like(activecolor=activecolor)
        # 将_activecolor属性设置为传入的activecolor值
        self._activecolor = activecolor
        # 调用set_radio_props方法设置单选按钮的属性，更新其颜色
        self.set_radio_props({'facecolor': activecolor})

    # 选择指定索引位置的按钮
    def set_active(self, index):
        """
        Select button with number *index*.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            The index of the button to activate.

        Raises
        ------
        ValueError
            If the index is invalid.
        """
        # 检查索引是否在有效范围内，如果不是则引发ValueError异常
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid RadioButton index: {index}')
        # 设置选中的值为labels中对应索引的文本内容
        self.value_selected = self.labels[index].get_text()
        # 记录选中的索引值
        self.index_selected = index
        # 获取当前按钮的颜色，并更新选中按钮的颜色
        button_facecolors = self._buttons.get_facecolor()
        button_facecolors[:] = colors.to_rgba("none")
        button_facecolors[index] = colors.to_rgba(self._active_colors[index])
        self._buttons.set_facecolor(button_facecolors)

        # 如果需要绘制，则执行相应的绘制操作
        if self.drawon:
            # 如果使用blit技术，且有背景存在，则恢复背景
            if self._useblit:
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                # 绘制更新后的按钮
                self.ax.draw_artist(self._buttons)
                # 刷新canvas上的指定区域
                self.canvas.blit(self.ax.bbox)
            else:
                # 普通绘制操作
                self.canvas.draw()

        # 如果开启事件监听，则调用观察者处理点击事件，并传递选中按钮的文本内容
        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    # 将活动按钮重置为初始活动按钮
    def clear(self):
        """Reset the active button to the initially active one."""
        # 调用set_active方法，将活动按钮重置为初始活动按钮
        self.set_active(self._initial_active)

    # 将回调函数与按钮点击事件连接起来
    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
        # 将回调函数func连接到'clicked'事件，并返回连接ID
        return self._observers.connect('clicked', func)

    # 断开指定连接ID的观察者
    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        # 断开指定连接ID的观察者
        self._observers.disconnect(cid)
class SubplotTool(Widget):
    """
    A tool to adjust the subplot params of a `.Figure`.
    """

    def __init__(self, targetfig, toolfig):
        """
        Parameters
        ----------
        targetfig : `~matplotlib.figure.Figure`
            要调整的图形实例。
        toolfig : `~matplotlib.figure.Figure`
            要嵌入子图工具的图形实例。
        """

        self.figure = toolfig
        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)
        toolfig.suptitle("Click on slider to adjust subplot param")

        self._sliders = []
        names = ["left", "bottom", "right", "top", "wspace", "hspace"]
        # 最后一个子图在下面被移除，为“重置”按钮留出空间。
        for name, ax in zip(names, toolfig.subplots(len(names) + 1)):
            ax.set_navigate(False)
            slider = Slider(ax, name, 0, 1,
                            valinit=getattr(targetfig.subplotpars, name))
            slider.on_changed(self._on_slider_changed)
            self._sliders.append(slider)
        toolfig.axes[-1].remove()
        (self.sliderleft, self.sliderbottom, self.sliderright, self.slidertop,
         self.sliderwspace, self.sliderhspace) = self._sliders
        for slider in [self.sliderleft, self.sliderbottom,
                       self.sliderwspace, self.sliderhspace]:
            slider.closedmax = False
        for slider in [self.sliderright, self.slidertop]:
            slider.closedmin = False

        # 约束条件
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')
        self.buttonreset.on_clicked(self._on_reset)

    def _on_slider_changed(self, _):
        self.targetfig.subplots_adjust(
            **{slider.label.get_text(): slider.val
               for slider in self._sliders})
        if self.drawon:
            self.targetfig.canvas.draw()

    def _on_reset(self, event):
        with ExitStack() as stack:
            # 临时禁用自身和自身滑块的绘制，并断开滑块事件（因为子图参数在恢复的顺序可能导致临时无效）。
            stack.enter_context(cbook._setattr_cm(self, drawon=False))
            for slider in self._sliders:
                stack.enter_context(
                    cbook._setattr_cm(slider, drawon=False, eventson=False))
            # 将滑块重置到初始位置。
            for slider in self._sliders:
                slider.reset()
        if self.drawon:
            event.canvas.draw()  # 重新绘制子图工具画布。
        self._on_slider_changed(None)  # 将更改应用到目标窗口。
    """
    A crosshair cursor that spans the Axes and moves with mouse cursor.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting` for details.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    Examples
    --------
    See :doc:`/gallery/widgets/cursor`.
    """
    # 定义一个类，实现跨越 Axes 的十字光标，并随鼠标移动而移动

    def __init__(self, ax, *, horizOn=True, vertOn=True, useblit=False,
                 **lineprops):
        # 调用父类构造函数初始化
        super().__init__(ax)

        # 连接鼠标移动事件和重绘事件到对应的处理方法
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        # 初始化属性
        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit

        # 根据是否使用 blitting 设置属性
        if self.useblit:
            lineprops['animated'] = True
        
        # 创建水平和垂直线条对象，并初始化为不可见状态
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)

        # 初始化背景和清除标志
        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        # 如果事件被忽略或者画布发生变化，则退出
        if self.ignore(event) or self._changed_canvas():
            return
        
        # 如果使用 blitting，则保存画布的背景
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        # 如果事件被忽略，则退出
        if self.ignore(event):
            return
        
        # 如果当前不可用或者鼠标不在 Axes 区域内，则隐藏十字光标并返回
        if not self.canvas.widgetlock.available(self) or not self.ax.contains(event)[0]:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)
            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        
        # 标记需要清除，并获取鼠标事件的数据坐标
        self.needclear = True
        xdata, ydata = self._get_data_coords(event)
        
        # 设置垂直和水平线条的位置和可见性
        self.linev.set_xdata((xdata, xdata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_ydata((ydata, ydata))
        self.lineh.set_visible(self.visible and self.horizOn)
        
        # 如果垂直或水平线条不可见，则返回
        if not (self.visible and (self.vertOn or self.horizOn)):
            return
        
        # 重新绘制
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
"""
提供一个垂直（默认）和/或水平线游标，用于多个 Axes（坐标轴）。

为了保持游标的响应性，必须保持对它的引用。

Parameters
----------
canvas : object
    此参数完全未使用，仅为向后兼容而保留。

axes : list of `~matplotlib.axes.Axes`
    要附加游标的 `~.axes.Axes` 对象列表。

useblit : bool, 默认: True
    如果后端支持，则使用 blitting 进行更快的绘制。
    详见教程 :ref:`blitting`。

horizOn : bool, 默认: False
    是否绘制水平线。

vertOn : bool, 默认: True
    是否绘制垂直线。

Other Parameters
----------------
**lineprops
    控制线条外观的 `.Line2D` 属性。
    参见 `~.Axes.axhline`。

Examples
--------
参见 :doc:`/gallery/widgets/multicursor`。
"""

class MultiCursor(Widget):
    def __init__(self, canvas, axes, *, useblit=True, horizOn=False, vertOn=True,
                 **lineprops):
        """
        初始化函数，创建 MultiCursor 对象。

        Parameters
        ----------
        canvas : object
            此参数仅为了兼容而保留，将来不再需要存储。

        axes : list of `~matplotlib.axes.Axes`
            要附加游标的坐标轴对象列表。

        useblit : bool, 默认: True
            如果支持，使用 blitting 进行更快的绘制。

        horizOn : bool, 默认: False
            是否绘制水平线。

        vertOn : bool, 默认: True
            是否绘制垂直线。

        **lineprops
            控制线条外观的 `.Line2D` 属性。

        Notes
        -----
        初始化时设置了游标的初始位置和属性，以及事件连接。

        Raises
        ------
        None
        """
        self._canvas = canvas  # 仅为了提供已弃用的 .canvas 属性而保留。

        self.axes = axes  # 存储传入的坐标轴列表
        self.horizOn = horizOn  # 是否绘制水平线
        self.vertOn = vertOn  # 是否绘制垂直线

        # 存储每个坐标轴对应的画布信息，包括事件连接和背景信息
        self._canvas_infos = {
            ax.figure.canvas: {"cids": [], "background": None} for ax in axes}

        # 获取最后一个坐标轴的 x 和 y 范围，并计算中点
        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[-1].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.visible = True  # 游标是否可见
        self.useblit = (
            useblit
            and all(canvas.supports_blit for canvas in self._canvas_infos)
        )  # 是否使用 blitting 进行绘制

        if self.useblit:
            lineprops['animated'] = True

        # 初始化垂直线列表
        self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
                       for ax in axes]
        # 初始化水平线列表
        self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
                       for ax in axes]

        self.connect()  # 连接事件处理函数

    def connect(self):
        """
        连接事件处理函数。

        Notes
        -----
        将 'motion_notify_event' 和 'draw_event' 事件连接到相应的处理函数上。

        Raises
        ------
        None
        """
        for canvas, info in self._canvas_infos.items():
            info["cids"] = [
                canvas.mpl_connect('motion_notify_event', self.onmove),
                canvas.mpl_connect('draw_event', self.clear),
            ]

    def disconnect(self):
        """
        断开事件连接。

        Notes
        -----
        断开之前连接的所有事件处理函数。

        Raises
        ------
        None
        """
        for canvas, info in self._canvas_infos.items():
            for cid in info["cids"]:
                canvas.mpl_disconnect(cid)
            info["cids"].clear()
    def clear(self, event):
        """Clear the cursor."""
        # 如果事件被忽略，则返回
        if self.ignore(event):
            return
        # 如果使用blit技术
        if self.useblit:
            # 遍历所有画布及其信息
            for canvas, info in self._canvas_infos.items():
                # 检查画布是否已经被更改
                # 这种情况会发生在`savefig`需要保存到之前的后端不支持的格式时
                if canvas is not canvas.figure.canvas:
                    continue
                # 将画布特定区域的背景保存在信息中
                info["background"] = canvas.copy_from_bbox(canvas.figure.bbox)

    def onmove(self, event):
        # 根据事件获取包含该事件的坐标轴列表
        axs = [ax for ax in self.axes if ax.contains(event)[0]]
        # 如果事件被忽略或没有找到坐标轴或当前的事件锁不可用，则返回
        if self.ignore(event) or not axs or not event.canvas.widgetlock.available(self):
            return
        # 从列表中获取顶层的艺术家对象
        ax = cbook._topmost_artist(axs)
        # 根据事件是否在坐标轴内来获取x和y数据
        xdata, ydata = ((event.xdata, event.ydata) if event.inaxes is ax
                        else ax.transData.inverted().transform((event.x, event.y)))
        # 更新垂直线的位置和可见性
        for line in self.vlines:
            line.set_xdata((xdata, xdata))
            line.set_visible(self.visible and self.vertOn)
        # 更新水平线的位置和可见性
        for line in self.hlines:
            line.set_ydata((ydata, ydata))
            line.set_visible(self.visible and self.horizOn)
        # 如果垂直或水平线不可见，则返回
        if not (self.visible and (self.vertOn or self.horizOn)):
            return
        # 重新绘制
        if self.useblit:
            # 恢复之前保存的背景
            for canvas, info in self._canvas_infos.items():
                if info["background"]:
                    canvas.restore_region(info["background"])
            # 如果需要绘制垂直线，则在每个坐标轴上绘制
            if self.vertOn:
                for ax, line in zip(self.axes, self.vlines):
                    ax.draw_artist(line)
            # 如果需要绘制水平线，则在每个坐标轴上绘制
            if self.horizOn:
                for ax, line in zip(self.axes, self.hlines):
                    ax.draw_artist(line)
            # 使用blit技术将所有画布更新
            for canvas in self._canvas_infos:
                canvas.blit()
        else:
            # 如果不使用blit技术，则在每个画布上进行空闲绘制
            for canvas in self._canvas_infos:
                canvas.draw_idle()
class _SelectorWidget(AxesWidget):
    # 继承 AxesWidget 类来创建 _SelectorWidget 类

    def __init__(self, ax, onselect, useblit=False, button=None,
                 state_modifier_keys=None, use_data_coordinates=False):
        # 初始化函数，接受参数：
        # ax: 绘图的坐标轴
        # onselect: 选择事件的回调函数
        # useblit: 是否使用 blitting 技术，默认为 False
        # button: 触发选择的鼠标按钮，可以是整数或列表
        # state_modifier_keys: 状态修改键的映射字典
        # use_data_coordinates: 是否使用数据坐标系进行操作，默认为 False
        super().__init__(ax)

        # 控制部件是否可见
        self._visible = True
        # 设置选择事件的回调函数
        self.onselect = onselect
        # 是否启用 blitting 技术，并且检查画布是否支持
        self.useblit = useblit and self.canvas.supports_blit
        # 连接默认的事件处理函数
        self.connect_default_events()

        # 定义状态修改键的默认映射，可以修改这些键的默认值
        self._state_modifier_keys = dict(move=' ', clear='escape',
                                         square='shift', center='control',
                                         rotate='r')
        self._state_modifier_keys.update(state_modifier_keys or {})
        # 是否使用数据坐标系进行操作
        self._use_data_coordinates = use_data_coordinates

        # 背景对象初始化为空
        self.background = None

        # 如果 button 是整数，则将其设置为有效按钮列表，否则直接使用传入的按钮列表
        if isinstance(button, Integral):
            self.validButtons = [button]
        else:
            self.validButtons = button

        # 标记当前是否完成了选择动作，初始为 False
        self._selection_completed = False

        # 将保存鼠标按下时的事件数据（位置）
        self._eventpress = None
        # 将保存鼠标释放时的事件数据（位置）
        self._eventrelease = None
        # 上一个事件对象
        self._prev_event = None
        # 当前状态的集合，用于记录当前活动的状态
        self._state = set()

    def set_active(self, active):
        # 设置部件的活动状态，并调用基类的同名方法
        super().set_active(active)
        # 如果部件被激活，则更新背景
        if active:
            self.update_background(None)

    def _get_animated_artists(self):
        """
        Convenience method to get all animated artists of the figure containing
        this widget, excluding those already present in self.artists.
        The returned tuple is not sorted by 'z_order': z_order sorting is
        valid only when considering all artists and not only a subset of all
        artists.
        """
        # 获取包含此部件的图形中所有动画艺术家对象，不包括已经存在于 self.artists 中的对象
        # 返回的元组未按 'z_order' 排序：仅在考虑所有艺术家而不是所有的子集时 'z_order' 排序是有效的
        return tuple(a for ax_ in self.ax.get_figure().get_axes()
                     for a in ax_.get_children()
                     if a.get_animated() and a not in self.artists)
    def update_background(self, event):
        """Force an update of the background."""
        # 如果不使用双缓冲技术，直接返回，不更新背景
        if not self.useblit:
            return
        # 通过重新渲染背景确保小部件艺术家不会被意外包含在背景中
        # 如果需要，重新渲染背景（然后重新渲染包含小部件艺术家的画布）
        # 需要移除所有将在更新选择器时绘制的艺术家：如果图中有动画艺术家，出于安全考虑，默认重新绘制
        # 在重新绘制时需要遵守zorder（绘制顺序）
        artists = sorted(self.artists + self._get_animated_artists(),
                         key=lambda a: a.get_zorder())
        # 检查是否有任何可见的艺术家需要重新绘制
        needs_redraw = any(artist.get_visible() for artist in artists)
        with ExitStack() as stack:
            if needs_redraw:
                # 设置所有艺术家不可见
                for artist in artists:
                    stack.enter_context(artist._cm_set(visible=False))
                # 绘制画布
                self.canvas.draw()
            # 复制指定区域的背景到缓冲区
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        if needs_redraw:
            # 重新绘制所有需要的艺术家到画布
            for artist in artists:
                self.ax.draw_artist(artist)

    def connect_default_events(self):
        """Connect the major canvas events to methods."""
        # 连接主要的画布事件到对应的方法
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_press_event', self.press)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.update_background)
        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('key_release_event', self.on_key_release)
        self.connect_event('scroll_event', self.on_scroll)

    def ignore(self, event):
        # docstring inherited
        # 如果不处于活动状态或者Axes不可见，则忽略事件
        if not self.active or not self.ax.get_visible():
            return True
        # 如果画布被锁定，则忽略事件
        if not self.canvas.widgetlock.available(self):
            return True
        # 如果事件没有按钮属性，则设置为None
        if not hasattr(event, 'button'):
            event.button = None
        # 只有当事件由所需按钮触发时才执行矩形选择
        if (self.validButtons is not None
                and event.button not in self.validButtons):
            return True
        # 如果没有按钮被按下，且事件发生在Axes外部，则忽略事件
        if self._eventpress is None:
            return not self.ax.contains(event)[0]
        # 如果有按钮被按下，检查释放按钮是否相同
        if event.button == self._eventpress.button:
            return False
        # 如果有按钮被按下，检查释放按钮是否相同
        return (not self.ax.contains(event)[0] or
                event.button != self._eventpress.button)
    def update(self):
        """
        根据 self.useblit 的值使用 blit() 或 draw_idle() 进行绘制。

        如果图形不可见或者图形的渲染器为 None，则直接返回。
        如果使用 blit，如果存在背景，则恢复背景，否则更新背景。
        需要绘制所有艺术家，包括 self._get_animated_artists()，并且按照 z_order 排序。
        最后使用 blit 将画布上的内容绘制到 Axes 的 bbox 区域。
        否则，使用 draw_idle() 将画布重新绘制。
        """
        if (not self.ax.get_visible() or
                self.ax.figure._get_renderer() is None):
            return
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            else:
                self.update_background(None)
            artists = sorted(self.artists + self._get_animated_artists(),
                             key=lambda a: a.get_zorder())
            for artist in artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _get_data(self, event):
        """
        获取事件的 xdata 和 ydata，并根据 Axes 的边界进行裁剪。

        如果 event 的 xdata 为 None，则返回 None, None。
        否则，获取事件的数据坐标，并根据 Axes 的 x 和 y 轴边界进行裁剪。
        返回裁剪后的 xdata 和 ydata。
        """
        if event.xdata is None:
            return None, None
        xdata, ydata = self._get_data_coords(event)
        xdata = np.clip(xdata, *self.ax.get_xbound())
        ydata = np.clip(ydata, *self.ax.get_ybound())
        return xdata, ydata

    def _clean_event(self, event):
        """
        预处理事件：

        - 如果 event 的 xdata 为 None，则使用之前的事件 _prev_event。
        - 获取事件的 xdata 和 ydata，并裁剪到 Axes 的边界。
        - 更新 _prev_event 为当前事件。
        """
        if event.xdata is None:
            event = self._prev_event
        else:
            event = copy.copy(event)
        event.xdata, event.ydata = self._get_data(event)
        self._prev_event = event
        return event

    def press(self, event):
        """
        按钮按下事件处理器和验证器。

        如果不忽略事件，则清理事件并设置 _eventpress 和 _prev_event。
        如果按下的键是移动状态的修改键，则添加 'move' 到状态集合 _state 中。
        调用 _press 处理按钮按下事件。
        返回 True 表示事件被处理，否则返回 False。
        """
        if not self.ignore(event):
            event = self._clean_event(event)
            self._eventpress = event
            self._prev_event = event
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            if key == self._state_modifier_keys['move']:
                self._state.add('move')
            self._press(event)
            return True
        return False

    def _press(self, event):
        """
        按钮按下事件处理器。
        """
        pass

    def release(self, event):
        """
        按钮释放事件处理器和验证器。

        如果不忽略事件且 _eventpress 存在，则清理事件并设置 _eventrelease。
        调用 _release 处理按钮释放事件。
        清空 _eventpress 和 _eventrelease，并从状态集合 _state 中移除 'move'。
        返回 True 表示事件被处理，否则返回 False。
        """
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            self._eventrelease = event
            self._release(event)
            self._eventpress = None
            self._eventrelease = None
            self._state.discard('move')
            return True
        return False

    def _release(self, event):
        """
        按钮释放事件处理器。
        """
        pass
    def onmove(self, event):
        """Cursor move event handler and validator."""
        # 检查事件是否需要忽略并且鼠标左键是否按下
        if not self.ignore(event) and self._eventpress:
            # 清理并更新事件，然后调用实际的移动处理方法
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler."""
        # 实际的光标移动处理方法，具体操作由子类实现

    def on_scroll(self, event):
        """Mouse scroll event handler and validator."""
        # 检查事件是否需要忽略，然后调用滚轮滚动处理方法
        if not self.ignore(event):
            self._on_scroll(event)

    def _on_scroll(self, event):
        """Mouse scroll event handler."""
        # 实际的滚轮滚动处理方法，具体操作由子类实现

    def on_key_press(self, event):
        """Key press event handler and validator for all selection widgets."""
        # 检查是否处于激活状态
        if self.active:
            # 获取按下的键并处理特定的按键操作
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            # 处理特定的状态修饰键
            if key == self._state_modifier_keys['clear']:
                # 清除选择
                self.clear()
                return
            for (state, modifier) in self._state_modifier_keys.items():
                if modifier in key.split('+'):
                    # 处理状态修饰键的添加或删除
                    if state == 'rotate':
                        if state in self._state:
                            self._state.discard(state)
                        else:
                            self._state.add(state)
                    else:
                        self._state.add(state)
            # 调用键盘按下处理方法
            self._on_key_press(event)

    def _on_key_press(self, event):
        """Key press event handler - for widget-specific key press actions."""
        # 实际的键盘按下处理方法，由子类根据需要实现具体操作

    def on_key_release(self, event):
        """Key release event handler and validator."""
        # 检查是否处于激活状态
        if self.active:
            key = event.key or ''
            for (state, modifier) in self._state_modifier_keys.items():
                # 处理状态修饰键的删除
                if modifier in key.split('+') and state != 'rotate':
                    self._state.discard(state)
            # 调用键盘释放处理方法
            self._on_key_release(event)

    def _on_key_release(self, event):
        """Key release event handler."""
        # 实际的键盘释放处理方法，由子类根据需要实现具体操作

    def set_visible(self, visible):
        """Set the visibility of the selector artists."""
        # 设置选择器图形的可见性
        self._visible = visible
        for artist in self.artists:
            artist.set_visible(visible)

    def get_visible(self):
        """Get the visibility of the selector artists."""
        # 获取选择器图形的可见性
        return self._visible

    @property
    def visible(self):
        _api.warn_deprecated("3.8", alternative="get_visible")
        return self.get_visible()

    def clear(self):
        """Clear the selection and set the selector ready to make a new one."""
        # 清除选择并准备选择器进行新的选择操作
        self._clear_without_update()
        self.update()

    def _clear_without_update(self):
        # 清除选择状态，但不更新显示
        self._selection_completed = False
        self.set_visible(False)
    # 返回包含选择器艺术家的元组
    def artists(self):
        """Tuple of the artists of the selector."""
        # 获取已处理艺术家的属性或空元组
        handles_artists = getattr(self, '_handles_artists', ())
        # 返回选择器艺术家和已处理艺术家的元组
        return (self._selection_artist,) + handles_artists

    # 设置选择器艺术家的属性
    def set_props(self, **props):
        """
        Set the properties of the selector artist.

        See the *props* argument in the selector docstring to know which properties are
        supported.
        """
        # 获取选择器艺术家对象
        artist = self._selection_artist
        # 规范化属性参数
        props = cbook.normalize_kwargs(props, artist)
        # 应用属性到艺术家对象
        artist.set(**props)
        # 如果使用 blit 技术，则更新显示
        if self.useblit:
            self.update()

    # 设置处理器艺术家的属性
    def set_handle_props(self, **handle_props):
        """
        Set the properties of the handles selector artist. See the
        `handle_props` argument in the selector docstring to know which
        properties are supported.
        """
        # 如果选择器没有处理器艺术家，则抛出未实现错误
        if not hasattr(self, '_handles_artists'):
            raise NotImplementedError("This selector doesn't have handles.")

        # 获取第一个处理器艺术家对象
        artist = self._handles_artists[0]
        # 规范化处理器属性参数
        handle_props = cbook.normalize_kwargs(handle_props, artist)
        # 对每个处理器艺术家应用属性
        for handle in self._handles_artists:
            handle.set(**handle_props)
        # 如果使用 blit 技术，则更新显示
        if self.useblit:
            self.update()
        # 更新处理器属性字典
        self._handle_props.update(handle_props)

    # 验证状态的有效性
    def _validate_state(self, state):
        # 支持的状态列表，排除 'clear' 和 'not-applicable' 状态
        supported_state = [
            key for key, value in self._state_modifier_keys.items()
            if key != 'clear' and value != 'not-applicable'
            ]
        # 检查状态是否在支持列表中
        _api.check_in_list(supported_state, state=state)

    # 添加状态以定义小部件的行为
    def add_state(self, state):
        """
        Add a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """
        # 验证状态的有效性
        self._validate_state(state)
        # 将状态添加到状态集合
        self._state.add(state)

    # 移除状态以定义小部件的行为
    def remove_state(self, state):
        """
        Remove a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """
        # 验证状态的有效性
        self._validate_state(state)
        # 从状态集合中移除状态
        self._state.remove(state)
class SpanSelector(_SelectorWidget):
    """
    Visually select a min/max range on a single axis and call a function with
    those values.

    To guarantee that the selector remains responsive, keep a reference to it.

    In order to turn off the SpanSelector, set ``span_selector.active`` to
    False.  To turn it back on, set it to True.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes instance to which the SpanSelector is attached.

    onselect : callable with signature ``func(min: float, max: float)``
        A callback function that is called after a release event and the
        selection is created, changed or removed.

    direction : {"horizontal", "vertical"}
        The direction along which to draw the span selector.

    minspan : float, default: 0
        If selection is less than or equal to *minspan*, the selection is
        removed (when already existing) or cancelled.

    useblit : bool, default: False
        If True, use the backend-dependent blitting features for faster
        canvas updates. See the tutorial :ref:`blitting` for details.

    props : dict, default: {'facecolor': 'red', 'alpha': 0.5}
        Dictionary of `.Patch` properties defining the appearance of the
        span selector.

    onmove_callback : callable with signature ``func(min: float, max: float)``, optional
        Called on mouse move while the span is being selected.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    button : `.MouseButton` or list of `.MouseButton`, default: all buttons
        The mouse buttons which activate the span selector.

    handle_props : dict, default: None
        Properties of the handle lines at the edges of the span. Only used
        when *interactive* is True. See `.Line2D` for valid properties.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be activated.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget's behavior. Values
        amend the defaults, which are:

        - "clear": Clear the current shape, default: "escape".

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be ignored.

    snap_values : 1D array-like, optional
        Snap the selector edges to the given values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(vmin, vmax):
    ...     print(vmin, vmax)
    >>> span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
    """
    # 创建一个用于在 matplotlib 图形中选择范围的交互式小部件
    def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
                 props={'facecolor': 'red', 'alpha': 0.5}, onmove_callback=None,
                 interactive=False, button=None, handle_props=None,
                 grab_range=10, state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, snap_values=None):
        # 调用父类的初始化方法，传入Axes实例和相关属性
        super().__init__(ax, onselect, useblit=useblit,
                         button=button, minspan=minspan,
                         props=props, onmove_callback=onmove_callback,
                         direction=direction, state_modifier_keys=state_modifier_keys,
                         interactive=interactive, handle_props=handle_props,
                         drag_from_anywhere=drag_from_anywhere,
                         ignore_event_outside=ignore_event_outside,
                         snap_values=snap_values, grab_range=grab_range)
    ...                              props=dict(facecolor='blue', alpha=0.5))
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selector`
    """
    
    # 类的文档字符串，通常包含一些相关的示例和参考信息
    # 此处示例显示如何使用SpanSelector小部件，提供了一个相关的文档链接

    def __init__(self, ax, onselect, direction, *, minspan=0, useblit=False,
                 props=None, onmove_callback=None, interactive=False,
                 button=None, handle_props=None, grab_range=10,
                 state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, snap_values=None):
        """
        初始化方法，设置SpanSelector对象的各种属性和参数。

        Parameters:
        - ax: 绘制SpanSelector的Axes对象
        - onselect: 当选择完成时调用的回调函数
        - direction: 选择的方向，'horizontal'或'vertical'
        - minspan: 最小选择距离
        - useblit: 是否使用blitting来优化绘图性能
        - props: 用于绘制选择区域的属性字典，默认为红色半透明
        - onmove_callback: 拖动选择区域时调用的回调函数
        - interactive: 是否启用交互模式
        - button: 触发选择操作的鼠标按钮
        - handle_props: 手柄的属性字典
        - grab_range: 拖动手柄的触发范围
        - state_modifier_keys: 状态修改键的映射，用于清除、方形、居中、旋转等操作
        - drag_from_anywhere: 是否允许从任意位置开始拖动选择区域
        - ignore_event_outside: 是否忽略区域外的事件
        - snap_values: 选择区域的捕捉数值

        Notes:
        - 设置默认状态修改键的映射，如果未提供则使用默认值
        - 根据useblit设置动画属性
        """

        if state_modifier_keys is None:
            state_modifier_keys = dict(clear='escape',
                                       square='not-applicable',
                                       center='not-applicable',
                                       rotate='not-applicable')
        super().__init__(ax, onselect, useblit=useblit, button=button,
                         state_modifier_keys=state_modifier_keys)

        if props is None:
            props = dict(facecolor='red', alpha=0.5)

        props['animated'] = self.useblit

        # 设置SpanSelector的方向和其他属性
        self.direction = direction
        self._extents_on_press = None
        self.snap_values = snap_values

        self.onmove_callback = onmove_callback
        self.minspan = minspan

        self.grab_range = grab_range
        self._interactive = interactive
        self._edge_handles = None
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside

        # 重置画布，以便`new_axes`连接事件
        self.canvas = None
        self.new_axes(ax, _props=props)

        # 设置手柄的属性字典
        self._handle_props = {
            'color': props.get('facecolor', 'r'),
            **cbook.normalize_kwargs(handle_props, Line2D)}

        # 如果启用了交互模式，设置边缘手柄
        if self._interactive:
            self._edge_order = ['min', 'max']
            self._setup_edge_handles(self._handle_props)

        self._active_handle = None

    def new_axes(self, ax, *, _props=None):
        """
        将SpanSelector设置为操作新的Axes。

        Parameters:
        - ax: 新的Axes对象
        - _props: 可选的属性字典，用于更新选择区域的绘制属性

        Resets the SpanSelector to operate on a new Axes.
        """
        self.ax = ax
        if self.canvas is not ax.figure.canvas:
            if self.canvas is not None:
                self.disconnect_events()

            self.canvas = ax.figure.canvas
            self.connect_default_events()

        # 重置选择完成的状态
        self._selection_completed = False

        # 根据选择方向设置变换和矩形尺寸
        if self.direction == 'horizontal':
            trans = ax.get_xaxis_transform()
            w, h = 0, 1
        else:
            trans = ax.get_yaxis_transform()
            w, h = 1, 0
        rect_artist = Rectangle((0, 0), w, h, transform=trans, visible=False)
        if _props is not None:
            rect_artist.update(_props)
        elif self._selection_artist is not None:
            rect_artist.update_from(self._selection_artist)

        # 将矩形添加到Axes对象
        self.ax.add_patch(rect_artist)
        self._selection_artist = rect_artist
    def _setup_edge_handles(self, props):
        # 根据方向设置初始位置，使用轴的边界来保持相同的边界
        if self.direction == 'horizontal':
            # 如果选择器方向为水平，则使用 x 轴的边界作为初始位置
            positions = self.ax.get_xbound()
        else:
            # 否则使用 y 轴的边界作为初始位置
            positions = self.ax.get_ybound()
        # 使用 ToolLineHandles 类初始化边缘处理器
        self._edge_handles = ToolLineHandles(self.ax, positions,
                                             direction=self.direction,
                                             line_props=props,
                                             useblit=self.useblit)

    @property
    def _handles_artists(self):
        # 返回边缘处理器中的艺术家对象列表
        if self._edge_handles is not None:
            return self._edge_handles.artists
        else:
            return ()

    def _set_cursor(self, enabled):
        """根据选择器的方向更新画布上的光标样式。"""
        if enabled:
            cursor = (backend_tools.Cursors.RESIZE_HORIZONTAL
                      if self.direction == 'horizontal' else
                      backend_tools.Cursors.RESIZE_VERTICAL)
        else:
            cursor = backend_tools.Cursors.POINTER

        # 设置画布上的光标样式
        self.ax.figure.canvas.set_cursor(cursor)

    def connect_default_events(self):
        # 继承自父类的文档字符串
        super().connect_default_events()
        if getattr(self, '_interactive', False):
            # 如果设置为交互模式，则连接 'motion_notify_event' 事件到 self._hover 方法
            self.connect_event('motion_notify_event', self._hover)

    def _press(self, event):
        """按钮按下事件处理程序。"""
        # 设置光标为相应的调整方向
        self._set_cursor(True)
        if self._interactive and self._selection_artist.get_visible():
            # 如果处于交互模式且选择器可见，则设置活动处理器
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if self._active_handle is None or not self._interactive:
            # 在绘制新的矩形前清除之前的矩形。
            self.update()

        xdata, ydata = self._get_data_coords(event)
        v = xdata if self.direction == 'horizontal' else ydata

        if self._active_handle is None and not self.ignore_event_outside:
            # 当按下事件在跨度外部时，初始设置可见性为 False 并设置范围为 (v, v)
            # 在设置范围时将调用更新
            self._visible = False
            self._set_extents((v, v))
            # 需要重新设置可见性为 True，以便在必要时绘制跨度选择器（跨度宽度 > 0）
            self._visible = True
        else:
            self.set_visible(True)

        return False

    @property
    def direction(self):
        """跨度选择器的方向：'vertical' 或 'horizontal'。"""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the direction of the span selector."""
        # 检查direction参数是否在允许的列表中
        _api.check_in_list(['horizontal', 'vertical'], direction=direction)
        # 如果已经定义了_direction属性，并且新的direction与旧的不同
        if hasattr(self, '_direction') and direction != self._direction:
            # 移除先前的绘图元素
            self._selection_artist.remove()
            # 如果是交互模式，移除边缘处理
            if self._interactive:
                self._edge_handles.remove()
            # 更新_direction属性为新的direction
            self._direction = direction
            # 调用new_axes方法更新图形对象
            self.new_axes(self.ax)
            # 如果是交互模式，重新设置边缘处理
            if self._interactive:
                self._setup_edge_handles(self._handle_props)
        else:
            # 否则直接更新_direction属性为新的direction
            self._direction = direction

    def _release(self, event):
        """Button release event handler."""
        # 设置鼠标光标为非活动状态
        self._set_cursor(False)

        # 如果不是交互模式，隐藏选择绘图元素
        if not self._interactive:
            self._selection_artist.set_visible(False)

        # 如果没有活动的处理对象，并且选择已完成且忽略事件在选择框外
        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        # 获取当前选择的最小和最大值
        vmin, vmax = self.extents
        # 计算选择的跨度
        span = vmax - vmin

        # 如果跨度小于等于最小跨度
        if span <= self.minspan:
            # 隐藏选择框，并且设置选择未完成
            self.set_visible(False)
            if self._selection_completed:
                # 当选择已存在时调用onselect方法
                self.onselect(vmin, vmax)
            self._selection_completed = False
        else:
            # 否则调用onselect方法，并设置选择已完成
            self.onselect(vmin, vmax)
            self._selection_completed = True

        # 更新绘图
        self.update()

        # 重置活动处理对象为None
        self._active_handle = None

        return False

    def _hover(self, event):
        """Update the canvas cursor if it's over a handle."""
        # 如果事件应该被忽略则返回
        if self.ignore(event):
            return

        # 如果有活动处理对象或者选择未完成，则不执行任何操作
        if self._active_handle is not None or not self._selection_completed:
            return

        # 获取最接近事件坐标的边缘处理对象和距离
        _, e_dist = self._edge_handles.closest(event.x, event.y)
        # 根据距离设置鼠标光标
        self._set_cursor(e_dist <= self.grab_range)
    def _onmove(self, event):
        """Motion notify event handler."""

        # 获取数据坐标
        xdata, ydata = self._get_data_coords(event)

        # 根据方向选择要操作的坐标
        if self.direction == 'horizontal':
            v = xdata
            vpress = self._eventpress.xdata
        else:
            v = ydata
            vpress = self._eventpress.ydata

        # 移动现有的跨度
        if self._active_handle == 'C' and self._extents_on_press is not None:
            # 获取起始和结束值
            vmin, vmax = self._extents_on_press
            # 计算移动的距离
            dv = v - vpress
            # 更新跨度的位置
            vmin += dv
            vmax += dv

        # 调整现有形状的大小
        elif self._active_handle and self._active_handle != 'C':
            # 获取起始和结束值
            vmin, vmax = self._extents_on_press
            # 根据活动句柄调整起始或结束值
            if self._active_handle == 'min':
                vmin = v
            else:
                vmax = v

        # 创建新的形状
        else:
            # 如果 ignore_event_outside 为 True 并且已完成选择，则不创建新跨度
            if self.ignore_event_outside and self._selection_completed:
                return
            # 设置新形状的起始和结束值
            vmin, vmax = vpress, v
            # 确保 vmin 比 vmax 小
            if vmin > vmax:
                vmin, vmax = vmax, vmin

        # 设置更新后的跨度
        self._set_extents((vmin, vmax))

        # 如果有移动回调函数，则调用
        if self.onmove_callback is not None:
            self.onmove_callback(vmin, vmax)

        return False

    def _draw_shape(self, vmin, vmax):
        """根据给定的 vmin 和 vmax 绘制形状。"""

        # 确保 vmin 小于 vmax
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # 根据水平或垂直方向绘制形状
        if self.direction == 'horizontal':
            self._selection_artist.set_x(vmin)
            self._selection_artist.set_width(vmax - vmin)
        else:
            self._selection_artist.set_y(vmin)
            self._selection_artist.set_height(vmax - vmin)

    def _set_active_handle(self, event):
        """根据鼠标事件的位置设置活动句柄。"""
        
        # 注意：event.xdata/ydata 是数据坐标，event.x/y 是像素坐标
        # 找到最接近的边缘句柄及其距离
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)

        # 优先选择中心句柄
        if 'move' in self._state:
            self._active_handle = 'C'
        elif e_dist > self.grab_range:
            # 不靠近任何句柄
            self._active_handle = None
            # 如果允许从任意位置拖动并且点击在区域内，则设置中心句柄
            if self.drag_from_anywhere and self._contains(event):
                self._active_handle = 'C'
                self._extents_on_press = self.extents
            else:
                self._active_handle = None
                return
        else:
            # 最接近的是边缘句柄
            self._active_handle = self._edge_order[e_idx]

        # 记录句柄移动开始时的矩形坐标
        self._extents_on_press = self.extents
    def _contains(self, event):
        """Return True if event is within the patch."""
        # 判断事件是否在图形区域内部
        return self._selection_artist.contains(event, radius=0)[0]

    @staticmethod
    def _snap(values, snap_values):
        """Snap values to a given array values (snap_values)."""
        # 考虑机器精度，计算最小间隔
        eps = np.min(np.abs(np.diff(snap_values))) * 1e-12
        # 对给定的数值 values 进行与 snap_values 最接近的吸附
        return tuple(
            snap_values[np.abs(snap_values - v + np.sign(v) * eps).argmin()]
            for v in values)

    @property
    def extents(self):
        """
        (float, float)
            The values, in data coordinates, for the start and end points of the current
            selection. If there is no selection then the start and end values will be
            the same.
        """
        # 获取当前选择的起始和结束点的数据坐标范围
        if self.direction == 'horizontal':
            vmin = self._selection_artist.get_x()
            vmax = vmin + self._selection_artist.get_width()
        else:
            vmin = self._selection_artist.get_y()
            vmax = vmin + self._selection_artist.get_height()
        return vmin, vmax

    @extents.setter
    def extents(self, extents):
        # 设置当前选择的起始和结束点的数据坐标范围
        self._set_extents(extents)
        self._selection_completed = True

    def _set_extents(self, extents):
        # 更新显示的形状
        if self.snap_values is not None:
            extents = tuple(self._snap(extents, self.snap_values))
        self._draw_shape(*extents)
        if self._interactive:
            # 更新显示的手柄
            self._edge_handles.set_data(self.extents)
        self.set_visible(self._visible)
        self.update()
class ToolLineHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    positions : 1D array
        Positions of handles in data coordinates.
    direction : {"horizontal", "vertical"}
        Direction of handles, either 'vertical' or 'horizontal'
    line_props : dict, optional
        Additional line properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    def __init__(self, ax, positions, direction, *, line_props=None,
                 useblit=True):
        # 初始化函数，设置属性和创建线条对象

        self.ax = ax

        # 检查 direction 是否在指定的列表中
        _api.check_in_list(['horizontal', 'vertical'], direction=direction)
        self._direction = direction

        # 设置线条的属性，包括可见性和是否动画
        line_props = {
            **(line_props if line_props is not None else {}),
            'visible': False,
            'animated': useblit,
        }

        # 根据 direction 选择创建水平或垂直线条的方法
        line_fun = ax.axvline if self._direction == 'horizontal' else ax.axhline

        # 创建与 positions 相关的线条对象列表
        self._artists = [line_fun(p, **line_props) for p in positions]

    @property
    def artists(self):
        # 返回所有线条对象的元组
        return tuple(self._artists)

    @property
    def positions(self):
        """Positions of the handle in data coordinates."""
        # 根据 direction 返回所有线条的位置信息
        method = 'get_xdata' if self._direction == 'horizontal' else 'get_ydata'
        return [getattr(line, method)()[0] for line in self.artists]

    @property
    def direction(self):
        """Direction of the handle: 'vertical' or 'horizontal'."""
        # 返回线条的方向信息
        return self._direction

    def set_data(self, positions):
        """
        Set x- or y-positions of handles, depending on if the lines are
        vertical or horizontal.

        Parameters
        ----------
        positions : tuple of length 2
            Set the positions of the handle in data coordinates
        """
        # 根据 direction 设置线条的数据位置
        method = 'set_xdata' if self._direction == 'horizontal' else 'set_ydata'
        for line, p in zip(self.artists, positions):
            getattr(line, method)([p, p])

    def set_visible(self, value):
        """Set the visibility state of the handles artist."""
        # 设置所有线条对象的可见性
        for artist in self.artists:
            artist.set_visible(value)

    def set_animated(self, value):
        """Set the animated state of the handles artist."""
        # 设置所有线条对象的动画状态
        for artist in self.artists:
            artist.set_animated(value)

    def remove(self):
        """Remove the handles artist from the figure."""
        # 从图形中移除所有线条对象
        for artist in self._artists:
            artist.remove()
    # 定义一个方法来找到最接近给定坐标(x, y)的控制点，并返回其索引和距离。

    if self.direction == 'horizontal':
        # 如果控制器方向为水平方向：
        # 将所有控制点的水平坐标转换为画布坐标系下的像素位置
        p_pts = np.array([
            self.ax.transData.transform((p, 0))[0] for p in self.positions
            ])
        # 计算每个控制点到给定 x 坐标的距离
        dist = abs(p_pts - x)
    else:
        # 如果控制器方向为垂直方向：
        # 将所有控制点的垂直坐标转换为画布坐标系下的像素位置
        p_pts = np.array([
            self.ax.transData.transform((0, p))[1] for p in self.positions
            ])
        # 计算每个控制点到给定 y 坐标的距离
        dist = abs(p_pts - y)

    # 找到距离数组中最小值的索引，即最接近给定坐标的控制点索引
    index = np.argmin(dist)

    # 返回最接近点的索引和距离
    return index, dist[index]
class ToolHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str, default: 'o'
        Shape of marker used to display handle. See `~.pyplot.plot`.
    marker_props : dict, optional
        Additional marker properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    def __init__(self, ax, x, y, *, marker='o', marker_props=None, useblit=True):
        # 初始化函数，设置控制柄的显示属性和位置
        self.ax = ax
        # 定义控制柄的绘制属性
        props = {'marker': marker, 'markersize': 7, 'markerfacecolor': 'w',
                 'linestyle': 'none', 'alpha': 0.5, 'visible': False,
                 'label': '_nolegend_',
                 **cbook.normalize_kwargs(marker_props, Line2D._alias_map)}
        # 创建 Line2D 对象来表示控制柄，并添加到 Axes 中
        self._markers = Line2D(x, y, animated=useblit, **props)
        self.ax.add_line(self._markers)

    @property
    def x(self):
        # 返回控制柄的 x 坐标数据
        return self._markers.get_xdata()

    @property
    def y(self):
        # 返回控制柄的 y 坐标数据
        return self._markers.get_ydata()

    @property
    def artists(self):
        # 返回用于显示控制柄的艺术家对象
        return (self._markers, )

    def set_data(self, pts, y=None):
        """Set x and y positions of handles."""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        # 设置控制柄的位置数据
        self._markers.set_data(pts)

    def set_visible(self, val):
        # 设置控制柄的可见性
        self._markers.set_visible(val)

    def set_animated(self, val):
        # 设置控制柄的动画属性
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        # 获取所有控制柄的坐标点
        pts = np.column_stack([self.x, self.y])
        # 将数据坐标转换为像素坐标
        pts = self.ax.transData.transform(pts)
        # 计算每个控制柄与指定点 (x, y) 的像素距离
        diff = pts - [x, y]
        dist = np.hypot(*diff.T)
        # 找到距离最近的控制柄的索引和距离
        min_index = np.argmin(dist)
        return min_index, dist[min_index]


_RECTANGLESELECTOR_PARAMETERS_DOCSTRING = \
    r"""
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function
        A callback function that is called after a release event and the
        selection is created, changed or removed.
        It must have the signature::

            def onselect(eclick: MouseEvent, erelease: MouseEvent)

        where *eclick* and *erelease* are the mouse click and release
        `.MouseEvent`\s that start and complete the selection.

    minspanx : float, default: 0
        Selections with an x-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.

    minspany : float, default: 0
        Selections with an y-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.
    """
    # 是否使用 blitting 加快绘图速度（如果后端支持）。详见 blitting 教程
    useblit : bool, default: False

    # 绘制 __ARTIST_NAME__ 的属性。查看 `.Patch` 获取有效属性。
    # 默认属性: `dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)`
    props : dict, optional

    # *minspanx* 和 *minspany* 的解释方式，可以是 "data" 或 "pixels" 坐标
    spancoords : {"data", "pixels"}, default: "data"

    # 触发矩形选择的按钮。`.MouseButton` 或 `.MouseButton` 列表，默认为所有按钮
    button : `.MouseButton`, list of `.MouseButton`, default: all buttons

    # 激活交互工具手柄的像素距离
    grab_range : float, default: 10

    # 交互式手柄（标记艺术家）的绘制属性。查看 `.Line2D` 的标记参数获取有效属性。
    # 默认值在 `mpl.rcParams` 中定义，除了 `markeredgecolor`，其默认值与 `props` 的 `edgecolor` 属性相同。
    handle_props : dict, optional

    # 绘制部件后是否允许与小部件进行交互的一组手柄
    interactive : bool, default: False

    # 修改小部件行为的键盘修改键。值修改默认值，包括：
    # - "move": 移动现有形状，默认为无修改键。
    # - "clear": 清除当前形状，默认为 "escape"。
    # - "square": 使形状成为正方形，默认为 "shift"。
    # - "center": 在其中心周围更改形状，默认为 "ctrl"。
    # - "rotate": 将形状绕中心旋转 -45° 到 45° 之间，默认为 "r"。
    # 可以组合 "square" 和 "center"。正方形形状可以在创建选择器时指定的 `use_data_coordinates` 参数确定的数据或显示坐标中定义。
    state_modifier_keys : dict, optional

    # 如果为 `True`，则可以通过在其边界内的任意位置单击来移动小部件。
    drag_from_anywhere : bool, default: False

    # 如果为 `True`，则在跨度选择器外触发的事件将被忽略。
    ignore_event_outside : bool, default: False

    # 如果为 `True`，则选择器的 "square" 形状将在数据坐标而不是显示坐标中定义。
    use_data_coordinates : bool, default: False
# 用类装饰器将 _RECTANGLESELECTOR_PARAMETERS_DOCSTRING 替换为 rectangle 的参数文档字符串，并应用于 RectangleSelector 类
@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'rectangle'))
class RectangleSelector(_SelectorWidget):
    """
    Select a rectangular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(eclick, erelease):
    ...     print(eclick.xdata, eclick.ydata)
    ...     print(erelease.xdata, erelease.ydata)
    >>> props = dict(facecolor='blue', alpha=0.5)
    >>> rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
    ...                                   props=props)
    >>> fig.show()
    >>> rect.add_state('square')

    See also: :doc:`/gallery/widgets/rectangle_selector`
    """
    # 初始化方法，接受多个参数并设置可选的默认值
    def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
                 props=None, spancoords='data', button=None, grab_range=10,
                 handle_props=None, interactive=False,
                 state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, use_data_coordinates=False):
        # 调用父类的初始化方法，传递一些参数
        super().__init__(ax, onselect, useblit=useblit, button=button,
                         state_modifier_keys=state_modifier_keys,
                         use_data_coordinates=use_data_coordinates)

        # 设置对象是否为交互式
        self._interactive = interactive
        # 是否允许从任意位置拖动
        self.drag_from_anywhere = drag_from_anywhere
        # 是否忽略界外事件
        self.ignore_event_outside = ignore_event_outside
        # 初始化旋转角度
        self._rotation = 0.0
        # 初始化纵横比修正值
        self._aspect_ratio_correction = 1.0

        # 允许创建交互式选择器的状态标记，用于在PolygonSelector中作为交互式边界框使用，方便调整多边形大小
        self._allow_creation = True

        # 如果未提供props参数，则设置默认属性
        if props is None:
            props = dict(facecolor='red', edgecolor='black',
                         alpha=0.2, fill=True)
        # 更新props字典，添加动画属性，如果self.useblit为真
        props = {**props, 'animated': self.useblit}
        # 设置对象的可见性，从props中取出visible属性，默认为self._visible
        self._visible = props.pop('visible', self._visible)
        # 初始化图形形状并添加到轴上
        to_draw = self._init_shape(**props)
        self.ax.add_patch(to_draw)

        # 设置选择艺术家为刚创建的形状
        self._selection_artist = to_draw
        # 设置纵横比修正值
        self._set_aspect_ratio_correction()

        # 设置最小横向和纵向跨度
        self.minspanx = minspanx
        self.minspany = minspany

        # 检查spancoords参数是否在指定列表中
        _api.check_in_list(['data', 'pixels'], spancoords=spancoords)
        # 设置跨度坐标系属性
        self.spancoords = spancoords

        # 设置抓取范围
        self.grab_range = grab_range

        # 如果对象是交互式的，设置相关属性和处理器
        if self._interactive:
            # 设置处理器的属性，如边缘颜色等
            self._handle_props = {
                'markeredgecolor': (props or {}).get('edgecolor', 'black'),
                **cbook.normalize_kwargs(handle_props, Line2D)}

            # 设置角落处理器的顺序和位置
            self._corner_order = ['SW', 'SE', 'NE', 'NW']
            xc, yc = self.corners
            self._corner_handles = ToolHandles(self.ax, xc, yc,
                                               marker_props=self._handle_props,
                                               useblit=self.useblit)

            # 设置边缘处理器的顺序和位置
            self._edge_order = ['W', 'S', 'E', 'N']
            xe, ye = self.edge_centers
            self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                             marker_props=self._handle_props,
                                             useblit=self.useblit)

            # 设置中心处理器的位置
            xc, yc = self.center
            self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s',
                                              marker_props=self._handle_props,
                                              useblit=self.useblit)

            # 初始化活动处理器为空
            self._active_handle = None

        # 初始化按下时的范围
        self._extents_on_press = None

    # 返回所有处理器的艺术家列表的属性
    @property
    def _handles_artists(self):
        return (*self._center_handle.artists, *self._corner_handles.artists,
                *self._edge_handles.artists)
    # 在对象初始化时创建一个矩形，并返回该矩形对象
    def _init_shape(self, **props):
        return Rectangle((0, 0), 0, 1, visible=False,
                         rotation_point='center', **props)

    # 处理按钮按下事件的方法
    def _press(self, event):
        """Button press event handler."""
        
        # 如果是交互模式且选择对象可见，则设置活动操作点
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        # 如果没有活动操作点或者不是交互模式，并且允许创建新的矩形
        if ((self._active_handle is None or not self._interactive) and
                self._allow_creation):
            # 清除之前绘制的矩形，准备绘制新的矩形
            self.update()

        # 如果没有活动操作点并且不忽略外部事件，并且允许创建新矩形
        if (self._active_handle is None and not self.ignore_event_outside and
                self._allow_creation):
            # 获取事件中的数据坐标
            x, y = self._get_data_coords(event)
            # 隐藏当前图形
            self._visible = False
            # 设置矩形范围为单个点的范围
            self.extents = x, x, y, y
            # 显示图形
            self._visible = True
        else:
            # 设置图形可见性为真
            self.set_visible(True)

        # 保存按下时的矩形范围
        self._extents_on_press = self.extents
        # 保存按下时的旋转角度
        self._rotation_on_press = self._rotation
        # 设置纵横比校正
        self._set_aspect_ratio_correction()

        # 返回假，表示事件处理完成
        return False
    # Button release event handler.
    def _release(self, event):
        # 如果不是交互模式，隐藏选择艺术家对象
        if not self._interactive:
            self._selection_artist.set_visible(False)

        # 如果没有活动句柄且选择已完成，并且忽略外部事件，则返回
        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        # 更新 eventpress 和 eventrelease 对象的数据范围
        x0, x1, y0, y1 = self.extents
        self._eventpress.xdata = x0
        self._eventpress.ydata = y0
        # 将数据坐标转换为绘图坐标
        xy0 = self.ax.transData.transform([x0, y0])
        self._eventpress.x, self._eventpress.y = xy0

        self._eventrelease.xdata = x1
        self._eventrelease.ydata = y1
        xy1 = self.ax.transData.transform([x1, y1])
        self._eventrelease.x, self._eventrelease.y = xy1

        # 计算选择框或线的尺寸
        if self.spancoords == 'data':
            spanx = abs(self._eventpress.xdata - self._eventrelease.xdata)
            spany = abs(self._eventpress.ydata - self._eventrelease.ydata)
        elif self.spancoords == 'pixels':
            spanx = abs(self._eventpress.x - self._eventrelease.x)
            spany = abs(self._eventpress.y - self._eventrelease.y)
        else:
            _api.check_in_list(['data', 'pixels'],
                               spancoords=self.spancoords)

        # 检查绘制的距离是否在 x 或 y 方向上不太小
        if spanx <= self.minspanx or spany <= self.minspany:
            if self._selection_completed:
                # 当选择已经存在时调用 onselect 函数
                self.onselect(self._eventpress, self._eventrelease)
            # 清除选择但不更新
            self._clear_without_update()
        else:
            self.onselect(self._eventpress, self._eventrelease)
            self._selection_completed = True

        # 更新绘图
        self.update()
        self._active_handle = None
        self._extents_on_press = None

        return False

    @property
    def _rect_bbox(self):
        # 返回选择艺术家对象的边界框
        return self._selection_artist.get_bbox().bounds

    def _set_aspect_ratio_correction(self):
        # 获取当前绘图区域的纵横比
        aspect_ratio = self.ax._get_aspect_ratio()
        self._selection_artist._aspect_ratio_correction = aspect_ratio
        # 如果使用数据坐标系，则纵横比校正为1，否则为当前绘图区域的纵横比
        if self._use_data_coordinates:
            self._aspect_ratio_correction = 1
        else:
            self._aspect_ratio_correction = aspect_ratio

    def _get_rotation_transform(self):
        # 获取当前绘图区域的纵横比
        aspect_ratio = self.ax._get_aspect_ratio()
        # 构建旋转变换矩阵
        return Affine2D().translate(-self.center[0], -self.center[1]) \
                .scale(1, aspect_ratio) \
                .rotate(self._rotation) \
                .scale(1, 1 / aspect_ratio) \
                .translate(*self.center)
    def corners(self):
        """
        Corners of rectangle in data coordinates from lower left,
        moving clockwise.
        """
        # 获取矩形的边界框坐标及宽高
        x0, y0, width, height = self._rect_bbox
        # 计算矩形四个角点的 x 和 y 坐标
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        # 获取旋转变换矩阵
        transform = self._get_rotation_transform()
        # 应用旋转变换，计算出旋转后的角点坐标
        coords = transform.transform(np.array([xc, yc]).T).T
        return coords[0], coords[1]

    @property
    def edge_centers(self):
        """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
        # 获取矩形的边界框坐标及宽高
        x0, y0, width, height = self._rect_bbox
        # 计算每条边中点的 x 和 y 坐标
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        # 获取旋转变换矩阵
        transform = self._get_rotation_transform()
        # 应用旋转变换，计算出旋转后的边中点坐标
        coords = transform.transform(np.array([xe, ye]).T).T
        return coords[0], coords[1]

    @property
    def center(self):
        """Center of rectangle in data coordinates."""
        # 获取矩形的边界框坐标及宽高，计算出矩形中心点的坐标
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """
        Return (xmin, xmax, ymin, ymax) in data coordinates as defined by the
        bounding box before rotation.
        """
        # 获取矩形的边界框坐标及宽高
        x0, y0, width, height = self._rect_bbox
        # 计算矩形未旋转前的边界范围（xmin, xmax, ymin, ymax）
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # 更新显示的形状
        self._draw_shape(extents)
        if self._interactive:
            # 更新显示的角点
            self._corner_handles.set_data(*self.corners)
            # 更新显示的边中点
            self._edge_handles.set_data(*self.edge_centers)
            # 更新显示的中心点
            x, y = self.center
            self._center_handle.set_data([x], [y])
        # 设置是否可见
        self.set_visible(self._visible)
        # 更新显示
        self.update()

    @property
    def rotation(self):
        """
        Rotation in degree in interval [-45°, 45°]. The rotation is limited in
        range to keep the implementation simple.
        """
        # 返回矩形的旋转角度（度数）
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        # 将旋转角度限制在 [-45°, 45°] 的范围内，以避免改变句柄的顺序
        if -45 <= value and value <= 45:
            self._rotation = np.deg2rad(value)
            # 调用 extents 的 setter 方法来绘制形状并更新句柄位置
            self.extents = self.extents
    def _draw_shape(self, extents):
        # 解构 extents 元组，获取 x0, x1, y0, y1 四个变量
        x0, x1, y0, y1 = extents
        # 将 x0, x1, y0, y1 进行排序并分别赋值给 xmin, xmax, ymin, ymax
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        # 获取当前坐标轴的 x 和 y 范围，并进行排序
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        # 调整 xmin, ymin, xmax, ymax 使其不超出当前坐标轴范围
        xmin = max(xlim[0], xmin)
        ymin = max(ylim[0], ymin)
        xmax = min(xmax, xlim[1])
        ymax = min(ymax, ylim[1])

        # 设置选择框的位置和大小，以及角度（根据 self.rotation）
        self._selection_artist.set_x(xmin)
        self._selection_artist.set_y(ymin)
        self._selection_artist.set_width(xmax - xmin)
        self._selection_artist.set_height(ymax - ymin)
        self._selection_artist.set_angle(self.rotation)

    def _set_active_handle(self, event):
        """根据鼠标事件的位置设置活动的手柄。"""
        # 获取最接近鼠标事件位置的角落手柄的索引和距离
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        # 获取最接近鼠标事件位置的边缘手柄的索引和距离
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        # 获取最接近鼠标事件位置的中心手柄的索引和距离
        m_idx, m_dist = self._center_handle.closest(event.x, event.y)

        # 如果处于移动状态，则设置活动手柄为 'C'
        if 'move' in self._state:
            self._active_handle = 'C'
        # 如果鼠标点击足够接近中心手柄，优先选择中心手柄
        elif m_dist < self.grab_range * 2:
            self._active_handle = 'C'
        # 如果离角落和边缘手柄的距离都超过抓取范围，且允许从任意位置拖动且鼠标事件在选择区域内，则设置活动手柄为 'C'
        elif c_dist > self.grab_range and e_dist > self.grab_range:
            if self.drag_from_anywhere and self._contains(event):
                self._active_handle = 'C'
            else:
                self._active_handle = None
                return
        # 如果离角落手柄更近，则选择最接近的角落手柄作为活动手柄
        elif c_dist < e_dist:
            self._active_handle = self._corner_order[c_idx]
        # 否则选择最接近的边缘手柄作为活动手柄
        else:
            self._active_handle = self._edge_order[e_idx]

    def _contains(self, event):
        """如果事件在选择区域内，返回 True。"""
        return self._selection_artist.contains(event, radius=0)[0]

    @property
    def geometry(self):
        """
        返回一个形状为 (2, 5) 的数组，包含矩形的四个角的数据坐标 x 和 y。
        顶点从左上角开始，顺时针排列。
        """
        if hasattr(self._selection_artist, 'get_verts'):
            # 获取选择区域的顶点坐标并转换为数据坐标系
            xfm = self.ax.transData.inverted()
            y, x = xfm.transform(self._selection_artist.get_verts()).T
            return np.array([x, y])
        else:
            # 如果选择区域没有顶点坐标，直接返回选择区域的数据坐标
            return np.array(self._selection_artist.get_data())
@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'ellipse'))
class EllipseSelector(RectangleSelector):
    """
    Select an elliptical region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """

    # 初始化椭圆选择器的形状，返回一个椭圆对象
    def _init_shape(self, **props):
        return Ellipse((0, 0), 0, 1, visible=False, **props)

    # 绘制椭圆选择器的形状
    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        center = [x0 + (x1 - x0) / 2., y0 + (y1 - y0) / 2.]
        a = (xmax - xmin) / 2.
        b = (ymax - ymin) / 2.

        # 设置选择器的中心点、宽度、高度和角度
        self._selection_artist.center = center
        self._selection_artist.width = 2 * a
        self._selection_artist.height = 2 * b
        self._selection_artist.angle = self.rotation

    # 返回椭圆选择器形状的边界框
    @property
    def _rect_bbox(self):
        x, y = self._selection_artist.center
        width = self._selection_artist.width
        height = self._selection_artist.height
        return x - width / 2., y - height / 2., width, height


class LassoSelector(_SelectorWidget):
    """
    Selection curve of an arbitrary shape.

    For the selector to remain responsive you must keep a reference to it.

    The selected path can be used in conjunction with `~.Path.contains_point`
    to select data points from an image.

    In contrast to `Lasso`, `LassoSelector` is written with an interface
    similar to `RectangleSelector` and `SpanSelector`, and will continue to
    interact with the Axes until disconnected.

    Example usage::

        ax = plt.subplot()
        ax.plot(x, y)

        def onselect(verts):
            print(verts)
        lasso = LassoSelector(ax, onselect)

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    onselect : function
        Whenever the lasso is released, the *onselect* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props : dict, optional
        Properties with which the line is drawn, see `.Line2D`
        for valid properties. Default values are defined in ``mpl.rcParams``.
    button : `.MouseButton` or list of `.MouseButton`, optional
        The mouse buttons used for rectangle selection.  Default is ``None``,
        which corresponds to all buttons.
    """
    # 初始化方法，接受画布对象 ax、选择事件处理函数 onselect，并支持一些可选参数
    def __init__(self, ax, onselect, *, useblit=True, props=None, button=None):
        # 调用父类的初始化方法，传入画布对象 ax、选择事件处理函数 onselect，及额外的参数
        super().__init__(ax, onselect, useblit=useblit, button=button)
        # 初始化顶点列表为 None
        self.verts = None
        # 设置属性字典，合并传入的 props 参数（如果有的话），注意 self.useblit 可能与 useblit 不同，如果画布不支持 blitting。
        props = {
            **(props if props is not None else {}),
            # 注意：如果画布不支持 blitting，self.useblit 可能与 useblit 不同。
            'animated': self.useblit, 'visible': False,
        }
        # 创建 Line2D 对象，使用设置好的属性字典
        line = Line2D([], [], **props)
        # 将线对象添加到画布对象 ax 上
        self.ax.add_line(line)
        # 设置选择艺术家对象为刚创建的线对象
        self._selection_artist = line

    # 按下事件处理方法，接受事件对象 event
    def _press(self, event):
        # 将当前事件的数据点添加到顶点列表中
        self.verts = [self._get_data(event)]
        # 设置选择艺术家对象可见
        self._selection_artist.set_visible(True)

    # 松开事件处理方法，接受事件对象 event
    def _release(self, event):
        # 如果顶点列表不为 None
        if self.verts is not None:
            # 将当前事件的数据点添加到顶点列表中
            self.verts.append(self._get_data(event))
            # 调用选择事件处理函数 onselect，传入顶点列表 verts
            self.onselect(self.verts)
        # 清空选择艺术家对象的数据，使其不显示
        self._selection_artist.set_data([[], []])
        self._selection_artist.set_visible(False)
        # 将顶点列表重置为 None
        self.verts = None

    # 移动事件处理方法，接受事件对象 event
    def _onmove(self, event):
        # 如果顶点列表为 None，则返回
        if self.verts is None:
            return
        # 将当前事件的数据点添加到顶点列表中
        self.verts.append(self._get_data(event))
        # 更新选择艺术家对象的数据，使其显示出当前所有顶点的线段
        self._selection_artist.set_data(list(zip(*self.verts)))

        # 更新画布显示
        self.update()
# PolygonSelector 类，继承自 _SelectorWidget 类
class PolygonSelector(_SelectorWidget):
    """
    在 Axes 上选择多边形区域。

    每次鼠标点击都会放置顶点，通过完成多边形（点击第一个顶点）来选择区域。
    一旦绘制完成，可以通过左键点击和拖动来移动单个顶点，或者通过右键点击删除顶点。

    此外，可以使用以下修饰键：

    - 按住 *ctrl* 键并点击拖动顶点，在完成多边形之前重新定位它。
    - 按住 *shift* 键并在 Axes 中的任意位置点击并拖动，移动所有顶点。
    - 按 *esc* 键开始新的多边形。

    要保持选择器响应，请务必保持对它的引用。

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        小部件的父 Axes。

    onselect : function
        当完成多边形或修改完成后调用 *onselect* 函数，并传递顶点列表作为 ``(xdata, ydata)`` 元组。

    useblit : bool, default: False
        是否使用 blitting 进行更快的绘制（如果后端支持）。详细信息请参阅教程 :ref:`blitting`。

    props : dict, optional
        用于绘制线条的属性，参见 `.Line2D` 获取有效属性。
        默认值为::

            dict(color='k', linestyle='-', linewidth=2, alpha=0.5)

    handle_props : dict, optional
        用于绘制多边形顶点标记的 Artist 属性。查看 `.Line2D` 中关于标记参数的有效属性。
        默认值在 ``mpl.rcParams`` 中定义，除非 ``markeredgecolor`` 的默认值与 *props* 中的 ``color`` 属性相同。

    grab_range : float, default: 10
        如果鼠标点击在距离顶点 *grab_range* 像素内，则选择顶点（完成多边形或移动顶点）。

    draw_bounding_box : bool, optional
        如果为 `True`，则在完成多边形选择器后绘制一个边界框。此框可用于移动和调整选择器的大小。

    box_handle_props : dict, optional
        为框架处理设置的属性。更多信息请参阅 `RectangleSelector` 中 *handle_props* 参数的文档。

    box_props : dict, optional
        为框架设置的属性。更多信息请参阅 `RectangleSelector` 中 *props* 参数的文档。

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_simple`
    :doc:`/gallery/widgets/polygon_selector_demo`

    Notes
    -----
    如果删除顶点后只剩下一个点，则选择器将恢复到不完整状态，可以从现有点开始绘制新的多边形。
    """
    def __init__(self, ax, onselect, *, useblit=False,
                 props=None, handle_props=None, grab_range=10,
                 draw_bounding_box=False, box_handle_props=None,
                 box_props=None):
        # 创建状态修改键映射，这些键用于控制选择器的行为，但对于 PolygonSelector 没有全部支持
        # 注意：不能直接使用现有的 'move' 状态修改键来代替 'move_all'，因为 _SelectorWidget 在释放按钮时会自动丢弃 'move'
        state_modifier_keys = dict(clear='escape', move_vertex='control',
                                   move_all='shift', move='not-applicable',
                                   square='not-applicable',
                                   center='not-applicable',
                                   rotate='not-applicable')
        # 调用父类初始化方法，传入必要的参数和状态修改键映射
        super().__init__(ax, onselect, useblit=useblit,
                         state_modifier_keys=state_modifier_keys)

        # 初始化选中点的坐标列表
        self._xys = [(0, 0)]

        # 如果未提供 props 参数，则使用默认属性设置
        if props is None:
            props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        # 将动画属性 animated 设置为 useblit 的值，并更新 props 字典
        props = {**props, 'animated': self.useblit}
        # 创建 Line2D 对象作为选择对象的视觉表示，并添加到当前的 Axes
        self._selection_artist = line = Line2D([], [], **props)
        self.ax.add_line(line)

        # 如果未提供 handle_props 参数，则使用默认的处理点属性设置
        if handle_props is None:
            handle_props = dict(markeredgecolor='k',
                                markerfacecolor=props.get('color', 'k'))
        # 设置处理点的属性
        self._handle_props = handle_props
        # 创建 ToolHandles 对象来管理多边形的处理点
        self._polygon_handles = ToolHandles(self.ax, [], [],
                                            useblit=self.useblit,
                                            marker_props=self._handle_props)

        # 初始化活动的处理点索引为 -1（表示无活动点）
        self._active_handle_idx = -1
        # 设置默认的抓取范围
        self.grab_range = grab_range

        # 设置选择器的可见性为 True
        self.set_visible(True)
        # 设置是否绘制边界框
        self._draw_box = draw_bounding_box
        # 初始化边界框对象为 None
        self._box = None

        # 如果未提供 box_handle_props 参数，则更新为默认的处理点属性设置
        if box_handle_props is None:
            box_handle_props = {}
        # 更新边界框处理点属性，使用的是 _handle_props 的更新结果
        self._box_handle_props = self._handle_props.update(box_handle_props)
        # 设置边界框的属性
        self._box_props = box_props

    def _get_bbox(self):
        # 获取选择对象的边界框
        return self._selection_artist.get_bbox()

    def _add_box(self):
        # 创建 RectangleSelector 对象作为边界框
        self._box = RectangleSelector(self.ax,
                                      onselect=lambda *args, **kwargs: None,
                                      useblit=self.useblit,
                                      grab_range=self.grab_range,
                                      handle_props=self._box_handle_props,
                                      props=self._box_props,
                                      interactive=True)
        # 移除边界框的旋转功能
        self._box._state_modifier_keys.pop('rotate')
        # 连接事件处理程序来处理多边形的缩放
        self._box.connect_event('motion_notify_event', self._scale_polygon)
        # 更新边界框的显示
        self._update_box()
        # 设置状态以阻止用户创建 RectangleSelector 对象
        self._box._allow_creation = False
        # 设置选择已完成的状态
        self._box._selection_completed = True
        # 绘制多边形
        self._draw_polygon()
    # 如果存在选框对象，则隐藏该选框并置空
    def _remove_box(self):
        if self._box is not None:
            self._box.set_visible(False)
            self._box = None

    # 更新选框的边界，使其与多边形的边界一致
    def _update_box(self):
        if self._box is not None:
            bbox = self._get_bbox()
            self._box.extents = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
            # 保存一个副本
            self._old_box_extents = self._box.extents

    # 缩放多边形选择器的点，当边界框被移动或缩放时触发
    def _scale_polygon(self, event):
        """
        当边界框移动或缩放时，缩放多边形选择器的点。

        这个函数被设置为边界框 RectangleSelector 的回调函数。
        """
        if not self._selection_completed:
            return

        if self._old_box_extents == self._box.extents:
            return

        # 创建从旧边界框到新边界框的变换
        x1, y1, w1, h1 = self._box._rect_bbox
        old_bbox = self._get_bbox()
        t = (transforms.Affine2D()
             .translate(-old_bbox.x0, -old_bbox.y0)
             .scale(1 / old_bbox.width, 1 / old_bbox.height)
             .scale(w1, h1)
             .translate(x1, y1))

        # 更新多边形顶点。必须是元组列表以保持一致性。
        new_verts = [(x, y) for x, y in t.transform(np.array(self.verts))]
        self._xys = [*new_verts, new_verts[0]]
        self._draw_polygon()
        self._old_box_extents = self._box.extents

    @property
    # 返回多边形处理器的艺术家列表
    def _handles_artists(self):
        return self._polygon_handles.artists

    # 移除指定索引的顶点
    def _remove_vertex(self, i):
        """移除索引为 i 的顶点。"""
        if (len(self._xys) > 2 and
                self._selection_completed and
                i in (0, len(self._xys) - 1)):
            # 如果选择的是第一个或最后一个顶点，同时移除第一个和最后一个顶点（因为闭合多边形它们是相同的）
            self._xys.pop(0)
            self._xys.pop(-1)
            # 通过将新的第一个顶点追加到末尾，再次闭合多边形
            self._xys.append(self._xys[0])
        else:
            self._xys.pop(i)
        if len(self._xys) <= 2:
            # 如果只剩下一个点，回到未完成状态以便用户重新开始绘制
            self._selection_completed = False
            self._remove_box()

    # 按钮按下事件处理器
    def _press(self, event):
        """按钮按下事件处理函数。"""
        # 检查是否选择了工具手柄。
        if ((self._selection_completed or 'move_vertex' in self._state)
                and len(self._xys) > 0):
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.grab_range:
                self._active_handle_idx = h_idx
        # 在按下事件时保存顶点位置（支持 'move_all' 状态修改器）。
        self._xys_at_press = self._xys.copy()
    # 按钮释放事件处理函数
    def _release(self, event):
        """Button release event handler."""
        
        # 如果有活动的工具句柄被释放
        if self._active_handle_idx >= 0:
            # 如果事件按钮是右键（button == 3）
            if event.button == 3:
                # 移除指定索引处的顶点
                self._remove_vertex(self._active_handle_idx)
                # 重新绘制多边形
                self._draw_polygon()
            # 重置活动句柄索引为 -1
            self._active_handle_idx = -1

        # 完成多边形的闭合
        elif len(self._xys) > 3 and self._xys[-1] == self._xys[0]:
            # 标记选择已完成
            self._selection_completed = True
            # 如果需要绘制框并且框不存在，则添加框
            if self._draw_box and self._box is None:
                self._add_box()

        # 放置新顶点
        elif (not self._selection_completed
              and 'move_all' not in self._state
              and 'move_vertex' not in self._state):
            # 在倒数第二个位置插入新顶点的数据坐标
            self._xys.insert(-1, self._get_data_coords(event))

        # 如果选择已完成，则触发选择事件回调
        if self._selection_completed:
            self.onselect(self.verts)

    # 光标移动事件处理和验证函数
    def onmove(self, event):
        """Cursor move event handler and validator."""
        
        # 覆盖 _SelectorWidget.onmove 方法，因为多边形选择器需要处理移动回调，即使没有按钮按下
        # _SelectorWidget.onmove 方法包含逻辑，用于忽略移动事件，如果 _eventpress 为 None
        if not self.ignore(event):
            # 清理并处理事件
            event = self._clean_event(event)
            # 执行移动事件处理
            self._onmove(event)
            return True
        return False
    # 光标移动事件处理程序。
    def _onmove(self, event):
        """Cursor move event handler."""
        
        # 如果有活动的顶点（ToolHandle），移动该顶点。
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            # 更新活动顶点的坐标为当前光标位置的数据坐标。
            self._xys[idx] = self._get_data_coords(event)
            
            # 如果第一个顶点是活动的，并且多边形已完成，同时更新多边形线的结束点。
            if idx == 0 and self._selection_completed:
                self._xys[-1] = self._get_data_coords(event)

        # 如果是移动所有顶点。
        elif 'move_all' in self._state and self._eventpress:
            xdata, ydata = self._get_data_coords(event)
            dx = xdata - self._eventpress.xdata
            dy = ydata - self._eventpress.ydata
            # 对所有顶点进行平移。
            for k in range(len(self._xys)):
                x_at_press, y_at_press = self._xys_at_press[k]
                self._xys[k] = x_at_press + dx, y_at_press + dy

        # 如果已完成选择或者正在等待移动操作，则不做任何操作。
        elif (self._selection_completed
              or 'move_vertex' in self._state or 'move_all' in self._state):
            return

        # 否则，处理待定顶点的位置。
        else:
            # 计算到起始顶点的距离。
            x0, y0 = \
                self._selection_artist.get_transform().transform(self._xys[0])
            v0_dist = np.hypot(x0 - event.x, y0 - event.y)
            
            # 如果顶点数大于3且距离起始顶点小于抓取范围，则锁定到起始顶点。
            if len(self._xys) > 3 and v0_dist < self.grab_range:
                self._xys[-1] = self._xys[0]
            else:
                self._xys[-1] = self._get_data_coords(event)

        # 重新绘制多边形。
        self._draw_polygon()

    # 按键按下事件处理程序。
    def _on_key_press(self, event):
        """Key press event handler."""
        
        # 如果处于 'move_vertex' 或 'move_all' 模式，并且未完成选择，则移除待定顶点。
        if (not self._selection_completed
                and ('move_vertex' in self._state or
                     'move_all' in self._state)):
            self._xys.pop()
            self._draw_polygon()

    # 按键释放事件处理程序。
    def _on_key_release(self, event):
        """Key release event handler."""
        
        # 如果未完成选择，并且按下的键与 'move_vertex' 或 'move_all' 模式匹配，则添加回待定顶点。
        if (not self._selection_completed
                and
                (event.key == self._state_modifier_keys.get('move_vertex')
                 or event.key == self._state_modifier_keys.get('move_all'))):
            self._xys.append(self._get_data_coords(event))
            self._draw_polygon()
        
        # 如果释放的键是 'clear' 键，则重置多边形。
        elif event.key == self._state_modifier_keys.get('clear'):
            event = self._clean_event(event)
            self._xys = [self._get_data_coords(event)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)
    # 重新绘制多边形，根据新的顶点位置，但不调用update()方法。
    def _draw_polygon_without_update(self):
        """Redraw the polygon based on new vertex positions, no update()."""
        # 如果存在顶点列表_xys，则将x和y坐标分别提取到xs和ys中
        xs, ys = zip(*self._xys) if self._xys else ([], [])
        # 更新选择区域的图形数据
        self._selection_artist.set_data(xs, ys)
        # 更新多边形的边界框
        self._update_box()
        # 只在多边形完成时或用户锁定在起始顶点时显示起始和结束顶点的工具手柄
        if (self._selection_completed
                or (len(self._xys) > 3
                    and self._xys[-1] == self._xys[0])):
            # 设置多边形手柄数据，排除最后一个顶点以避免重叠
            self._polygon_handles.set_data(xs[:-1], ys[:-1])
        else:
            # 设置多边形手柄数据，包括所有顶点
            self._polygon_handles.set_data(xs, ys)

    # 重新绘制多边形，根据新的顶点位置，并调用update()方法。
    def _draw_polygon(self):
        """Redraw the polygon based on the new vertex positions."""
        # 调用不包含update()的多边形重绘方法
        self._draw_polygon_without_update()
        # 更新显示，即调用update()方法
        self.update()

    @property
    def verts(self):
        """The polygon vertices, as a list of ``(x, y)`` pairs."""
        # 返回多边形顶点_xys列表，不包含最后一个重复的顶点
        return self._xys[:-1]

    @verts.setter
    def verts(self, xys):
        """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
        # 设置多边形顶点_xys列表，将新顶点列表xys和其第一个顶点复制一份并添加到末尾
        self._xys = [*xys, xys[0]]
        # 标记多边形选取已完成
        self._selection_completed = True
        # 设置多边形可见
        self.set_visible(True)
        # 如果需要绘制边界框且当前没有边界框，则添加边界框
        if self._draw_box and self._box is None:
            self._add_box()
        # 重新绘制多边形
        self._draw_polygon()

    # 清除多边形但不调用update()方法。
    def _clear_without_update(self):
        # 标记多边形选取未完成
        self._selection_completed = False
        # 将顶点列表_xys重置为一个包含(0, 0)的列表
        self._xys = [(0, 0)]
        # 清除多边形的显示，不包含update()方法的调用
        self._draw_polygon_without_update()
    """
    Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    `~matplotlib.path.Path.contains_point` to select data points from an image.

    Unlike `LassoSelector`, this must be initialized with a starting
    point *xy*, and the `Lasso` events are destroyed upon release.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    xy : (float, float)
        Coordinates of the start of the lasso.
    callback : callable
        Whenever the lasso is released, the *callback* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props: dict, optional
        Lasso line properties. See `.Line2D` for valid properties.
        Default *props* are::

            {'linestyle' : '-', 'color' : 'black', 'lw' : 2}

        .. versionadded:: 3.9
    """
    # 定义 Lasso 类，继承自 AxesWidget
    def __init__(self, ax, xy, callback, *, useblit=True, props=None):
        # 调用父类的初始化方法
        super().__init__(ax)

        # 根据是否支持 blit 功能来确定是否使用 blit
        self.useblit = useblit and self.canvas.supports_blit
        # 如果使用 blit，创建并保存当前 Axes 区域的背景
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # 设置默认的样式
        style = {'linestyle': '-', 'color': 'black', 'lw': 2}

        # 如果提供了额外的样式参数，更新默认样式
        if props is not None:
            style.update(props)

        # 获取起始点的坐标
        x, y = xy
        # 初始化顶点列表，并创建用于绘制的 Line2D 对象
        self.verts = [(x, y)]
        self.line = Line2D([x], [y], **style)
        # 将线条对象添加到 Axes 中进行显示
        self.ax.add_line(self.line)
        # 存储回调函数，用于处理释放后的事件
        self.callback = callback
        # 连接按钮释放事件和鼠标移动事件到相应的处理方法
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

    # 处理释放鼠标按钮事件的方法
    def onrelease(self, event):
        # 如果事件应该被忽略，则直接返回
        if self.ignore(event):
            return
        # 如果存在顶点列表，将当前数据坐标添加到顶点列表中
        if self.verts is not None:
            self.verts.append(self._get_data_coords(event))
            # 如果顶点数量超过2个，调用回调函数处理选中的路径顶点
            if len(self.verts) > 2:
                self.callback(self.verts)
            # 移除当前绘制的线条对象
            self.line.remove()
        # 重置顶点列表为 None，并断开所有事件连接
        self.verts = None
        self.disconnect_events()

    # 处理鼠标移动事件的方法
    def onmove(self, event):
        # 如果事件应该被忽略，则直接返回
        if (self.ignore(event)
                or self.verts is None
                or event.button != 1
                or not self.ax.contains(event)[0]):
            return
        # 将当前数据坐标添加到顶点列表中
        self.verts.append(self._get_data_coords(event))
        # 更新线条对象的数据
        self.line.set_data(list(zip(*self.verts)))

        # 如果使用 blit，恢复背景并重新绘制线条
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
```