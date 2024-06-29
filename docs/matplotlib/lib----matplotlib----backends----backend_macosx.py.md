# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_macosx.py`

```py
import os  # 导入标准库 os

import matplotlib as mpl  # 导入 matplotlib 库，使用 mpl 别名
from matplotlib import _api, cbook  # 导入 matplotlib 的内部模块 _api 和 cbook
from matplotlib._pylab_helpers import Gcf  # 导入 matplotlib 内部的 Gcf 类
from . import _macosx  # 导入当前包下的 _macosx 模块
from .backend_agg import FigureCanvasAgg  # 从当前包下的 backend_agg 模块导入 FigureCanvasAgg 类
from matplotlib.backend_bases import (  # 导入 matplotlib 的 backend_bases 模块中的多个类
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    ResizeEvent, TimerBase, _allow_interrupt)


class TimerMac(_macosx.Timer, TimerBase):
    """Subclass of `.TimerBase` using CFRunLoop timer events."""
    # TimerMac 类继承自 _macosx.Timer 和 TimerBase，使用 CFRunLoop 定时器事件


def _allow_interrupt_macos():
    """A context manager that allows terminating a plot by sending a SIGINT."""
    return _allow_interrupt(
        lambda rsock: _macosx.wake_on_fd_write(rsock.fileno()), _macosx.stop)
    # 返回一个上下文管理器，允许通过发送 SIGINT 终止绘图


class FigureCanvasMac(FigureCanvasAgg, _macosx.FigureCanvas, FigureCanvasBase):
    # docstring inherited
    # FigureCanvasMac 类继承自 FigureCanvasAgg、_macosx.FigureCanvas 和 FigureCanvasBase

    # Ideally this class would be `class FCMacAgg(FCAgg, FCMac)`
    # (FC=FigureCanvas) where FCMac would be an ObjC-implemented mac-specific
    # class also inheriting from FCBase (this is the approach with other GUI
    # toolkits).  However, writing an extension type inheriting from a Python
    # base class is slightly tricky (the extension type must be a heap type),
    # and we can just as well lift the FCBase base up one level, keeping it *at
    # the end* to have the right method resolution order.

    # Events such as button presses, mouse movements, and key presses are
    # handled in C and events (MouseEvent, etc.) are triggered from there.
    # 按钮按下、鼠标移动和按键等事件在 C 语言中处理，事件（如 MouseEvent 等）从那里触发。

    required_interactive_framework = "macosx"
    _timer_cls = TimerMac  # 设置 _timer_cls 属性为 TimerMac 类
    manager_class = _api.classproperty(lambda cls: FigureManagerMac)  # 设置 manager_class 属性为 FigureManagerMac 类

    def __init__(self, figure):
        super().__init__(figure=figure)  # 调用父类的初始化方法
        self._draw_pending = False  # 初始化 _draw_pending 属性为 False
        self._is_drawing = False  # 初始化 _is_drawing 属性为 False
        # Keep track of the timers that are alive
        self._timers = set()  # 初始化 _timers 属性为一个空集合

    def draw(self):
        """Render the figure and update the macosx canvas."""
        # 渲染图形并更新 macOS 的画布

        # The renderer draw is done here; delaying causes problems with code
        # that uses the result of the draw() to update plot elements.
        # 渲染绘制操作在这里完成；延迟会导致使用 draw() 结果更新图形元素的代码出现问题。
        if self._is_drawing:  # 如果正在绘制中，直接返回
            return
        with cbook._setattr_cm(self, _is_drawing=True):  # 使用 cbook._setattr_cm 设置 _is_drawing 为 True
            super().draw()  # 调用父类 FigureCanvasAgg 的 draw 方法
        self.update()  # 更新画布

    def draw_idle(self):
        # docstring inherited
        # 继承自父类的方法文档字符串

        if not (getattr(self, '_draw_pending', False) or
                getattr(self, '_is_drawing', False)):
            self._draw_pending = True  # 设置 _draw_pending 为 True
            # Add a singleshot timer to the eventloop that will call back
            # into the Python method _draw_idle to take care of the draw
            # 向事件循环添加一个单次定时器，该定时器会回调到 Python 方法 _draw_idle 处理绘制
            self._single_shot_timer(self._draw_idle)  # 调用 _single_shot_timer 方法，传入 self._draw_idle 方法作为回调
    def _single_shot_timer(self, callback):
        """
        Add a single shot timer with the given callback
        
        This method sets up a timer that triggers once after a short interval.
        Upon triggering, it executes the provided callback function and removes
        the timer from the internal list of active timers.
        """
        def callback_func(callback, timer):
            # Execute the provided callback function
            callback()
            # Remove the timer from the set of active timers
            self._timers.remove(timer)
        # Create a new single shot timer with interval 0
        timer = self.new_timer(interval=0)
        timer.single_shot = True
        # Associate the callback function with the timer
        timer.add_callback(callback_func, callback, timer)
        # Add the timer to the set of active timers
        self._timers.add(timer)
        # Start the timer
        timer.start()

    def _draw_idle(self):
        """
        Draw method for singleshot timer
        
        This method is intended to be used with a singleshot timer to handle
        drawing operations. It ensures that drawing is performed only once,
        even if multiple draw requests occur in quick succession.
        """
        with self._idle_draw_cntx():
            if not self._draw_pending:
                # Short-circuit if draw request has already been handled
                return
            # Mark draw as no longer pending
            self._draw_pending = False
            # Perform the draw operation
            self.draw()

    def blit(self, bbox=None):
        """
        Blit the region specified by bbox
        
        This method overrides the inherited blit method to update the figure
        after blitting the specified bounding box region.
        """
        # Call the inherited blit method to perform blitting
        super().blit(bbox)
        # Update the figure
        self.update()

    def resize(self, width, height):
        """
        Resize the figure
        
        This method adjusts the size of the figure based on the provided width
        and height values, taking into account the device pixel ratio.
        """
        # Calculate the scale factor based on DPI and device pixel ratio
        scale = self.figure.dpi / self.device_pixel_ratio
        # Adjust width and height according to the scale factor
        width /= scale
        height /= scale
        # Set the new size of the figure in inches
        self.figure.set_size_inches(width, height, forward=False)
        # Trigger a resize event to update the figure
        ResizeEvent("resize_event", self)._process()
        # Perform an idle draw to update the display
        self.draw_idle()

    def start_event_loop(self, timeout=0):
        """
        Start the event loop with optional timeout
        
        This method initiates the event loop, handling events and interactions
        within the plot, optionally specifying a timeout duration.
        """
        # Set up a SIGINT handler for macOS to allow terminating via CTRL-C
        with _allow_interrupt_macos():
            # Forward the start event loop request to the ObjC implementation
            self._start_event_loop(timeout=timeout)
class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        # 获取Matplotlib数据路径中的'images'文件夹路径
        data_path = cbook._get_data_path('images')
        # 从NavigationToolbar2的工具项中提取不为None的元组项
        _, tooltips, image_names, _ = zip(*NavigationToolbar2.toolitems)
        # 调用_macosx.NavigationToolbar2的构造函数，初始化工具栏
        _macosx.NavigationToolbar2.__init__(
            self, canvas,
            # 生成PDF文件名列表作为工具栏图标的路径
            tuple(str(data_path / image_name) + ".pdf"
                  for image_name in image_names if image_name is not None),
            # 过滤并生成工具提示文本元组
            tuple(tooltip for tooltip in tooltips if tooltip is not None))
        # 调用NavigationToolbar2的构造函数，初始化工具栏
        NavigationToolbar2.__init__(self, canvas)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        # 在画布上绘制橡皮筋矩形
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def remove_rubberband(self):
        # 从画布上移除橡皮筋矩形
        self.canvas.remove_rubberband()

    def save_figure(self, *args):
        # 获取保存图形的目录路径
        directory = os.path.expanduser(mpl.rcParams['savefig.directory'])
        # 选择保存图形文件的对话框
        filename = _macosx.choose_save_file('Save the figure',
                                            directory,
                                            self.canvas.get_default_filename())
        if filename is None:  # 如果用户取消保存操作
            return
        # 如果设置了保存图形的目录路径，则更新为新的目录
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(filename)
        # 保存图形到指定的文件名
        self.canvas.figure.savefig(filename)


class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    _toolbar2_class = NavigationToolbar2Mac

    def __init__(self, canvas, num):
        # 初始化FigureManagerMac对象
        self._shown = False
        # 调用_macosx.FigureManager的构造函数，初始化画布
        _macosx.FigureManager.__init__(self, canvas)
        # 获取Matplotlib数据路径中的'images/matplotlib.pdf'的路径
        icon_path = str(cbook._get_data_path('images/matplotlib.pdf'))
        # 设置FigureManager的图标
        _macosx.FigureManager.set_icon(icon_path)
        # 调用FigureManagerBase的构造函数，初始化画布和编号
        FigureManagerBase.__init__(self, canvas, num)
        # 设置窗口模式为配置文件中指定的模式
        self._set_window_mode(mpl.rcParams["macosx.window_mode"])
        # 如果存在工具栏，则更新工具栏
        if self.toolbar is not None:
            self.toolbar.update()
        # 如果处于交互模式，则显示画布并刷新
        if mpl.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def _close_button_pressed(self):
        # 销毁当前图形窗口
        Gcf.destroy(self)
        # 刷新画布事件队列
        self.canvas.flush_events()

    def destroy(self):
        # 清除所有未触发的定时器，避免内存泄漏
        while self.canvas._timers:
            timer = self.canvas._timers.pop()
            timer.stop()
        super().destroy()

    @classmethod
    def start_main_loop(cls):
        # 设置SIGINT信号处理器，允许通过CTRL-C终止绘图
        with _allow_interrupt_macos():
            _macosx.show()

    def show(self):
        # 如果图形已过期，则刷新画布
        if self.canvas.figure.stale:
            self.canvas.draw_idle()
        # 如果图形窗口尚未显示，则显示图形窗口
        if not self._shown:
            self._show()
            self._shown = True
        # 如果配置文件中设置图形窗口提升，则提升图形窗口
        if mpl.rcParams["figure.raise_window"]:
            self._raise()


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac
    # 将 FigureManagerMac 类的 start_main_loop 方法赋值给 mainloop 变量
    mainloop = FigureManagerMac.start_main_loop
```