# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_qt.py`

```py
# 导入 functools 模块，用于高阶函数操作
import functools
# 导入 os 模块，提供对操作系统的接口
import os
# 导入 sys 模块，提供对 Python 解释器的访问
import sys
# 导入 traceback 模块，用于提取和格式化异常的回溯信息

# 导入 matplotlib 库，并从中导入 _api, backend_tools, cbook 模块
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
# 从 matplotlib._pylab_helpers 模块导入 Gcf 类
from matplotlib._pylab_helpers import Gcf
# 从 matplotlib.backend_bases 模块导入一系列基础类
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase, cursors, ToolContainerBase, MouseButton,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent,
    _allow_interrupt)
# 导入 matplotlib.backends.qt_editor.figureoptions 模块
import matplotlib.backends.qt_editor.figureoptions as figureoptions
# 从当前包中导入 qt_compat 模块和其中的 QtCore, QtGui, QtWidgets, __version__, QT_API, _to_int, _isdeleted 函数
from . import qt_compat
from .qt_compat import (
    QtCore, QtGui, QtWidgets, __version__, QT_API, _to_int, _isdeleted)


# SPECIAL_KEYS 定义了特殊键的映射关系，以 Qt::Key 形式存储
SPECIAL_KEYS = {
    _to_int(getattr(QtCore.Qt.Key, k)): v for k, v in [
        ("Key_Escape", "escape"),
        ("Key_Tab", "tab"),
        ("Key_Backspace", "backspace"),
        ("Key_Return", "enter"),
        ("Key_Enter", "enter"),
        ("Key_Insert", "insert"),
        ("Key_Delete", "delete"),
        ("Key_Pause", "pause"),
        ("Key_SysReq", "sysreq"),
        ("Key_Clear", "clear"),
        ("Key_Home", "home"),
        ("Key_End", "end"),
        ("Key_Left", "left"),
        ("Key_Up", "up"),
        ("Key_Right", "right"),
        ("Key_Down", "down"),
        ("Key_PageUp", "pageup"),
        ("Key_PageDown", "pagedown"),
        ("Key_Shift", "shift"),
        # macOS 下，控制键和超级键（cmd/apple）的位置相反
        ("Key_Control", "control" if sys.platform != "darwin" else "cmd"),
        ("Key_Meta", "meta" if sys.platform != "darwin" else "control"),
        ("Key_Alt", "alt"),
        ("Key_CapsLock", "caps_lock"),
        ("Key_F1", "f1"),
        ("Key_F2", "f2"),
        ("Key_F3", "f3"),
        ("Key_F4", "f4"),
        ("Key_F5", "f5"),
        ("Key_F6", "f6"),
        ("Key_F7", "f7"),
        ("Key_F8", "f8"),
        ("Key_F9", "f9"),
        ("Key_F10", "f10"),
        ("Key_F10", "f11"),
        ("Key_F12", "f12"),
        ("Key_Super_L", "super"),
        ("Key_Super_R", "super"),
    ]
}

# _MODIFIER_KEYS 定义了需要收集的键盘事件中的修饰键，以 Qt::KeyboardModifiers 和 Qt::Key 元组形式存储
_MODIFIER_KEYS = [
    (_to_int(getattr(QtCore.Qt.KeyboardModifier, mod)),
     _to_int(getattr(QtCore.Qt.Key, key)))
    for mod, key in [
        ("ControlModifier", "Key_Control"),
        ("AltModifier", "Key_Alt"),
        ("ShiftModifier", "Key_Shift"),
        ("MetaModifier", "Key_Meta"),
    ]
]

# cursord 定义了指针形状的映射关系，以 Matplotlib 中定义的常量为键，Qt 中的光标类型为值
cursord = {
    k: getattr(QtCore.Qt.CursorShape, v) for k, v in [
        (cursors.MOVE, "SizeAllCursor"),
        (cursors.HAND, "PointingHandCursor"),
        (cursors.POINTER, "ArrowCursor"),
        (cursors.SELECT_REGION, "CrossCursor"),
        (cursors.WAIT, "WaitCursor"),
        (cursors.RESIZE_HORIZONTAL, "SizeHorCursor"),
        (cursors.RESIZE_VERTICAL, "SizeVerCursor"),
    ]
}
# lru_cache 保留对 QApplication 实例的引用，防止其被垃圾回收。
@functools.lru_cache(1)
# 创建并返回一个 QApplication 实例
def _create_qApp():
    # 获取当前的 QApplication 实例
    app = QtWidgets.QApplication.instance()

    # 如果不存在 QApplication 实例，则创建一个新的 QApplication 并进行配置，因为同一时间只能存在一个 QApplication
    if app is None:
        # 检查显示是否有效，仅当在 Linux 上无法打开 X11 或 Wayland 显示时返回 False
        if not mpl._c_internal_utils.display_is_valid():
            raise RuntimeError('Invalid DISPLAY variable')

        # 检查确保不会在进程中实例化来自不同主要 Qt 版本的 QApplication
        if QT_API in {'PyQt6', 'PySide6'}:
            other_bindings = ('PyQt5', 'PySide2')
            qt_version = 6
        elif QT_API in {'PyQt5', 'PySide2'}:
            other_bindings = ('PyQt6', 'PySide6')
            qt_version = 5
        else:
            raise RuntimeError("Should never be here")

        # 检查其他主要版本的 Qt 中是否已经存在一个 QApplication 实例
        for binding in other_bindings:
            mod = sys.modules.get(f'{binding}.QtWidgets')
            if mod is not None and mod.QApplication.instance() is not None:
                other_core = sys.modules.get(f'{binding}.QtCore')
                _api.warn_external(
                    f'Matplotlib is using {QT_API} which wraps '
                    f'{QtCore.qVersion()} however an instantiated '
                    f'QApplication from {binding} which wraps '
                    f'{other_core.qVersion()} exists.  Mixing Qt major '
                    'versions may not work as expected.'
                )
                break

        # 根据 Qt 版本设置特定的应用程序属性和高 DPI 缩放策略
        if qt_version == 5:
            try:
                QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
            except AttributeError:  # 仅适用于 Qt>=5.6，<6.
                pass
        try:
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        except AttributeError:  # 仅适用于 Qt>=5.14.
            pass

        # 创建新的 QApplication 实例，并根据操作系统设置图标和退出策略
        app = QtWidgets.QApplication(["matplotlib"])
        if sys.platform == "darwin":
            image = str(cbook._get_data_path('images/matplotlib.svg'))
            icon = QtGui.QIcon(image)
            app.setWindowIcon(icon)
        app.setQuitOnLastWindowClosed(True)
        cbook._setup_new_guiapp()

        # 如果是 Qt 版本 5，设置特定的应用程序属性
        if qt_version == 5:
            app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    # 返回创建或已存在的 QApplication 实例
    return app


def _allow_interrupt_qt(qapp_or_eventloop):
    """A context manager that allows terminating a plot by sending a SIGINT."""

    # Use QSocketNotifier to read the socketpair while the Qt event loop runs.
    def prepare_notifier(rsock):
        # 创建一个 Qt 的套接字通知器，关联到给定套接字的文件描述符上，用于读取事件
        sn = QtCore.QSocketNotifier(rsock.fileno(), QtCore.QSocketNotifier.Type.Read)

        @sn.activated.connect
        def _may_clear_sock():
            # 在套接字被激活时运行一个 Python 函数，这样解释器有机会在 Python 层面处理信号。
            # 我们还需要通过 recv() 方法从套接字读取数据，重新启用它，因为它将作为唤醒的一部分被写入。
            # （我们需要这样做是因为 set_wakeup_fd 可能捕获到除了 SIGINT 以外的信号，我们需要继续等待。）
            try:
                rsock.recv(1)
            except BlockingIOError:
                # 在 Windows 上可能会过早触发或多次触发，因此对读取空套接字要宽容处理。
                pass

        return sn  # 实际上保持通知器活动。

    def handle_sigint():
        # 如果 qapp_or_eventloop 对象有 closeAllWindows 方法，调用它关闭所有窗口。
        if hasattr(qapp_or_eventloop, 'closeAllWindows'):
            qapp_or_eventloop.closeAllWindows()
        # 退出 Qt 应用程序事件循环或事件处理器。
        qapp_or_eventloop.quit()

    # 返回一个函数 _allow_interrupt，它接受 prepare_notifier 和 handle_sigint 作为参数。
    return _allow_interrupt(prepare_notifier, handle_sigint)
class TimerQT(TimerBase):
    """Subclass of `.TimerBase` using QTimer events."""

    def __init__(self, *args, **kwargs):
        # 创建一个新的 QTimer 对象，并将 timeout() 信号连接到 _on_timer 方法
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._on_timer)
        super().__init__(*args, **kwargs)

    def __del__(self):
        # 检查 _timer 是否已删除，以避免在 PySide2 动画关闭时出错
        if not _isdeleted(self._timer):
            self._timer_stop()

    def _timer_set_single_shot(self):
        # 设置 QTimer 为单次触发模式
        self._timer.setSingleShot(self._single)

    def _timer_set_interval(self):
        # 设置 QTimer 的间隔时间
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        # 启动 QTimer
        self._timer.start()

    def _timer_stop(self):
        # 停止 QTimer
        self._timer.stop()


class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):
    required_interactive_framework = "qt"
    _timer_cls = TimerQT
    manager_class = _api.classproperty(lambda cls: FigureManagerQT)

    buttond = {
        getattr(QtCore.Qt.MouseButton, k): v for k, v in [
            ("LeftButton", MouseButton.LEFT),
            ("RightButton", MouseButton.RIGHT),
            ("MiddleButton", MouseButton.MIDDLE),
            ("XButton1", MouseButton.BACK),
            ("XButton2", MouseButton.FORWARD),
        ]
    }

    def __init__(self, figure=None):
        _create_qApp()
        super().__init__(figure=figure)

        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None
        self._in_resize_event = False

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.resize(*self.get_width_height())

        palette = QtGui.QPalette(QtGui.QColor("white"))
        self.setPalette(palette)

    def _update_pixel_ratio(self):
        if self._set_device_pixel_ratio(
                self.devicePixelRatioF() or 1):  # rarely, devicePixelRatioF=0
            # 触发 resizeEvent 以重新调整画布大小
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    def _update_screen(self, screen):
        # 处理窗口连接到的屏幕变化
        self._update_pixel_ratio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._update_pixel_ratio)
            screen.logicalDotsPerInchChanged.connect(self._update_pixel_ratio)

    def showEvent(self, event):
        # 当窗口显示时设置正确的像素比例，并连接任何与像素比例变化相关的信号
        window = self.window().windowHandle()
        window.screenChanged.connect(self._update_screen)
        self._update_screen(window.screen())
    # 设置鼠标光标位置
    def set_cursor(self, cursor):
        # 调用继承的文档字符串
        self.setCursor(_api.check_getitem(cursord, cursor=cursor))

    # 计算鼠标事件的物理像素坐标
    def mouseEventCoords(self, pos=None):
        """
        Calculate mouse coordinates in physical pixels.

        Qt uses logical pixels, but the figure is scaled to physical
        pixels for rendering.  Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        if pos is None:
            # 使用全局鼠标位置映射到窗口坐标系中
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
        elif hasattr(pos, "position"):  # qt6 QtGui.QEvent
            # 如果 pos 具有 position 属性，使用其位置
            pos = pos.position()
        elif hasattr(pos, "pos"):  # qt5 QtCore.QEvent
            # 如果 pos 具有 pos 属性，使用其位置
            pos = pos.pos()
        # (否则，pos 已经是 QPoint 类型)
        x = pos.x()
        # 翻转 y 轴，使 y=0 在画布底部
        y = self.figure.bbox.height / self.device_pixel_ratio - pos.y()
        return x * self.device_pixel_ratio, y * self.device_pixel_ratio

    # 鼠标进入窗口事件处理
    def enterEvent(self, event):
        # 强制查询修饰键的状态，因为缓存的修饰键状态可能在窗口失去焦点期间无效
        mods = QtWidgets.QApplication.instance().queryKeyboardModifiers()
        if self.figure is None:
            return
        # 创建 LocationEvent 实例并处理 "figure_enter_event" 事件
        LocationEvent("figure_enter_event", self,
                      *self.mouseEventCoords(event),
                      modifiers=self._mpl_modifiers(mods),
                      guiEvent=event)._process()

    # 鼠标离开窗口事件处理
    def leaveEvent(self, event):
        # 恢复覆盖的光标状态
        QtWidgets.QApplication.restoreOverrideCursor()
        if self.figure is None:
            return
        # 创建 LocationEvent 实例并处理 "figure_leave_event" 事件
        LocationEvent("figure_leave_event", self,
                      *self.mouseEventCoords(),
                      modifiers=self._mpl_modifiers(),
                      guiEvent=event)._process()

    # 鼠标按下事件处理
    def mousePressEvent(self, event):
        # 获取事件按钮对应的按钮名称
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            # 创建 MouseEvent 实例并处理 "button_press_event" 事件
            MouseEvent("button_press_event", self,
                       *self.mouseEventCoords(event), button,
                       modifiers=self._mpl_modifiers(),
                       guiEvent=event)._process()

    # 鼠标双击事件处理
    def mouseDoubleClickEvent(self, event):
        # 获取事件按钮对应的按钮名称
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            # 创建 MouseEvent 实例并处理 "button_press_event" 事件，指示为双击
            MouseEvent("button_press_event", self,
                       *self.mouseEventCoords(event), button, dblclick=True,
                       modifiers=self._mpl_modifiers(),
                       guiEvent=event)._process()

    # 鼠标移动事件处理
    def mouseMoveEvent(self, event):
        if self.figure is None:
            return
        # 创建 MouseEvent 实例并处理 "motion_notify_event" 事件
        MouseEvent("motion_notify_event", self,
                   *self.mouseEventCoords(event),
                   modifiers=self._mpl_modifiers(),
                   guiEvent=event)._process()
    # 处理鼠标释放事件的方法
    def mouseReleaseEvent(self, event):
        # 获取鼠标释放的按钮
        button = self.buttond.get(event.button())
        # 如果按钮不为空且图形对象存在
        if button is not None and self.figure is not None:
            # 创建 MouseEvent 对象并处理
            MouseEvent("button_release_event", self,
                       *self.mouseEventCoords(event), button,
                       modifiers=self._mpl_modifiers(),
                       guiEvent=event)._process()

    # 处理滚轮事件的方法
    def wheelEvent(self, event):
        # 根据文档，检查 pixelDelta 是否为空或者在 X11 平台上不可靠
        if (event.pixelDelta().isNull()
                or QtWidgets.QApplication.instance().platformName() == "xcb"):
            # 如果 pixelDelta 为空或在 X11 平台上，则使用 angleDelta 计算步数
            steps = event.angleDelta().y() / 120
        else:
            # 否则使用 pixelDelta 的 y 值作为步数
            steps = event.pixelDelta().y()
        # 如果步数不为零且图形对象存在
        if steps and self.figure is not None:
            # 创建 MouseEvent 对象并处理
            MouseEvent("scroll_event", self,
                       *self.mouseEventCoords(event), step=steps,
                       modifiers=self._mpl_modifiers(),
                       guiEvent=event)._process()

    # 处理键盘按下事件的方法
    def keyPressEvent(self, event):
        # 获取按键信息
        key = self._get_key(event)
        # 如果按键信息不为空且图形对象存在
        if key is not None and self.figure is not None:
            # 创建 KeyEvent 对象并处理
            KeyEvent("key_press_event", self,
                     key, *self.mouseEventCoords(),
                     guiEvent=event)._process()

    # 处理键盘释放事件的方法
    def keyReleaseEvent(self, event):
        # 获取按键信息
        key = self._get_key(event)
        # 如果按键信息不为空且图形对象存在
        if key is not None and self.figure is not None:
            # 创建 KeyEvent 对象并处理
            KeyEvent("key_release_event", self,
                     key, *self.mouseEventCoords(),
                     guiEvent=event)._process()

    # 处理窗口大小调整事件的方法
    def resizeEvent(self, event):
        # 防止 PyQt6 递归
        if self._in_resize_event:
            return
        # 如果图形对象不存在，则返回
        if self.figure is None:
            return
        # 设置 resize 事件标志
        self._in_resize_event = True
        try:
            # 计算调整后的宽度和高度，并根据设备像素比例调整
            w = event.size().width() * self.device_pixel_ratio
            h = event.size().height() * self.device_pixel_ratio
            dpival = self.figure.dpi
            winch = w / dpival
            hinch = h / dpival
            # 设置图形对象的大小
            self.figure.set_size_inches(winch, hinch, forward=False)
            # 调用 QWidget 的 resizeEvent 方法完成调整
            QtWidgets.QWidget.resizeEvent(self, event)
            # 发送 resize_event 事件
            ResizeEvent("resize_event", self)._process()
            # 更新绘图
            self.draw_idle()
        finally:
            # 清除 resize 事件标志
            self._in_resize_event = False

    # 返回推荐的窗口大小的方法
    def sizeHint(self):
        # 获取宽度和高度，返回 QSize 对象
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    # 返回最小推荐大小的方法
    def minumumSizeHint(self):
        # 返回固定的 QSize 对象
        return QtCore.QSize(10, 10)
    def _mpl_modifiers(modifiers=None, *, exclude=None):
        # 如果 modifiers 为 None，则获取当前应用程序的键盘修饰符
        if modifiers is None:
            modifiers = QtWidgets.QApplication.instance().keyboardModifiers()
        # 将修饰符转换为整数表示
        modifiers = _to_int(modifiers)
        # 获取按下的修饰键的名称
        # 当作为单独的按键时，'control' 被称为 'control'，但作为修饰符时被称为 'ctrl'
        # 使用位操作从修饰符位掩码中提取修饰键
        # 如果 exclude 是一个修饰键，则在 mods 中不应重复出现
        return [SPECIAL_KEYS[key].replace('control', 'ctrl')
                for mask, key in _MODIFIER_KEYS
                if exclude != key and modifiers & mask]

    def _get_key(self, event):
        # 获取事件的按键值
        event_key = event.key()
        # 使用 _mpl_modifiers 函数获取修饰键列表，排除当前按下的按键
        mods = self._mpl_modifiers(exclude=event_key)
        try:
            # 对于某些按键（如回车、左键、退格键等），使用一个单词来表示按键，而不是 Unicode
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            # Unicode 定义了最高到 0x10ffff 的代码点（sys.maxunicode）
            # QT 会使用大于此值的键码代表不是 Unicode 字符的键（如多媒体键）
            # 跳过这些键
            # 如果确实需要它们，应将它们添加到 SPECIAL_KEYS 中
            if event_key > sys.maxunicode:
                return None
            # 将键码转换为对应的字符
            key = chr(event_key)
            # QT 传递的字母是大写的。修正大小写
            # 注意 CapsLock 是被忽略的
            if 'shift' in mods:
                mods.remove('shift')
            else:
                key = key.lower()

        # 返回修饰键和按键组合成的字符串
        return '+'.join(mods + [key])

    def flush_events(self):
        # 继承的文档字符串
        QtWidgets.QApplication.instance().processEvents()

    def start_event_loop(self, timeout=0):
        # 继承的文档字符串
        if hasattr(self, "_event_loop") and self._event_loop.isRunning():
            raise RuntimeError("Event loop already running")
        # 创建事件循环对象
        self._event_loop = event_loop = QtCore.QEventLoop()
        # 如果设置了超时时间，创建一个定时器，在超时后退出事件循环
        if timeout > 0:
            _ = QtCore.QTimer.singleShot(int(timeout * 1000), event_loop.quit)

        # 允许 Qt 中断的上下文管理器
        with _allow_interrupt_qt(event_loop):
            qt_compat._exec(event_loop)

    def stop_event_loop(self, event=None):
        # 继承的文档字符串
        if hasattr(self, "_event_loop"):
            # 退出事件循环
            self._event_loop.quit()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw."""
        # 在此进行渲染器的绘制；延迟会导致使用 draw() 结果更新图表元素的问题
        if self._is_drawing:
            return
        # 设置 _is_drawing 属性为 True，确保在 draw() 运行期间不重入
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        # 更新 Qt 绘制
        self.update()
    def draw_idle(self):
        """Queue redraw of the Agg buffer and request Qt paintEvent."""
        # 将 Agg 缓冲区的重绘排队，并请求 Qt 的 paintEvent。
        # Agg 绘制需要由 Matplotlib 修改场景图的同一线程处理。
        # 将 Agg 绘制请求发布到当前事件循环，以确保线程关联性，并累积来自事件处理的多个绘制请求。
        # TODO: 使用队列信号连接可能比 singleShot 更安全
        if not (getattr(self, '_draw_pending', False) or
                getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def blit(self, bbox=None):
        # docstring inherited
        # 如果 bbox 为 None 并且存在 figure，则将整个画布进行 blit。
        if bbox is None and self.figure:
            bbox = self.figure.bbox
        # 重新绘制使用逻辑像素，而不是像渲染器一样使用物理像素。
        l, b, w, h = [int(pt / self.device_pixel_ratio) for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.rect().height() - t, w, h)

    def _draw_idle(self):
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() < 0 or self.width() < 0:
                return
            try:
                self.draw()
            except Exception:
                # 未捕获的异常对于 PyQt5 是致命的，因此捕获它们。
                traceback.print_exc()

    def drawRectangle(self, rect):
        # 将缩放矩形绘制到 QPainter。_draw_rect_callback 需要在 paintEvent 结束时调用。
        if rect is not None:
            x0, y0, w, h = [int(pt / self.device_pixel_ratio) for pt in rect]
            x1 = x0 + w
            y1 = y0 + h
            def _draw_rect_callback(painter):
                pen = QtGui.QPen(
                    QtGui.QColor("black"),
                    1 / self.device_pixel_ratio
                )

                pen.setDashPattern([3, 3])
                for color, offset in [
                        (QtGui.QColor("black"), 0),
                        (QtGui.QColor("white"), 3),
                ]:
                    pen.setDashOffset(offset)
                    pen.setColor(color)
                    painter.setPen(pen)
                    # 从 x0, y0 向 x1, y1 绘制线条，以避免缩放框移动时虚线“跳动”。
                    painter.drawLine(x0, y0, x0, y1)
                    painter.drawLine(x0, y0, x1, y0)
                    painter.drawLine(x0, y1, x1, y1)
                    painter.drawLine(x1, y0, x1, y1)
        else:
            def _draw_rect_callback(painter):
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()
class MainWindow(QtWidgets.QMainWindow):
    # 定义一个信号用于窗口关闭事件
    closing = QtCore.Signal()

    # 重写窗口关闭事件
    def closeEvent(self, event):
        # 发射关闭信号
        self.closing.emit()
        # 调用父类的关闭事件处理
        super().closeEvent(event)


class FigureManagerQT(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        图形画布的实例
    num : int or str
        图形的编号
    toolbar : qt.QToolBar
        Qt 工具栏实例
    window : qt.QMainWindow
        Qt 主窗口实例
    """

    def __init__(self, canvas, num):
        # 创建一个主窗口
        self.window = MainWindow()
        # 调用父类的初始化方法
        super().__init__(canvas, num)
        # 连接窗口关闭信号到自定义方法 _widgetclosed
        self.window.closing.connect(self._widgetclosed)

        # 如果不是 macOS 平台，则设置图标
        if sys.platform != "darwin":
            image = str(cbook._get_data_path('images/matplotlib.svg'))
            icon = QtGui.QIcon(image)
            self.window.setWindowIcon(icon)

        # 标记窗口尚未销毁
        self.window._destroying = False

        # 如果存在工具栏，则添加到主窗口
        if self.toolbar:
            self.window.addToolBar(self.toolbar)
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0

        # 调整主窗口大小以适应画布请求的大小
        cs = canvas.sizeHint()
        cs_height = cs.height()
        height = cs_height + tbs_height
        self.window.resize(cs.width(), height)

        # 将画布设置为主窗口的中央部件
        self.window.setCentralWidget(self.canvas)

        # 如果 Matplotlib 处于交互模式，则显示窗口并绘制画布
        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()

        # 将键盘焦点设置到画布上，而不是管理器
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()

        # 将窗口置于顶层
        self.window.raise_()

    def full_screen_toggle(self):
        # 切换全屏模式
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    def _widgetclosed(self):
        # 处理窗口关闭事件
        CloseEvent("close_event", self.canvas)._process()
        # 如果窗口正在销毁中，则直接返回
        if self.window._destroying:
            return
        # 标记窗口正在销毁
        self.window._destroying = True
        try:
            # 销毁 FigureManagerQT 实例
            Gcf.destroy(self)
        except AttributeError:
            # 捕获 AttributeError 异常
            pass
            # 当 Python 会话被终止时，可能会出现 Gcf 在 Gcf.destroy 之前已被销毁的情况，
            # 导致无效的 AttributeError。

    def resize(self, width, height):
        # Qt 方法返回 'virtual' 像素大小，因此需要从物理像素转换为逻辑像素
        width = int(width / self.canvas.device_pixel_ratio)
        height = int(height / self.canvas.device_pixel_ratio)
        extra_width = self.window.width() - self.canvas.width()
        extra_height = self.window.height() - self.canvas.height()
        # 调整画布大小
        self.canvas.resize(width, height)
        # 调整主窗口大小
        self.window.resize(width + extra_width, height + extra_height)
    @classmethod
    def start_main_loop(cls):
        # 获取当前的 QtWidgets.QApplication 实例
        qapp = QtWidgets.QApplication.instance()
        # 如果 qapp 存在
        if qapp:
            # 在允许中断 Qt 的上下文中执行主事件循环
            with _allow_interrupt_qt(qapp):
                qt_compat._exec(qapp)

    def show(self):
        # 标记窗口未在销毁状态
        self.window._destroying = False
        # 显示窗口
        self.window.show()
        # 如果 mpl.rcParams['figure.raise_window'] 为真
        if mpl.rcParams['figure.raise_window']:
            # 激活并提升窗口至前台
            self.window.activateWindow()
            self.window.raise_()

    def destroy(self, *args):
        # 检查 QtWidgets.QApplication 实例是否存在，PySide 在其 atexit 处理程序中会删除它
        if QtWidgets.QApplication.instance() is None:
            return
        # 如果窗口正在销毁，则返回
        if self.window._destroying:
            return
        # 标记窗口正在销毁
        self.window._destroying = True
        # 如果存在工具栏，则销毁工具栏
        if self.toolbar:
            self.toolbar.destroy()
        # 关闭窗口
        self.window.close()

    def get_window_title(self):
        # 返回窗口的标题
        return self.window.windowTitle()

    def set_window_title(self, title):
        # 设置窗口的标题
        self.window.setWindowTitle(title)
class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
    _message = QtCore.Signal(str)  # 信号用于发出消息，已弃用，将在3.8版本中移除。
    message = _api.deprecate_privatize_attribute("3.8")  # 消息属性，用于标记已弃用，将在3.8版本中移除。

    # 复制基类的工具项列表，并在"Subplots"后面插入"Customize"工具项
    toolitems = [*NavigationToolbar2.toolitems]
    toolitems.insert(
        [name for name, *_ in toolitems].index("Subplots") + 1,
        ("Customize", "Edit axis, curve and image parameters",
         "qt4_editor_options", "edit_parameters"))

    def __init__(self, canvas, parent=None, coordinates=True):
        """coordinates: 是否在工具栏右侧显示坐标信息"""
        QtWidgets.QToolBar.__init__(self, parent)
        # 设置工具栏允许出现在顶部和底部
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(
            _to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) |
            _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        self.coordinates = coordinates
        self._actions = {}  # 映射工具项方法名到QActions的字典
        self._subplot_dialog = None

        # 遍历工具项列表，为每个工具项添加对应的操作按钮
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()  # 添加分隔符
            else:
                slot = getattr(self, callback)
                # 修复Qt的一个bug（PYSIDE-2512），通过functools.partial包装回调函数
                slot = functools.wraps(slot)(functools.partial(slot))
                slot = QtCore.Slot()(slot)

                # 添加动作按钮到工具栏，设置图标、文本、回调函数
                a = self.addAction(self._icon(image_file + '.png'),
                                   text, slot)
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)  # 如果是'zoom'或'pan'工具，设置为可选中状态
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)  # 设置工具提示文本

        # 在工具栏右侧添加（x, y）坐标位置小部件
        # stretch factor为1，意味着在调整工具栏大小时，优先调整此标签而非按钮
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel("", self)
            self.locLabel.setAlignment(QtCore.Qt.AlignmentFlag(
                _to_int(QtCore.Qt.AlignmentFlag.AlignRight) |
                _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))

            self.locLabel.setSizePolicy(QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Ignored,
            ))
            labelAction = self.addWidget(self.locLabel)  # 添加小部件到工具栏
            labelAction.setVisible(True)

        NavigationToolbar2.__init__(self, canvas)  # 调用基类构造函数，初始化导航工具条
    def _icon(self, name):
        """
        Construct a `.QIcon` from an image file *name*, including the extension
        and relative to Matplotlib's "images" data directory.
        """
        # 从指定名称的图像文件构造一个 `.QIcon` 对象，包括扩展名，
        # 文件相对于 Matplotlib 的 "images" 数据目录。
        
        # 使用高分辨率图标（如果有 '_large' 后缀的版本）
        # 注意：用户提供的图标可能没有 '_large' 版本
        path_regular = cbook._get_data_path('images', name)
        path_large = path_regular.with_name(
            path_regular.name.replace('.png', '_large.png'))
        filename = str(path_large if path_large.exists() else path_regular)
        
        # 使用文件名创建 QPixmap 对象
        pm = QtGui.QPixmap(filename)
        # 设置设备像素比率，如果设备像素比率为 0，则设为 1
        pm.setDevicePixelRatio(
            self.devicePixelRatioF() or 1)  # rarely, devicePixelRatioF=0
        
        # 如果背景色亮度小于 128，则使用前景色填充图标
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            # 创建一个黑色掩码，并将其从图标中移除
            mask = pm.createMaskFromColor(
                QtGui.QColor('black'),
                QtCore.Qt.MaskMode.MaskOutColor)
            pm.fill(icon_color)
            pm.setMask(mask)
        
        # 返回 QIcon 对象
        return QtGui.QIcon(pm)

    def edit_parameters(self):
        # 获取当前图表中的所有坐标轴
        axes = self.canvas.figure.get_axes()
        
        # 如果没有坐标轴，则显示警告信息并返回
        if not axes:
            QtWidgets.QMessageBox.warning(
                self.canvas.parent(), "Error", "There are no Axes to edit.")
            return
        
        # 如果只有一个坐标轴，则直接使用这个坐标轴进行编辑
        elif len(axes) == 1:
            ax, = axes
        
        # 如果有多个坐标轴，则显示对话框让用户选择编辑的坐标轴
        else:
            # 获取所有坐标轴的标题
            titles = [
                ax.get_label() or
                ax.get_title() or
                ax.get_title("left") or
                ax.get_title("right") or
                " - ".join(filter(None, [ax.get_xlabel(), ax.get_ylabel()])) or
                f"<anonymous {type(ax).__name__}>"
                for ax in axes]
            
            # 找出重复的标题
            duplicate_titles = [
                title for title in titles if titles.count(title) > 1]
            
            # 对于有重复标题的坐标轴，给标题添加唯一标识符
            for i, ax in enumerate(axes):
                if titles[i] in duplicate_titles:
                    titles[i] += f" (id: {id(ax):#x})"  # Deduplicate titles.
            
            # 显示一个输入对话框，让用户选择要编辑的坐标轴
            item, ok = QtWidgets.QInputDialog.getItem(
                self.canvas.parent(),
                'Customize', 'Select Axes:', titles, 0, False)
            if not ok:
                return
            ax = axes[titles.index(item)]
        
        # 调用 figureoptions 模块的方法来编辑坐标轴参数
        figureoptions.figure_edit(ax, self)

    def _update_buttons_checked(self):
        # 同步按钮的选中状态以匹配活动模式
        if 'pan' in self._actions:
            self._actions['pan'].setChecked(self.mode.name == 'PAN')
        if 'zoom' in self._actions:
            self._actions['zoom'].setChecked(self.mode.name == 'ZOOM')

    def pan(self, *args):
        # 调用父类的 pan 方法
        super().pan(*args)
        # 更新按钮的选中状态
        self._update_buttons_checked()

    def zoom(self, *args):
        # 调用父类的 zoom 方法
        super().zoom(*args)
        # 更新按钮的选中状态
        self._update_buttons_checked()

    def set_message(self, s):
        # 发送消息信号
        self._message.emit(s)
        # 如果启用了坐标显示，则更新显示坐标的标签文本
        if self.coordinates:
            self.locLabel.setText(s)
    # 绘制橡皮筋效果，将画布的高度保存在变量中
    height = self.canvas.figure.bbox.height
    # 调整起始点和结束点的 y 坐标，以适应画布坐标系
    y1 = height - y1
    y0 = height - y0
    # 创建一个包含整数值的矩形列表，表示橡皮筋的位置和大小
    rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
    # 调用画布对象的方法绘制矩形
    self.canvas.drawRectangle(rect)

    # 移除橡皮筋效果，清空画布上的矩形
    self.canvas.drawRectangle(None)

    # 配置子图工具对话框，如果对话框尚未创建则创建它
    if self._subplot_dialog is None:
        self._subplot_dialog = SubplotToolQt(
            self.canvas.figure, self.canvas.parent())
        # 监听画布的关闭事件，并在关闭时拒绝（不保存）子图配置的更改
        self.canvas.mpl_connect(
            "close_event", lambda e: self._subplot_dialog.reject())
    # 更新子图工具对话框，使其显示当前的子图参数
    self._subplot_dialog.update_from_current_subplotpars()
    # 将对话框设置为模态（阻塞式）并显示
    self._subplot_dialog.setModal(True)
    self._subplot_dialog.show()
    # 返回子图工具对话框的引用
    return self._subplot_dialog

    # 保存当前图形到文件
    filetypes = self.canvas.get_supported_filetypes_grouped()
    sorted_filetypes = sorted(filetypes.items())
    default_filetype = self.canvas.get_default_filetype()

    # 获取保存路径的起始位置和默认文件名
    startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
    start = os.path.join(startpath, self.canvas.get_default_filename())
    filters = []
    selectedFilter = None

    # 构建文件类型过滤器列表，并找到默认选择的文件类型
    for name, exts in sorted_filetypes:
        exts_list = " ".join(['*.%s' % ext for ext in exts])
        filter = f'{name} ({exts_list})'
        if default_filetype in exts:
            selectedFilter = filter
        filters.append(filter)
    # 将过滤器列表转换为字符串形式
    filters = ';;'.join(filters)

    # 弹出文件保存对话框，获取用户选择的文件名和文件类型
    fname, filter = QtWidgets.QFileDialog.getSaveFileName(
        self.canvas.parent(), "Choose a filename to save to", start,
        filters, selectedFilter)
    if fname:
        # 如果设置了保存路径，则更新默认的保存路径设置
        if startpath != "":
            mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
        try:
            # 尝试将当前图形保存到指定文件名的文件中
            self.canvas.figure.savefig(fname)
        except Exception as e:
            # 如果保存失败，显示错误消息框
            QtWidgets.QMessageBox.critical(
                self, "Error saving file", str(e),
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.NoButton)

    # 设置历史记录按钮的状态，根据导航堆栈中的位置决定按钮是否可用
    can_backward = self._nav_stack._pos > 0
    can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
    if 'back' in self._actions:
        self._actions['back'].setEnabled(can_backward)
    if 'forward' in self._actions:
        self._actions['forward'].setEnabled(can_forward)
class SubplotToolQt(QtWidgets.QDialog):
    # 初始化函数，用于创建 SubplotToolQt 对象
    def __init__(self, targetfig, parent):
        super().__init__()
        # 设置窗口图标为 matplotlib 图标
        self.setWindowIcon(QtGui.QIcon(
            str(cbook._get_data_path("images/matplotlib.png"))))
        # 设置对象名称为 "SubplotTool"
        self.setObjectName("SubplotTool")
        # 用于存储各个 SpinBox 控件的字典
        self._spinboxes = {}
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        # 遍历不同的控件组，包括边框和间距
        for group, spinboxes, buttons in [
                ("Borders",
                 ["top", "bottom", "left", "right"],
                 [("Export values", self._export_values)]),
                ("Spacings",
                 ["hspace", "wspace"],
                 [("Tight layout", self._tight_layout),
                  ("Reset", self._reset),
                  ("Close", self.close)])]:
            # 创建垂直布局
            layout = QtWidgets.QVBoxLayout()
            main_layout.addLayout(layout)
            # 创建分组框
            box = QtWidgets.QGroupBox(group)
            layout.addWidget(box)
            # 在分组框中创建表单布局
            inner = QtWidgets.QFormLayout(box)
            # 遍历 SpinBox 的名称列表
            for name in spinboxes:
                # 创建 DoubleSpinBox 对象并设置范围、精度等属性
                self._spinboxes[name] = spinbox = QtWidgets.QDoubleSpinBox()
                spinbox.setRange(0, 1)
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.005)
                spinbox.setKeyboardTracking(False)
                # 连接值改变信号到对应的槽函数
                spinbox.valueChanged.connect(self._on_value_changed)
                inner.addRow(name, spinbox)
            layout.addStretch(1)
            # 遍历按钮名称和对应的方法
            for name, method in buttons:
                # 创建按钮对象
                button = QtWidgets.QPushButton(name)
                # 设置按钮不接受 <enter> 键，用于输入值
                button.setAutoDefault(False)
                # 连接按钮点击信号到对应的方法
                button.clicked.connect(method)
                layout.addWidget(button)
                # 如果按钮名称为 "Close"，设置焦点到该按钮
                if name == "Close":
                    button.setFocus()
        # 设置对象属性 _figure 为传入的 targetfig
        self._figure = targetfig
        # 初始化 _defaults 字典为空
        self._defaults = {}
        # 初始化 _export_values_dialog 为 None
        self._export_values_dialog = None
        # 更新界面的当前 subplot 参数设置
        self.update_from_current_subplotpars()

    # 更新当前 subplot 参数设置的方法
    def update_from_current_subplotpars(self):
        # 使用字典推导式从 _spinboxes 字典获取当前值，并存储到 _defaults 字典中
        self._defaults = {spinbox: getattr(self._figure.subplotpars, name)
                          for name, spinbox in self._spinboxes.items()}
        # 调用 _reset 方法，设置 SpinBox 的当前值，但不触发信号
        self._reset()  # Set spinbox current values without triggering signals.
    # 导出数值的对话框，显示当前设置的属性及其值，保留三位小数以避免出现形如 0.100...001 的数值
    def _export_values(self):
        # 创建一个模态对话框
        self._export_values_dialog = QtWidgets.QDialog()
        layout = QtWidgets.QVBoxLayout()
        self._export_values_dialog.setLayout(layout)
        
        # 创建一个只读的多行文本框
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        layout.addWidget(text)
        
        # 将属性名和其对应的数值格式化为字符串，并以逗号和换行符分隔，添加到文本框中
        text.setPlainText(
            ",\n".join(f"{attr}={spinbox.value():.3}"
                       for attr, spinbox in self._spinboxes.items()))
        
        # 调整文本框的高度，以便完整显示所有文本，同时添加一些填充空间
        size = text.maximumSize()
        size.setHeight(
            QtGui.QFontMetrics(text.document().defaultFont())
            .size(0, text.toPlainText()).height() + 20)
        text.setMaximumSize(size)
        
        # 显示导出数值的对话框
        self._export_values_dialog.show()

    # 当数值发生改变时触发的方法
    def _on_value_changed(self):
        spinboxes = self._spinboxes
        
        # 设置所有的最小值和最大值，以便在 _reset() 中也可以使用
        for lower, higher in [("bottom", "top"), ("left", "right")]:
            spinboxes[higher].setMinimum(spinboxes[lower].value() + .001)
            spinboxes[lower].setMaximum(spinboxes[higher].value() - .001)
        
        # 调整子图布局参数，根据当前各属性的数值设置
        self._figure.subplots_adjust(
            **{attr: spinbox.value() for attr, spinbox in spinboxes.items()})
        
        # 更新画布，使更改生效
        self._figure.canvas.draw_idle()

    # 调整子图布局以实现紧凑布局
    def _tight_layout(self):
        self._figure.tight_layout()
        
        # 将各个属性的值设置为当前子图布局参数的值
        for attr, spinbox in self._spinboxes.items():
            spinbox.blockSignals(True)
            spinbox.setValue(getattr(self._figure.subplotpars, attr))
            spinbox.blockSignals(False)
        
        # 更新画布，使更改生效
        self._figure.canvas.draw_idle()

    # 将所有控件重置为默认值
    def _reset(self):
        # 针对每个 spinbox，设置其范围为 0 到 1
        for spinbox, value in self._defaults.items():
            spinbox.setRange(0, 1)
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        
        # 触发数值改变事件，更新相关操作
        self._on_value_changed()
# 创建一个名为 ToolbarQt 的类，它继承自 ToolContainerBase 和 QtWidgets.QToolBar
class ToolbarQt(ToolContainerBase, QtWidgets.QToolBar):
    # 初始化方法
    def __init__(self, toolmanager, parent=None):
        # 调用 ToolContainerBase 的初始化方法
        ToolContainerBase.__init__(self, toolmanager)
        # 调用 QtWidgets.QToolBar 的初始化方法
        QtWidgets.QToolBar.__init__(self, parent)
        
        # 设置工具栏允许的区域为顶部和底部工具栏区域
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(
            _to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) |
            _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        
        # 创建一个消息标签，并设置其对齐方式为右对齐和垂直居中
        message_label = QtWidgets.QLabel("")
        message_label.setAlignment(QtCore.Qt.AlignmentFlag(
            _to_int(QtCore.Qt.AlignmentFlag.AlignRight) |
            _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))
        
        # 设置消息标签的大小策略，使其在水平方向上扩展，垂直方向上忽略
        message_label.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        ))
        
        # 将消息标签作为工具栏的一个部件添加
        self._message_action = self.addWidget(message_label)
        
        # 初始化工具项字典和组字典
        self._toolitems = {}
        self._groups = {}

    # 添加工具项方法
    def add_toolitem(
            self, name, group, position, image_file, description, toggle):
        # 创建一个 QToolButton 对象
        button = QtWidgets.QToolButton(self)
        
        # 如果提供了图像文件，则设置按钮的图标
        if image_file:
            button.setIcon(NavigationToolbar2QT._icon(self, image_file))
        
        # 设置按钮的文本为指定的名称
        button.setText(name)
        
        # 如果提供了描述，则设置按钮的工具提示为描述内容
        if description:
            button.setToolTip(description)

        # 定义按钮的点击处理函数，根据 toggle 参数决定是连接 toggled 信号还是 clicked 信号
        def handler():
            self.trigger_tool(name)
        
        # 如果 toggle 为 True，则设置按钮为可切换状态，并连接 toggled 信号
        if toggle:
            button.setCheckable(True)
            button.toggled.connect(handler)
        else:
            # 否则，连接 clicked 信号
            button.clicked.connect(handler)

        # 初始化工具项字典中指定名称的列表（如果不存在）
        self._toolitems.setdefault(name, [])
        
        # 将按钮添加到指定的组中，并按照指定的位置插入
        self._add_to_group(group, name, button, position)
        
        # 将按钮及其处理函数添加到工具项字典中指定名称的列表中
        self._toolitems[name].append((button, handler))

    # 将按钮添加到指定组中的私有方法
    def _add_to_group(self, group, name, button, position):
        # 获取指定组的按钮列表
        gr = self._groups.get(group, [])
        
        # 如果组列表为空，则在消息标签之后插入一个分隔符，并将其添加到组中
        if not gr:
            sep = self.insertSeparator(self._message_action)
            gr.append(sep)
        
        # 获取要插入的位置之前的按钮
        before = gr[position]
        
        # 将按钮插入到指定位置之前，并将其添加到组列表中
        widget = self.insertWidget(before, button)
        gr.insert(position, widget)
        
        # 更新组字典中指定组的按钮列表
        self._groups[group] = gr

    # 切换工具项状态的方法
    def toggle_toolitem(self, name, toggled):
        # 如果指定名称的工具项不存在，则直接返回
        if name not in self._toolitems:
            return
        
        # 遍历指定名称的所有按钮，并设置其切换状态
        for button, handler in self._toolitems[name]:
            button.toggled.disconnect(handler)
            button.setChecked(toggled)
            button.toggled.connect(handler)

    # 移除工具项的方法
    def remove_toolitem(self, name):
        # 移除工具项字典中指定名称的所有按钮，并将它们的父对象设置为 None
        for button, handler in self._toolitems.pop(name, []):
            button.setParent(None)

    # 设置消息文本的方法
    def set_message(self, s):
        # 获取消息标签对应的部件，并设置其文本内容为 s
        self.widgetForAction(self._message_action).setText(s)


# 将 ConfigureSubplotsQt 类注册为 FigureCanvasQT 类型的工具类
@backend_tools._register_tool_class(FigureCanvasQT)
class ConfigureSubplotsQt(backend_tools.ConfigureSubplotsBase):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用父类 ConfigureSubplotsBase 的初始化方法
        super().__init__(*args, **kwargs)
        
        # 初始化 subplot 对话框为 None
        self._subplot_dialog = None

    # 触发方法，调用 NavigationToolbar2QT 类的 configure_subplots 方法
    def trigger(self, *args):
        NavigationToolbar2QT.configure_subplots(self)


# 将 SaveFigureQt 类注册为 FigureCanvasQT 类型的工具类
@backend_tools._register_tool_class(FigureCanvasQT)
class SaveFigureQt(backend_tools.SaveFigureBase):
    # 定义一个方法 trigger，接受任意数量的参数
    def trigger(self, *args):
        # 调用 NavigationToolbar2QT 的 save_figure 方法，传入参数为 self._make_classic_style_pseudo_toolbar() 的返回值
        NavigationToolbar2QT.save_figure(
            self._make_classic_style_pseudo_toolbar())
# 将FigureCanvasQT类注册为backend_tools中的工具类
@backend_tools._register_tool_class(FigureCanvasQT)
class RubberbandQt(backend_tools.RubberbandBase):
    
    # 在画布上绘制橡皮筋的方法
    def draw_rubberband(self, x0, y0, x1, y1):
        # 调用导航工具栏2QT的方法绘制橡皮筋，使用经典风格的伪工具栏
        NavigationToolbar2QT.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    # 移除画布上的橡皮筋
    def remove_rubberband(self):
        # 调用导航工具栏2QT的方法移除橡皮筋，使用经典风格的伪工具栏
        NavigationToolbar2QT.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


# 将FigureCanvasQT类注册为backend_tools中的工具类
@backend_tools._register_tool_class(FigureCanvasQT)
class HelpQt(backend_tools.ToolHelpBase):
    
    # 触发显示帮助信息的方法
    def trigger(self, *args):
        # 使用QtWidgets模块显示帮助信息的消息框，内容为HTML格式的帮助文本
        QtWidgets.QMessageBox.information(None, "Help", self._get_help_html())


# 将FigureCanvasQT类注册为backend_tools中的工具类
@backend_tools._register_tool_class(FigureCanvasQT)
class ToolCopyToClipboardQT(backend_tools.ToolCopyToClipboardBase):
    
    # 触发复制画布内容到剪贴板的方法
    def trigger(self, *args, **kwargs):
        # 抓取画布的图像作为pixmap
        pixmap = self.canvas.grab()
        # 获取Qt应用程序实例的剪贴板并将pixmap设置为其中的图像
        QtWidgets.QApplication.instance().clipboard().setPixmap(pixmap)


# 设置FigureManagerQT的工具栏类为NavigationToolbar2QT
FigureManagerQT._toolbar2_class = NavigationToolbar2QT
# 设置FigureManagerQT的工具管理器工具栏类为ToolbarQt
FigureManagerQT._toolmanager_toolbar_class = ToolbarQt


# 将_BackendQT类导出为后端
@_Backend.export
class _BackendQT(_Backend):
    backend_version = __version__
    # 设置FigureCanvas类为FigureCanvasQT
    FigureCanvas = FigureCanvasQT
    # 设置FigureManager类为FigureManagerQT
    FigureManager = FigureManagerQT
    # 设置主循环为FigureManagerQT的启动主循环方法
    mainloop = FigureManagerQT.start_main_loop
```