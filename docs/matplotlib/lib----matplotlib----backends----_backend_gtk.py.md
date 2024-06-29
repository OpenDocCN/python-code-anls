# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_backend_gtk.py`

```py
"""
Common code for GTK3 and GTK4 backends.
"""

import logging  # 导入日志模块
import sys  # 导入系统相关模块

import matplotlib as mpl  # 导入matplotlib库，并重命名为mpl
from matplotlib import _api, backend_tools, cbook  # 导入matplotlib中的子模块_api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf  # 导入matplotlib._pylab_helpers中的Gcf类
from matplotlib.backend_bases import (  # 从matplotlib.backend_bases中导入多个类
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase)
from matplotlib.backend_tools import Cursors  # 导入matplotlib.backend_tools中的Cursors类

import gi  # 导入gi库，用于与GTK的Python绑定交互
# GTK3/GTK4后端已经调用了`gi.require_version`来设置所需的GTK版本

from gi.repository import Gdk, Gio, GLib, Gtk  # 从gi.repository中导入Gdk, Gio, GLib, Gtk等模块


try:
    gi.require_foreign("cairo")  # 尝试导入cairo模块，GTK基础后端需要cairo库
except ImportError as e:
    raise ImportError("Gtk-based backends require cairo") from e

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
_application = None  # 全局变量，用于存储应用程序对象的引用，初始值为None，作为占位符


def _shutdown_application(app):
    # 当应用程序在IPython中被Ctrl-C中断时，关闭所有窗口以防止意外关闭
    for win in app.get_windows():
        win.close()
    # PyGObject包装器错误地认为None是不允许的，否则我们可以调用下面这行代码：
    # Gio.Application.set_default(None)
    # 取而代之，我们设置这个属性并忽略具有它的默认应用程序：
    app._created_by_matplotlib = True
    global _application
    _application = None


def _create_application():
    global _application

    if _application is None:
        app = Gio.Application.get_default()
        if app is None or getattr(app, '_created_by_matplotlib', False):
            # 如果显示变量无效（display_is_valid返回False），只有在Linux上既不是X11也不是Wayland时才会发生
            if not mpl._c_internal_utils.display_is_valid():
                raise RuntimeError('Invalid DISPLAY variable')
            # 创建一个新的GTK应用程序对象
            _application = Gtk.Application.new('org.matplotlib.Matplotlib3',
                                               Gio.ApplicationFlags.NON_UNIQUE)
            # 这里必须连接激活信号，但我们不需要处理它，因为我们不进行任何远程处理
            _application.connect('activate', lambda *args, **kwargs: None)
            _application.connect('shutdown', _shutdown_application)
            _application.register()
            cbook._setup_new_guiapp()  # 设置新的GUI应用程序
        else:
            _application = app

    return _application


def mpl_to_gtk_cursor_name(mpl_cursor):
    # 将matplotlib中的光标类型转换为GTK光标名称
    return _api.check_getitem({
        Cursors.MOVE: "move",  # 移动光标
        Cursors.HAND: "pointer",  # 手形光标
        Cursors.POINTER: "default",  # 默认光标
        Cursors.SELECT_REGION: "crosshair",  # 区域选择光标
        Cursors.WAIT: "wait",  # 等待光标
        Cursors.RESIZE_HORIZONTAL: "ew-resize",  # 水平调整大小光标
        Cursors.RESIZE_VERTICAL: "ns-resize",  # 垂直调整大小光标
    }, cursor=mpl_cursor)


class TimerGTK(TimerBase):
    """Subclass of `.TimerBase` using GTK timer events."""

    def __init__(self, *args, **kwargs):
        self._timer = None
        super().__init__(*args, **kwargs)
    # 启动定时器功能前，需要确保之前的定时器已经停止，否则可能导致定时器 id 泄露，无法停止。
    def _timer_start(self):
        self._timer_stop()  # 停止当前定时器（如果存在）
        self._timer = GLib.timeout_add(self._interval, self._on_timer)
    
    # 停止当前定时器，如果存在的话。
    def _timer_stop(self):
        if self._timer is not None:
            GLib.source_remove(self._timer)
            self._timer = None
    
    # 修改定时器的时间间隔，只有在定时器已经启动的情况下才会先停止再重新启动。
    def _timer_set_interval(self):
        if self._timer is not None:
            self._timer_stop()  # 停止当前定时器
            self._timer_start()  # 重新启动定时器
    
    # 定时器到期时的回调函数
    def _on_timer(self):
        super()._on_timer()  # 调用父类的定时器回调方法
        
        # 如果有注册的回调函数并且不是单次触发模式，则返回 True 继续定时器；否则返回 False 停止定时器。
        if self.callbacks and not self._single:
            return True
        else:
            self._timer = None  # 清空定时器对象
            return False
class _FigureCanvasGTK(FigureCanvasBase):
    _timer_cls = TimerGTK



class _FigureManagerGTK(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : Gtk.Toolbar or Gtk.Box
        The toolbar
    vbox : Gtk.VBox
        The Gtk.VBox containing the canvas and toolbar
    window : Gtk.Window
        The Gtk.Window
    """

    def __init__(self, canvas, num):
        self._gtk_ver = gtk_ver = Gtk.get_major_version()  # 获取当前 GTK 主版本号

        app = _create_application()  # 创建应用程序实例
        self.window = Gtk.Window()  # 创建 Gtk 窗口对象
        app.add_window(self.window)  # 将窗口添加到应用程序中
        super().__init__(canvas, num)  # 调用父类的初始化方法

        if gtk_ver == 3:
            self.window.set_wmclass("matplotlib", "Matplotlib")  # 设置窗口的 WM_CLASS
            icon_ext = "png" if sys.platform == "win32" else "svg"  # 根据平台确定图标文件的扩展名
            self.window.set_icon_from_file(
                str(cbook._get_data_path(f"images/matplotlib.{icon_ext}")))  # 设置窗口图标

        self.vbox = Gtk.Box()  # 创建 Gtk.Box 对象作为垂直容器

        # 根据 GTK 版本不同进行布局设置
        if gtk_ver == 3:
            self.window.add(self.vbox)  # 将 vbox 添加到窗口中
            self.vbox.show()  # 显示 vbox
            self.canvas.show()  # 显示 canvas
            self.vbox.pack_start(self.canvas, True, True, 0)  # 将 canvas 添加到 vbox 中
        elif gtk_ver == 4:
            self.window.set_child(self.vbox)  # 设置 Gtk4 中的子元素为 vbox
            self.vbox.prepend(self.canvas)  # 在 vbox 中前置添加 canvas

        # 计算窗口的大小
        w, h = self.canvas.get_width_height()

        if self.toolbar is not None:
            if gtk_ver == 3:
                self.toolbar.show()  # 显示工具栏
                self.vbox.pack_end(self.toolbar, False, False, 0)  # 将工具栏添加到 vbox 中
            elif gtk_ver == 4:
                sw = Gtk.ScrolledWindow(vscrollbar_policy=Gtk.PolicyType.NEVER)
                sw.set_child(self.toolbar)  # 设置 Gtk4 中的滚动窗口的子元素为工具栏
                self.vbox.append(sw)  # 将滚动窗口添加到 vbox 中
            min_size, nat_size = self.toolbar.get_preferred_size()  # 获取工具栏的推荐最小和自然大小
            h += nat_size.height  # 调整窗口高度以适应工具栏

        self.window.set_default_size(w, h)  # 设置窗口的默认大小

        self._destroying = False  # 初始化销毁标志为 False
        # 连接窗口的销毁事件和关闭请求事件，调用 Gcf.destroy(self)
        self.window.connect("destroy", lambda *args: Gcf.destroy(self))
        self.window.connect({3: "delete_event", 4: "close-request"}[gtk_ver],
                            lambda *args: Gcf.destroy(self))

        if mpl.is_interactive():
            self.window.show()  # 如果 matplotlib 交互模式开启，显示窗口
            self.canvas.draw_idle()  # 绘制 canvas

        self.canvas.grab_focus()  # 焦点设置到 canvas 上

    def destroy(self, *args):
        if self._destroying:
            # 避免重复调用 destroy 方法，当用户按下 'q' 键时会触发多次销毁操作
            return
        self._destroying = True
        self.window.destroy()  # 销毁窗口对象
        self.canvas.destroy()  # 销毁 canvas 对象

    @classmethod
    def start_main_loop(cls):
        global _application  # 声明使用全局变量 _application
        if _application is None:  # 如果 _application 为 None，则直接返回
            return

        try:
            _application.run()  # 调用 _application 的 run 方法，程序运行直到所有窗口关闭
        except KeyboardInterrupt:
            # 捕获键盘中断异常，确保所有窗口能够处理其关闭事件
            context = GLib.MainContext.default()
            while context.pending():
                context.iteration(True)
            raise  # 重新抛出 KeyboardInterrupt 异常
        finally:
            # 无论如何最终都将 _application 设置为 None
            _application = None

    def show(self):
        # 显示图形窗口
        self.window.show()
        self.canvas.draw()  # 绘制画布内容
        if mpl.rcParams["figure.raise_window"]:
            # 如果配置要求提升窗口
            meth_name = {3: "get_window", 4: "get_surface"}[self._gtk_ver]
            if getattr(self.window, meth_name)():
                self.window.present()  # 将窗口展示到前台
            else:
                # 如果在初始化早期由回调调用，可能 self.window (一个 GtkWindow) 还没有关联的低级 GdkWindow (在GTK3中) 或 GdkSurface (在GTK4中)
                # 调用 present() 可能会导致崩溃
                _api.warn_external("Cannot raise window yet to be setup")

    def full_screen_toggle(self):
        is_fullscreen = {
            3: lambda w: (w.get_window().get_state()
                          & Gdk.WindowState.FULLSCREEN),  # 获取窗口的全屏状态
            4: lambda w: w.is_fullscreen(),  # 获取窗口的全屏状态（GTK4）
        }[self._gtk_ver]
        if is_fullscreen(self.window):  # 如果窗口当前是全屏状态
            self.window.unfullscreen()  # 切换到非全屏状态
        else:
            self.window.fullscreen()  # 切换到全屏状态

    def get_window_title(self):
        return self.window.get_title()  # 返回窗口的标题

    def set_window_title(self, title):
        self.window.set_title(title)  # 设置窗口的标题为指定的 title

    def resize(self, width, height):
        width = int(width / self.canvas.device_pixel_ratio)  # 根据设备像素比例调整宽度
        height = int(height / self.canvas.device_pixel_ratio)  # 根据设备像素比例调整高度
        if self.toolbar:
            min_size, nat_size = self.toolbar.get_preferred_size()
            height += nat_size.height  # 考虑工具栏的高度

        canvas_size = self.canvas.get_allocation()  # 获取画布分配的大小
        if self._gtk_ver >= 4 or canvas_size.width == canvas_size.height == 1:
            # 如果是 GTK4 或者画布大小为 (1, 1)，表示窗口还未映射或小到无法存在
            # 这个调用必须在窗口映射和小部件大小之前，所以只能改变窗口的初始大小
            self.window.set_default_size(width, height)  # 设置窗口的默认大小
        else:
            self.window.resize(width, height)  # 改变窗口的大小
# 定义一个继承自 NavigationToolbar2 的 GTK 版本的工具栏类
class _NavigationToolbar2GTK(NavigationToolbar2):
    # 必须在 GTK3/GTK4 后端中实现的方法：
    # * __init__
    # * save_figure

    # 设置状态栏消息的显示内容，使用 GLib 对文本进行标记转义
    def set_message(self, s):
        escaped = GLib.markup_escape_text(s)
        self.message.set_markup(f'<small>{escaped}</small>')

    # 绘制橡皮筋效果，将事件和坐标转换为绘制区域坐标
    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas._draw_rubberband(rect)

    # 移除橡皮筋效果
    def remove_rubberband(self):
        self.canvas._draw_rubberband(None)

    # 更新工具栏按钮的状态，根据当前的模式选中相应按钮
    def _update_buttons_checked(self):
        for name, active in [("Pan", "PAN"), ("Zoom", "ZOOM")]:
            button = self._gtk_ids.get(name)
            if button:
                with button.handler_block(button._signal_handler):
                    button.set_active(self.mode.name == active)

    # 执行平移操作，并更新按钮状态
    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    # 执行缩放操作，并更新按钮状态
    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    # 设置历史记录按钮的状态，根据导航栈的当前位置判断是否可前进或后退
    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        if 'Back' in self._gtk_ids:
            self._gtk_ids['Back'].set_sensitive(can_backward)
        if 'Forward' in self._gtk_ids:
            self._gtk_ids['Forward'].set_sensitive(can_forward)


# 继承自 backend_tools.RubberbandBase 的 GTK 版本的橡皮筋工具类
class RubberbandGTK(backend_tools.RubberbandBase):
    # 绘制橡皮筋效果，调用 _NavigationToolbar2GTK 的绘制方法
    def draw_rubberband(self, x0, y0, x1, y1):
        _NavigationToolbar2GTK.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    # 移除橡皮筋效果，调用 _NavigationToolbar2GTK 的移除方法
    def remove_rubberband(self):
        _NavigationToolbar2GTK.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


# 继承自 backend_tools.ConfigureSubplotsBase 的 GTK 版本的子图配置类
class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase):
    # 触发配置子图操作，调用 _NavigationToolbar2GTK 的配置子图方法
    def trigger(self, *args):
        _NavigationToolbar2GTK.configure_subplots(self, None)


# 继承自 _Backend 的 GTK 版本的后端类
class _BackendGTK(_Backend):
    # 后端版本信息，格式化为 GTK 主版本、次版本、微版本的字符串
    backend_version = "{}.{}.{}".format(
        Gtk.get_major_version(),
        Gtk.get_minor_version(),
        Gtk.get_micro_version(),
    )
    # 设置主循环的启动方法为 _FigureManagerGTK 的 start_main_loop 方法
    mainloop = _FigureManagerGTK.start_main_loop
```