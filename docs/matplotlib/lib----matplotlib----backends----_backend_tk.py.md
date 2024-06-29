# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_backend_tk.py`

```
import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog

import numpy as np
from PIL import Image, ImageTk

import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase, ToolContainerBase, cursors, _Mode,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent)
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
from ._tkagg import TK_PHOTO_COMPOSITE_OVERLAY, TK_PHOTO_COMPOSITE_SET

# 设置日志记录器用于当前模块
_log = logging.getLogger(__name__)

# 定义指针样式映射字典
cursord = {
    cursors.MOVE: "fleur",
    cursors.HAND: "hand2",
    cursors.POINTER: "arrow",
    cursors.SELECT_REGION: "crosshair",
    cursors.WAIT: "watch",
    cursors.RESIZE_HORIZONTAL: "sb_h_double_arrow",
    cursors.RESIZE_VERTICAL: "sb_v_double_arrow",
}

@contextmanager
def _restore_foreground_window_at_end():
    """
    管理器函数，保存并在结束时恢复前台窗口的焦点状态。
    """
    foreground = _c_internal_utils.Win32_GetForegroundWindow()
    try:
        yield
    finally:
        if foreground and mpl.rcParams['tk.window_focus']:
            _c_internal_utils.Win32_SetForegroundWindow(foreground)

# 初始化用于传递参数的字典和 Tcl 命令名称
_blit_args = {}
_blit_tcl_name = "mpl_blit_" + uuid.uuid4().hex

def _blit(argsid):
    """
    通过 tkapp.call 调用的 blit 的轻量包装器函数。

    *argsid* 是一个唯一的字符串标识符，用于从 `_blit_args` 字典中获取正确的参数，
    因为不能直接传递参数。
    """
    # 从 _blit_args 字典中弹出参数并执行 blit 操作
    photoimage, data, offsets, bbox, comp_rule = _blit_args.pop(argsid)
    if not photoimage.tk.call("info", "commands", photoimage):
        return
    _tkagg.blit(photoimage.tk.interpaddr(), str(photoimage), data, comp_rule, offsets,
                bbox)

def blit(photoimage, aggimage, offsets, bbox=None):
    """
    将 *aggimage* 混合到 *photoimage* 中。

    *offsets* 是一个元组，描述如何填充 `Tk_PhotoImageBlock` 结构体的 `offset` 字段：
    对于 RGBA8888 数据应为 (0, 1, 2, 3)，
    对于小端 ARBG32 数据（例如 GBRA8888）应为 (2, 1, 0, 3)，
    对于大端 ARGB32 数据（例如 ARGB8888）应为 (1, 2, 3, 0)。

    如果传递了 *bbox*，则定义了要混合的区域。该区域将根据 alpha 通道与之前的数据混合。
    混合将被剪切到包含在画布内的像素，如果 *bbox* 区域完全在画布外部，则会静默地不执行任何操作。

    必须分派 Tcl 事件才能从非 Tcl 线程触发 blit 操作。
    """
    # 将 aggimage 转换为 numpy 数组
    data = np.asarray(aggimage)
    height, width = data.shape[:2]
    # 如果 bbox 参数不为 None，则从 bbox 中获取坐标信息并进行处理
    (x1, y1), (x2, y2) = bbox.__array__()
    # 对坐标进行下界取整和上界取整的处理，确保在图像边界内
    x1 = max(math.floor(x1), 0)
    x2 = min(math.ceil(x2), width)
    y1 = max(math.floor(y1), 0)
    y2 = min(math.ceil(y2), height)
    # 如果经过处理后，存在任意一个方向上的范围错误，则直接返回
    if (x1 > x2) or (y1 > y2):
        return
    # 将处理后的 bbox 坐标保存为 bboxptr
    bboxptr = (x1, x2, y1, y2)
    # 设置合成规则为 TK_PHOTO_COMPOSITE_OVERLAY

else:
    # 如果 bbox 参数为 None，则使用整个图像的范围
    bboxptr = (0, width, 0, height)
    # 设置合成规则为 TK_PHOTO_COMPOSITE_SET

# 注意：_tkagg.blit 在多线程环境下是线程不安全的，如果从非 Tcl 线程调用会导致进程崩溃（GH#13293）。
# 因此，避免在此处进行清空和绘制操作，而是使用 tkapp.call 来在需要时跨线程发布事件。
# 将参数打包到一个全局数据结构中，因为 tkapp.call 会将所有参数转换为字符串，以避免在 _blit 中进行字符串解析。
args = photoimage, data, offsets, bboxptr, comp_rule
# 为了避免线程竞争，需要一个唯一的键。
# 再次强调，将键设置为字符串，以避免在 _blit 中进行字符串解析。
argsid = str(id(args))
# 将参数保存到全局的 _blit_args 字典中
_blit_args[argsid] = args

try:
    # 尝试调用 photoimage 的 tk 对象的 _blit_tcl_name 方法，传递参数 argsid
    photoimage.tk.call(_blit_tcl_name, argsid)
except tk.TclError as e:
    # 如果捕获到 TclError 异常，并且异常信息中不包含 "invalid command name"，
    # 则重新抛出异常，否则继续处理。
    if "invalid command name" not in str(e):
        raise
    # 如果异常信息中包含 "invalid command name"，则说明 _blit_tcl_name 方法未定义，
    # 因此需要先通过 createcommand 方法定义 _blit_tcl_name 方法，然后再调用它。
    photoimage.tk.createcommand(_blit_tcl_name, _blit)
    # 重新调用 _blit_tcl_name 方法，传递参数 argsid
    photoimage.tk.call(_blit_tcl_name, argsid)
class TimerTk(TimerBase):
    """Subclass of `backend_bases.TimerBase` using Tk timer events."""

    def __init__(self, parent, *args, **kwargs):
        # 初始化 TimerTk 对象，继承 TimerBase 类的初始化方法
        self._timer = None
        super().__init__(*args, **kwargs)
        self.parent = parent

    def _timer_start(self):
        # 停止当前计时器
        self._timer_stop()
        # 使用 Tk 的 after 方法创建定时器事件，调用 self._on_timer 方法
        self._timer = self.parent.after(self._interval, self._on_timer)

    def _timer_stop(self):
        # 如果计时器存在，取消当前计时器事件
        if self._timer is not None:
            self.parent.after_cancel(self._timer)
        self._timer = None

    def _on_timer(self):
        # 调用父类 TimerBase 的 _on_timer 方法
        super()._on_timer()
        # 如果不是单次触发且计时器存在，则重新设置计时器
        if not self._single and self._timer:
            if self._interval > 0:
                # 如果间隔大于 0，继续创建计时器事件
                self._timer = self.parent.after(self._interval, self._on_timer)
            else:
                # 特殊情况：Tcl 的 after(0) 会在队列中插入事件，通过 after_idle 可以尽快运行
                self._timer = self.parent.after_idle(
                    lambda: self.parent.after(self._interval, self._on_timer)
                )
        else:
            # 否则将计时器设为 None
            self._timer = None


class FigureCanvasTk(FigureCanvasBase):
    required_interactive_framework = "tk"
    manager_class = _api.classproperty(lambda cls: FigureManagerTk)

    def _update_device_pixel_ratio(self, event=None):
        # 获取当前 Tk 的缩放比例，调整为 72 DPI 的比例
        ratio = round(self._tkcanvas.tk.call('tk', 'scaling') / (96 / 72), 2)
        # 如果设置设备像素比率成功
        if self._set_device_pixel_ratio(ratio):
            # 调整画布的宽度和高度以实现缩放
            w, h = self.get_width_height(physical=True)
            self._tkcanvas.configure(width=w, height=h)

    def resize(self, event):
        width, height = event.width, event.height

        # 计算期望的图形尺寸（单位为英寸）
        dpival = self.figure.dpi
        winch = width / dpival
        hinch = height / dpival
        # 设置图形的尺寸
        self.figure.set_size_inches(winch, hinch, forward=False)

        # 删除当前的 Tk 图像区域，并重新配置图像大小
        self._tkcanvas.delete(self._tkcanvas_image_region)
        self._tkphoto.configure(width=int(width), height=int(height))
        # 在画布中心创建新的图像区域
        self._tkcanvas_image_region = self._tkcanvas.create_image(
            int(width / 2), int(height / 2), image=self._tkphoto)
        # 发送 ResizeEvent 事件，处理后续逻辑
        ResizeEvent("resize_event", self)._process()
        # 刷新画布
        self.draw_idle()
    def draw_idle(self):
        """
        Schedule an idle draw of the figure.

        If an idle draw is already scheduled, do nothing.
        """
        # 如果已经有一个空闲绘制被调度，则直接返回
        if self._idle_draw_id:
            return

        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = None

        # 使用 after_idle 方法调度空闲绘制
        self._idle_draw_id = self._tkcanvas.after_idle(idle_draw)

    def get_tk_widget(self):
        """
        返回用于实现 FigureCanvasTkAgg 的 Tk 小部件。

        初始实现使用了一个 Tk 画布，但此方法旨在隐藏这一实现细节。
        """
        return self._tkcanvas

    def _event_mpl_coords(self, event):
        """
        计算事件在 Matplotlib 坐标系中的坐标。

        调用 canvasx/canvasy 方法以考虑滚动条的影响（即小部件的顶部可能滚出视图）。
        """
        return (self._tkcanvas.canvasx(event.x),
                # flipy 使得 y=0 对应画布底部
                self.figure.bbox.height - self._tkcanvas.canvasy(event.y))

    def motion_notify_event(self, event):
        """
        处理鼠标移动事件。

        创建 MouseEvent 对象并处理它。
        """
        MouseEvent("motion_notify_event", self,
                   *self._event_mpl_coords(event),
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()

    def enter_notify_event(self, event):
        """
        处理鼠标进入事件。

        创建 LocationEvent 对象并处理它。
        """
        LocationEvent("figure_enter_event", self,
                      *self._event_mpl_coords(event),
                      modifiers=self._mpl_modifiers(event),
                      guiEvent=event)._process()

    def leave_notify_event(self, event):
        """
        处理鼠标离开事件。

        创建 LocationEvent 对象并处理它。
        """
        LocationEvent("figure_leave_event", self,
                      *self._event_mpl_coords(event),
                      modifiers=self._mpl_modifiers(event),
                      guiEvent=event)._process()

    def button_press_event(self, event, dblclick=False):
        """
        处理鼠标按钮按下事件。

        设置焦点到画布以便接收键盘事件，创建 MouseEvent 对象并处理它。
        """
        self._tkcanvas.focus_set()

        num = getattr(event, 'num', None)
        if sys.platform == 'darwin':  # macOS 中的按钮 2 和 3 是颠倒的
            num = {2: 3, 3: 2}.get(num, num)
        MouseEvent("button_press_event", self,
                   *self._event_mpl_coords(event), num, dblclick=dblclick,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()

    def button_dblclick_event(self, event):
        """
        处理鼠标按钮双击事件。

        将双击事件转发给 button_press_event 方法处理。
        """
        self.button_press_event(event, dblclick=True)

    def button_release_event(self, event):
        """
        处理鼠标按钮释放事件。

        创建 MouseEvent 对象并处理它。
        """
        num = getattr(event, 'num', None)
        if sys.platform == 'darwin':  # macOS 中的按钮 2 和 3 是颠倒的
            num = {2: 3, 3: 2}.get(num, num)
        MouseEvent("button_release_event", self,
                   *self._event_mpl_coords(event), num,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()
    # 处理鼠标滚轮事件的方法
    def scroll_event(self, event):
        # 获取事件中的滚轮数值（num），如果不存在则设为 None
        num = getattr(event, 'num', None)
        # 根据滚轮数值确定滚动方向，num == 4 表示向上滚动，num == 5 表示向下滚动，其他情况为不滚动
        step = 1 if num == 4 else -1 if num == 5 else 0
        # 创建 MouseEvent 实例并处理事件
        MouseEvent("scroll_event", self,
                   *self._event_mpl_coords(event), step=step,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()

    # 处理 Windows 系统下的鼠标滚轮事件的方法
    def scroll_event_windows(self, event):
        """MouseWheel event processor"""
        # 获取包含鼠标的窗口对象 w
        w = event.widget.winfo_containing(event.x_root, event.y_root)
        # 如果窗口对象不是 self._tkcanvas，则返回
        if w != self._tkcanvas:
            return
        # 计算鼠标相对于 self._tkcanvas 的 x 和 y 坐标
        x = self._tkcanvas.canvasx(event.x_root - w.winfo_rootx())
        y = (self.figure.bbox.height
             - self._tkcanvas.canvasy(event.y_root - w.winfo_rooty()))
        # 计算滚动步长
        step = event.delta / 120
        # 创建 MouseEvent 实例并处理事件
        MouseEvent("scroll_event", self,
                   x, y, step=step, modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()

    @staticmethod
    def _mpl_modifiers(event, *, exclude=None):
        # 添加修饰键到键盘按键字符串中
        # 详细信息来源于 http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm
        # 根据不同平台设置不同的修饰键列表
        modifiers = [
            ("ctrl", 1 << 2, "control"),
            ("alt", 1 << 17, "alt"),
            ("shift", 1 << 0, "shift"),
        ] if sys.platform == "win32" else [
            ("ctrl", 1 << 2, "control"),
            ("alt", 1 << 4, "alt"),
            ("shift", 1 << 0, "shift"),
            ("cmd", 1 << 3, "cmd"),
        ] if sys.platform == "darwin" else [
            ("ctrl", 1 << 2, "control"),
            ("alt", 1 << 3, "alt"),
            ("shift", 1 << 0, "shift"),
            ("super", 1 << 6, "super"),
        ]
        # 返回当前事件中存在的修饰键列表（exclude 参数用于排除特定修饰键）
        return [name for name, mask, key in modifiers
                if event.state & mask and exclude != key]

    # 获取键盘事件的键值
    def _get_key(self, event):
        # 获取事件中的 Unicode 键值
        unikey = event.char
        # 将 Unicode 键值转换为 Matplotlib 的键值，并排除已有的修饰键
        key = cbook._unikey_or_keysym_to_mplkey(unikey, event.keysym)
        if key is not None:
            # 获取事件中的修饰键列表，排除已经计算过的 shift 修饰键
            mods = self._mpl_modifiers(event, exclude=key)
            if "shift" in mods and unikey:
                mods.remove("shift")
            # 返回修饰键和键值的组合字符串
            return "+".join([*mods, key])

    # 处理键盘按下事件的方法
    def key_press(self, event):
        # 创建 KeyEvent 实例并处理事件
        KeyEvent("key_press_event", self,
                 self._get_key(event), *self._event_mpl_coords(event),
                 guiEvent=event)._process()
    # 处理键盘释放事件的方法，创建一个 KeyEvent 实例并处理
    def key_release(self, event):
        KeyEvent("key_release_event", self,
                 self._get_key(event), *self._event_mpl_coords(event),
                 guiEvent=event)._process()

    # 创建并返回一个新的 TimerTk 实例，继承自父类的文档字符串
    def new_timer(self, *args, **kwargs):
        return TimerTk(self._tkcanvas, *args, **kwargs)

    # 刷新 Tkinter 窗口以处理事件，继承自父类的文档字符串
    def flush_events(self):
        self._tkcanvas.update()

    # 开始事件循环，等待事件发生或达到超时时间，继承自父类的文档字符串
    def start_event_loop(self, timeout=0):
        if timeout > 0:
            milliseconds = int(1000 * timeout)
            if milliseconds > 0:
                # 注册一个在指定毫秒后调用 self.stop_event_loop 的事件
                self._event_loop_id = self._tkcanvas.after(
                    milliseconds, self.stop_event_loop)
            else:
                # 如果超时时间小于等于 0，使用 after_idle 注册事件
                self._event_loop_id = self._tkcanvas.after_idle(
                    self.stop_event_loop)
        # 开始 Tkinter 的主事件循环
        self._tkcanvas.mainloop()

    # 停止事件循环，如果有活动的事件循环标识，则取消之前注册的事件并退出 Tkinter 主循环
    def stop_event_loop(self):
        if self._event_loop_id:
            self._tkcanvas.after_cancel(self._event_loop_id)
            self._event_loop_id = None
        self._tkcanvas.quit()

    # 设置鼠标光标的方法，根据给定的光标名称设置光标样式，如果名称无效则忽略异常
    def set_cursor(self, cursor):
        try:
            self._tkcanvas.configure(cursor=cursord[cursor])
        except tkinter.TclError:
            pass
    @classmethod
    # 创建一个带有指定画布类、图形和编号的 FigureManagerTk 实例
    def create_with_canvas(cls, canvas_class, figure, num):
        # docstring inherited

        # 在恢复前景窗口的上下文中执行以下操作
        with _restore_foreground_window_at_end():
            # 如果没有正在运行的交互式框架，则设置新的 GUI 应用程序
            if cbook._get_running_interactive_framework() is None:
                cbook._setup_new_guiapp()
                # 设置当前进程的 DPI 感知性为最大
                _c_internal_utils.Win32_SetProcessDpiAwareness_max()

            # 创建一个名为 "matplotlib" 的 Tkinter 窗口
            window = tk.Tk(className="matplotlib")
            # 隐藏窗口，暂时不显示
            window.withdraw()

            # 设置窗口的图标为 Matplotlib 图标，而不是默认的 Tk 图标
            # 详见 https://www.tcl.tk/man/tcl/TkCmd/wm.html#M50
            #
            # 当最低支持的 Tk 版本提升到 8.6 时，可以将 `ImageTk` 替换为 `tk`，
            # 因为 Tk 8.6+ 原生支持 PNG 图像。
            icon_fname = str(cbook._get_data_path('images/matplotlib.png'))
            icon_img = ImageTk.PhotoImage(file=icon_fname, master=window)

            icon_fname_large = str(cbook._get_data_path('images/matplotlib_large.png'))
            icon_img_large = ImageTk.PhotoImage(file=icon_fname_large, master=window)

            # 设置窗口的图标
            window.iconphoto(False, icon_img_large, icon_img)

            # 使用给定的图形和窗口创建一个指定类的画布实例
            canvas = canvas_class(figure, master=window)

            # 使用 canvas、num 和 window 创建 FigureManagerTk 的实例
            manager = cls(canvas, num, window)

            # 如果 Matplotlib 当前处于交互模式，则显示管理器和画布
            if mpl.is_interactive():
                manager.show()
                canvas.draw_idle()

            # 返回创建的 FigureManagerTk 实例
            return manager
    # 启动主事件循环，确保只有一个管理器拥有主事件循环
    def start_main_loop(cls):
        # 获取所有图形管理器
        managers = Gcf.get_all_fig_managers()
        if managers:
            # 获取第一个图形管理器
            first_manager = managers[0]
            # 获取管理器的类类型
            manager_class = type(first_manager)
            # 如果管理器类拥有主事件循环，则直接返回
            if manager_class._owns_mainloop:
                return
            # 设置管理器类拥有主事件循环的标志
            manager_class._owns_mainloop = True
            try:
                # 运行第一个管理器的窗口的主事件循环
                first_manager.window.mainloop()
            finally:
                # 确保无论如何都将管理器类的主事件循环标志复位
                manager_class._owns_mainloop = False

    # 更新窗口 DPI 设置
    def _update_window_dpi(self, *args):
        # 获取新的 DPI 值
        newdpi = self._window_dpi.get()
        # 调用 Tkinter 的 scaling 方法来更新 DPI 设置
        self.window.call('tk', 'scaling', newdpi / 72)
        # 如果存在工具栏且具有 _rescale 方法，则调用工具栏的 _rescale 方法
        if self.toolbar and hasattr(self.toolbar, '_rescale'):
            self.toolbar._rescale()
        # 更新画布的设备像素比率
        self.canvas._update_device_pixel_ratio()

    # 调整窗口大小
    def resize(self, width, height):
        # 最大窗口尺寸限制
        max_size = 1_400_000  # 在 xorg 1.20.8 上测得的最大尺寸为 1_409_023

        # 如果宽度或高度超过最大尺寸，并且系统平台为 Linux，则抛出 ValueError
        if (width > max_size or height > max_size) and sys.platform == 'linux':
            raise ValueError(
                'You have requested to resize the '
                f'Tk window to ({width}, {height}), one of which '
                f'is bigger than {max_size}.  At larger sizes xorg will '
                'either exit with an error on newer versions (~1.20) or '
                'cause corruption on older version (~1.19).  We '
                'do not expect a window over a million pixel wide or tall '
                'to be intended behavior.')
        # 更新画布的 Tkinter Canvas 配置，设置新的宽度和高度
        self.canvas._tkcanvas.configure(width=width, height=height)

    # 显示窗口
    def show(self):
        # 在结束时恢复前景窗口
        with _restore_foreground_window_at_end():
            # 如果窗口尚未显示
            if not self._shown:
                # 定义销毁窗口的函数
                def destroy(*args):
                    Gcf.destroy(self)
                # 设置窗口的 WM_DELETE_WINDOW 协议，使得窗口关闭时执行 destroy 函数
                self.window.protocol("WM_DELETE_WINDOW", destroy)
                # 显示窗口
                self.window.deiconify()
                # 设置焦点到画布的 Tkinter Canvas
                self.canvas._tkcanvas.focus_set()
            else:
                # 如果窗口已经显示，则空闲绘制画布
                self.canvas.draw_idle()
            # 如果 mpl.rcParams['figure.raise_window'] 为 True，则将窗口置顶
            if mpl.rcParams['figure.raise_window']:
                self.canvas.manager.window.attributes('-topmost', 1)
                self.canvas.manager.window.attributes('-topmost', 0)
            # 更新窗口显示状态为已显示
            self._shown = True
    # 销毁窗口对象及其相关资源
    def destroy(self, *args):
        # 如果存在空闲绘制标识，取消相应的绘制任务
        if self.canvas._idle_draw_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._idle_draw_id)
        # 如果存在事件循环标识，取消相应的事件循环任务
        if self.canvas._event_loop_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._event_loop_id)
        # 如果存在 DPI 变化的回调名称，移除对 DPI 变化的监听
        if self._window_dpi_cbname:
            self._window_dpi.trace_remove('write', self._window_dpi_cbname)

        # 注意：在销毁窗口之前需要刷新事件队列（GH #9956），
        # 但是 self.window.update() 可能会影响用户代码。异步回调是
        # 最安全的方式来完全处理事件队列，但是如果没有 tk 事件循环在运行，
        # 它会造成内存泄漏。因此，我们显式检查事件循环并选择最合适的方式。
        def delayed_destroy():
            self.window.destroy()

            # 如果程序拥有主事件循环并且没有图形管理器在运行，退出主事件循环
            if self._owns_mainloop and not Gcf.get_num_fig_managers():
                self.window.quit()

        # 如果当前使用的交互框架是 tk
        if cbook._get_running_interactive_framework() == "tk":
            # "after idle after 0" 避免 Tcl 错误/竞态条件 (GH #19940)
            self.window.after_idle(self.window.after, 0, delayed_destroy)
        else:
            # 刷新窗口状态
            self.window.update()
            # 立即销毁窗口
            delayed_destroy()

    # 获取窗口标题
    def get_window_title(self):
        return self.window.wm_title()

    # 设置窗口标题
    def set_window_title(self, title):
        self.window.wm_title(title)

    # 切换全屏模式
    def full_screen_toggle(self):
        # 获取当前窗口是否处于全屏模式
        is_fullscreen = bool(self.window.attributes('-fullscreen'))
        # 切换窗口的全屏属性
        self.window.attributes('-fullscreen', not is_fullscreen)
class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):
    # 继承自 NavigationToolbar2 和 tk.Frame 的自定义工具栏类
    def __init__(self, canvas, window=None, *, pack_toolbar=True):
        """
        Parameters
        ----------
        canvas : `FigureCanvas`
            操作的图表画布对象.
        window : tk.Window
            拥有此工具栏的 tk 窗口对象.
        pack_toolbar : bool, default: True
            如果为 True，在初始化时将工具栏添加到父窗口的 pack 管理器的 packing 列表中，
            使用 `side="bottom"` 和 `fill="x"`. 如果要使用不同的布局管理器，请使用
            `pack_toolbar=False`.
        """

        if window is None:
            window = canvas.get_tk_widget().master
        # 调用父类的初始化方法，创建一个高度为 50，宽度为图表宽度的 Frame
        tk.Frame.__init__(self, master=window, borderwidth=2,
                          width=int(canvas.figure.bbox.width), height=50)

        self._buttons = {}
        # 遍历工具栏中的各个工具项，创建对应的按钮或间隔
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # 添加一个占位符间隔；返回值未使用
                self._Spacer()
            else:
                # 创建一个按钮，根据工具项的参数设置按钮的文本、图像、toggle 状态和回调函数
                self._buttons[text] = button = self._Button(
                    text,
                    str(cbook._get_data_path(f"images/{image_file}.png")),
                    toggle=callback in ["zoom", "pan"],
                    command=getattr(self, callback),
                )
                if tooltip_text is not None:
                    # 如果有提示文本，为按钮添加提示
                    add_tooltip(button, tooltip_text)

        self._label_font = tkinter.font.Font(root=window, size=10)

        # 这个填充项确保工具栏至少有两行文本高度。否则，当鼠标悬停在图像上时，
        # 由于图像使用两行消息，会重新绘制工具栏。
        label = tk.Label(master=self, font=self._label_font,
                         text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}')
        label.pack(side=tk.RIGHT)

        self.message = tk.StringVar(master=self)
        # 创建一个消息标签，显示工具栏的状态消息，右对齐
        self._message_label = tk.Label(master=self, font=self._label_font,
                                       textvariable=self.message,
                                       justify=tk.RIGHT)
        self._message_label.pack(side=tk.RIGHT)

        # 调用父类 NavigationToolbar2 的初始化方法，设置工具栏的 canvas
        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            # 如果需要，将工具栏添加到底部并填充横向
            self.pack(side=tk.BOTTOM, fill=tk.X)
    def _rescale(self):
        """
        Scale all children of the toolbar to current DPI setting.

        Before this is called, the Tk scaling setting will have been updated to
        match the new DPI. Tk widgets do not update for changes to scaling, but
        all measurements made after the change will match the new scaling. Thus
        this function re-applies all the same sizes in points, which Tk will
        scale correctly to pixels.
        """
        # 遍历工具栏的所有子部件
        for widget in self.winfo_children():
            # 如果部件是按钮或复选框
            if isinstance(widget, (tk.Button, tk.Checkbutton)):
                # 如果部件有 '_image_file' 属性，更新按钮的图像
                if hasattr(widget, '_image_file'):
                    # 明确指定类，因为 ToolbarTk 调用了 _rescale。
                    NavigationToolbar2Tk._set_image_for_button(self, widget)
                else:
                    # 只有文本的按钮由字体设置处理
                    pass
            # 如果部件是框架
            elif isinstance(widget, tk.Frame):
                # 设置框架的高度为 '18p'
                widget.configure(height='18p')
                # 调整框架的填充为 '3p'
                widget.pack_configure(padx='3p')
            # 如果部件是标签
            elif isinstance(widget, tk.Label):
                # 文本由字体设置处理，无需额外操作
                pass
            # 如果部件类型未知
            else:
                # 记录警告，指出未知的子类
                _log.warning('Unknown child class %s', widget.winfo_class)
        # 设置标签字体的大小为 10
        self._label_font.configure(size=10)

    def _update_buttons_checked(self):
        # 同步按钮的选中状态以匹配活动模式
        for text, mode in [('Zoom', _Mode.ZOOM), ('Pan', _Mode.PAN)]:
            # 如果按钮存在于按钮字典中
            if text in self._buttons:
                # 如果当前模式与按钮对应模式相同，选中按钮
                if self.mode == mode:
                    self._buttons[text].select()  # NOT .invoke()
                else:
                    # 否则取消选中按钮
                    self._buttons[text].deselect()

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
        # 设置消息变量的值
        self.message.set(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        # 从 remove_rubberband 复制的代码，方便后端工具使用。
        # 如果存在白色橡皮筋矩形，删除它
        if self.canvas._rubberband_rect_white:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_white)
        # 如果存在黑色橡皮筋矩形，删除它
        if self.canvas._rubberband_rect_black:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_black)
        # 获取画布的高度
        height = self.canvas.figure.bbox.height
        # 调整 y0 和 y1 以匹配画布的高度
        y0 = height - y0
        y1 = height - y1
        # 创建黑色橡皮筋矩形
        self.canvas._rubberband_rect_black = (
            self.canvas._tkcanvas.create_rectangle(
                x0, y0, x1, y1))
        # 创建白色橡皮筋矩形
        self.canvas._rubberband_rect_white = (
            self.canvas._tkcanvas.create_rectangle(
                x0, y0, x1, y1, outline='white', dash=(3, 3)))
    # 移除白色橡皮筋矩形对象，如果存在的话
    def remove_rubberband(self):
        if self.canvas._rubberband_rect_white:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_white)
            self.canvas._rubberband_rect_white = None
        # 移除黑色橡皮筋矩形对象，如果存在的话
        if self.canvas._rubberband_rect_black:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_black)
            self.canvas._rubberband_rect_black = None

    # 创建按钮控件
    def _Button(self, text, image_file, toggle, command):
        # 如果不是切换按钮，创建普通按钮
        if not toggle:
            b = tk.Button(
                master=self, text=text, command=command,
                relief="flat", overrelief="groove", borderwidth=1,
            )
        else:
            # 解决 tkinter 在某些 Python 3.6 版本中的 bug，
            # 缺少这个变量可能导致其他附近的复选按钮的视觉切换问题
            # 参考：https://bugs.python.org/issue29402
            # 参考：https://bugs.python.org/issue25684
            var = tk.IntVar(master=self)
            b = tk.Checkbutton(
                master=self, text=text, command=command, indicatoron=False,
                variable=var, offrelief="flat", overrelief="groove",
                borderwidth=1
            )
            b.var = var
        # 设置按钮关联的图像文件属性
        b._image_file = image_file
        # 如果有图像文件，则调用特定类方法设置按钮的图像
        if image_file is not None:
            NavigationToolbar2Tk._set_image_for_button(self, b)
        else:
            # 否则配置按钮的字体
            b.configure(font=self._label_font)
        # 将按钮放置在左侧
        b.pack(side=tk.LEFT)
        return b

    # 创建占位框架
    def _Spacer(self):
        # 占位框架高度为 18 点
        s = tk.Frame(master=self, height='18p', relief=tk.RIDGE, bg='DarkGray')
        # 将占位框架放置在左侧，带有一定的水平间距
        s.pack(side=tk.LEFT, padx='3p')
        return s
    def save_figure(self, *args):
        # 获取当前画布支持的文件类型
        filetypes = self.canvas.get_supported_filetypes_grouped()
        # 将文件类型转换为适合 tkinter 的格式
        tk_filetypes = [
            (name, " ".join(f"*.{ext}" for ext in exts))
            for name, exts in sorted(filetypes.items())
        ]

        # 获取默认的文件扩展名
        default_extension = self.canvas.get_default_filetype()
        # 获取默认文件类型的描述
        default_filetype = self.canvas.get_supported_filetypes()[default_extension]
        # 创建 tkinter 的字符串变量，并设置默认文件类型
        filetype_variable = tk.StringVar(self.canvas.get_tk_widget(), default_filetype)

        # 添加默认扩展名似乎会破坏 asksaveasfilename 对话框，当从下拉列表中选择不同的保存类型时
        # 传递空字符串似乎有效 - JDH!
        # defaultextension = self.canvas.get_default_filetype()
        defaultextension = ''

        # 获取保存图像的初始目录
        initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])

        # 获取默认文件名，移除可能的默认扩展名，以使下拉列表功能正常
        initialfile = pathlib.Path(self.canvas.get_default_filename()).stem

        # 调用 tkinter 的文件保存对话框，获取用户选择的文件名
        fname = tkinter.filedialog.asksaveasfilename(
            master=self.canvas.get_tk_widget().master,
            title='Save the figure',
            filetypes=tk_filetypes,
            defaultextension=defaultextension,
            initialdir=initialdir,
            initialfile=initialfile,
            typevariable=filetype_variable
            )

        # 如果用户取消保存或未选择文件名，则直接返回
        if fname in ["", ()]:
            return

        # 如果初始目录不为空，则更新保存图像的默认目录为用户选择的目录
        if initialdir != "":
            mpl.rcParams['savefig.directory'] = (
                os.path.dirname(str(fname)))

        # 根据用户选择的文件名是否包含扩展名来决定保存文件的格式
        if pathlib.Path(fname).suffix[1:] != "":
            extension = None
        else:
            extension = filetypes[filetype_variable.get()][0]

        try:
            # 调用画布的保存方法将图像保存为指定格式的文件
            self.canvas.figure.savefig(fname, format=extension)
        except Exception as e:
            # 如果保存出错，显示错误对话框
            tkinter.messagebox.showerror("Error saving file", str(e))

    def set_history_buttons(self):
        # 定义按钮状态映射关系
        state_map = {True: tk.NORMAL, False: tk.DISABLED}
        # 检查是否可以向后导航或向前导航
        can_back = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1

        # 如果存在"Back"按钮，则根据 can_back 设置按钮状态
        if "Back" in self._buttons:
            self._buttons['Back']['state'] = state_map[can_back]
        
        # 如果存在"Forward"按钮，则根据 can_forward 设置按钮状态
        if "Forward" in self._buttons:
            self._buttons['Forward']['state'] = state_map[can_forward]
# 定义一个函数用于为给定的窗口部件添加工具提示
def add_tooltip(widget, text):
    # 初始化工具提示窗口
    tipwindow = None

    # 定义函数，在鼠标进入部件时显示工具提示
    def showtip(event):
        """Display text in tooltip window."""
        nonlocal tipwindow
        # 如果已经存在工具提示窗口或者文本为空，则直接返回
        if tipwindow or not text:
            return
        # 获取部件中插入点的边界框坐标
        x, y, _, _ = widget.bbox("insert")
        # 计算工具提示窗口的位置，使其位于部件右侧
        x = x + widget.winfo_rootx() + widget.winfo_width()
        y = y + widget.winfo_rooty()
        # 创建一个顶级窗口作为工具提示窗口
        tipwindow = tk.Toplevel(widget)
        # 设置工具提示窗口为无边框
        tipwindow.overrideredirect(1)
        # 设置工具提示窗口的位置
        tipwindow.geometry(f"+{x}+{y}")
        try:  # 用于 Mac OS 的特殊处理
            tipwindow.tk.call("::tk::unsupported::MacWindowStyle",
                              "style", tipwindow._w,
                              "help", "noActivates")
        except tk.TclError:
            pass
        # 在工具提示窗口中创建一个标签，显示指定的文本
        label = tk.Label(tipwindow, text=text, justify=tk.LEFT,
                         relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=1)

    # 定义函数，在鼠标离开部件时隐藏工具提示
    def hidetip(event):
        nonlocal tipwindow
        # 如果工具提示窗口存在，则销毁它
        if tipwindow:
            tipwindow.destroy()
        tipwindow = None

    # 绑定鼠标进入事件，使其触发显示工具提示
    widget.bind("<Enter>", showtip)
    # 绑定鼠标离开事件，使其触发隐藏工具提示
    widget.bind("<Leave>", hidetip)


# 为 FigureCanvasTk 类注册一个工具类，继承自 backend_tools.RubberbandBase
@backend_tools._register_tool_class(FigureCanvasTk)
class RubberbandTk(backend_tools.RubberbandBase):
    # 实现绘制橡皮筋的方法
    def draw_rubberband(self, x0, y0, x1, y1):
        NavigationToolbar2Tk.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    # 实现移除橡皮筋的方法
    def remove_rubberband(self):
        NavigationToolbar2Tk.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


# 定义 ToolbarTk 类，继承自 ToolContainerBase 和 tk.Frame
class ToolbarTk(ToolContainerBase, tk.Frame):
    # 初始化方法
    def __init__(self, toolmanager, window=None):
        # 调用父类 ToolContainerBase 的初始化方法
        ToolContainerBase.__init__(self, toolmanager)
        # 如果未指定窗口，则使用工具管理器中画布的主窗口
        if window is None:
            window = self.toolmanager.canvas.get_tk_widget().master
        # 获取画布的 X 轴边界范围
        xmin, xmax = self.toolmanager.canvas.figure.bbox.intervalx
        # 设置工具栏的高度和宽度
        height, width = 50, xmax - xmin
        # 调用 tk.Frame 的初始化方法，创建一个框架作为工具栏
        tk.Frame.__init__(self, master=window,
                          width=int(width), height=int(height),
                          borderwidth=2)
        # 设置工具栏标签的字体大小
        self._label_font = tkinter.font.Font(size=10)
        # 创建一个填充标签，确保工具栏至少有两行文本高度，避免因鼠标悬停在图像上而重新绘制画布
        label = tk.Label(master=self, font=self._label_font,
                         text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}')
        label.pack(side=tk.RIGHT)
        # 创建一个字符串变量，用于显示工具栏的消息
        self._message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, font=self._label_font,
                                       textvariable=self._message)
        self._message_label.pack(side=tk.RIGHT)
        self._toolitems = {}  # 工具项字典，存储工具栏中的工具项
        self.pack(side=tk.TOP, fill=tk.X)
        self._groups = {}  # 工具栏分组字典，存储工具栏中的分组信息

    # 重新缩放工具栏的方法
    def _rescale(self):
        return NavigationToolbar2Tk._rescale(self)
    # 向工具栏添加一个工具项
    def add_toolitem(
            self, name, group, position, image_file, description, toggle):
        # 获取指定组的框架，用于放置工具按钮
        frame = self._get_groupframe(group)
        # 获取当前框架中的所有按钮
        buttons = frame.pack_slaves()
        # 确定按钮插入的位置
        if position >= len(buttons) or position < 0:
            before = None  # 如果位置超出范围，则插入到末尾
        else:
            before = buttons[position]  # 否则插入到指定位置的按钮之前
        # 创建一个工具按钮，并将其放置到框架中
        button = NavigationToolbar2Tk._Button(frame, name, image_file, toggle,
                                              lambda: self._button_click(name))
        button.pack_configure(before=before)
        # 如果有描述信息，则添加工具提示
        if description is not None:
            add_tooltip(button, description)
        # 将按钮添加到工具项字典中的对应名称列表中
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append(button)

    # 获取指定组的框架，如果不存在则创建新框架
    def _get_groupframe(self, group):
        if group not in self._groups:
            # 如果组不存在且已有其他组存在，则添加分隔符
            if self._groups:
                self._add_separator()
            # 创建新的框架，并设置其属性
            frame = tk.Frame(master=self, borderwidth=0)
            frame.pack(side=tk.LEFT, fill=tk.Y)
            frame._label_font = self._label_font
            self._groups[group] = frame
        return self._groups[group]

    # 添加一个分隔符到工具栏
    def _add_separator(self):
        return NavigationToolbar2Tk._Spacer(self)

    # 处理工具按钮的点击事件
    def _button_click(self, name):
        self.trigger_tool(name)

    # 切换指定工具项的状态（选中或未选中）
    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        # 遍历指定名称的所有工具项，并设置选中或未选中状态
        for toolitem in self._toolitems[name]:
            if toggled:
                toolitem.select()
            else:
                toolitem.deselect()

    # 移除指定名称的所有工具项
    def remove_toolitem(self, name):
        # 从工具项字典中移除指定名称，并将其对应的按钮从界面中移除
        for toolitem in self._toolitems.pop(name, []):
            toolitem.pack_forget()

    # 设置状态栏的消息文本
    def set_message(self, s):
        self._message.set(s)
# 将 FigureCanvasTk 类注册为 backend_tools 模块的工具类
@backend_tools._register_tool_class(FigureCanvasTk)
# 继承自 backend_tools.SaveFigureBase，用于保存图形
class SaveFigureTk(backend_tools.SaveFigureBase):
    # 触发保存操作的方法，调用 NavigationToolbar2Tk 的 save_figure 方法
    def trigger(self, *args):
        # 创建经典风格的伪工具栏并保存图形
        NavigationToolbar2Tk.save_figure(
            self._make_classic_style_pseudo_toolbar())


# 将 FigureCanvasTk 类注册为 backend_tools 模块的工具类
@backend_tools._register_tool_class(FigureCanvasTk)
# 继承自 backend_tools.ConfigureSubplotsBase，用于配置子图
class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):
    # 触发配置子图操作的方法，调用 NavigationToolbar2Tk 的 configure_subplots 方法
    def trigger(self, *args):
        # 配置图形的子图
        NavigationToolbar2Tk.configure_subplots(self)


# 将 FigureCanvasTk 类注册为 backend_tools 模块的工具类
@backend_tools._register_tool_class(FigureCanvasTk)
# 继承自 backend_tools.ToolHelpBase，用于显示帮助信息
class HelpTk(backend_tools.ToolHelpBase):
    # 触发显示帮助信息的方法
    def trigger(self, *args):
        # 创建一个简单对话框，显示帮助文本，包含一个"OK"按钮
        dialog = SimpleDialog(
            self.figure.canvas._tkcanvas, self._get_help_text(), ["OK"])
        # 定义 done 方法作为 lambda 函数，用于隐藏对话框窗口
        dialog.done = lambda num: dialog.frame.master.withdraw()


# 将 ToolbarTk 类设置为 Toolbar
Toolbar = ToolbarTk
# 将 NavigationToolbar2Tk 设置为 FigureManagerTk 的工具栏类
FigureManagerTk._toolbar2_class = NavigationToolbar2Tk
# 将 ToolbarTk 设置为 FigureManagerTk 的工具管理器工具栏类
FigureManagerTk._toolmanager_toolbar_class = ToolbarTk


# 将 _BackendTk 类导出为 _Backend 模块的组件
@_Backend.export
# 继承自 _Backend，表示 Tkinter 后端
class _BackendTk(_Backend):
    # 设置后端版本为 tk.TkVersion
    backend_version = tk.TkVersion
    # 设置图形画布类为 FigureCanvasTk
    FigureCanvas = FigureCanvasTk
    # 设置图形管理器类为 FigureManagerTk
    FigureManager = FigureManagerTk
    # 设置主循环方法为 FigureManagerTk 的 start_main_loop 方法
    mainloop = FigureManagerTk.start_main_loop
```