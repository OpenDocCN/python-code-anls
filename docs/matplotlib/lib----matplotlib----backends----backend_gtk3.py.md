# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk3.py`

```py
import functools  # 导入 functools 模块，用于支持函数式编程的工具
import logging  # 导入 logging 模块，用于记录日志信息
import os  # 导入 os 模块，提供与操作系统交互的功能
from pathlib import Path  # 导入 Path 类，用于处理文件路径的对象

import matplotlib as mpl  # 导入 matplotlib 库，并将其命名为 mpl
from matplotlib import _api, backend_tools, cbook  # 从 matplotlib 中导入特定模块和对象
from matplotlib.backend_bases import (  # 从 matplotlib 的 backend_bases 模块中导入多个类
    ToolContainerBase, CloseEvent, KeyEvent, LocationEvent, MouseEvent,
    ResizeEvent)

try:
    import gi  # 尝试导入 gi 模块，用于 GTK3 后端
except ImportError as err:
    raise ImportError("The GTK3 backends require PyGObject") from err

try:
    gi.require_version("Gtk", "3.0")  # 确保 GTK 版本为 3.0
except ValueError as e:
    # 如果版本错误，则转换为 ImportError 异常以便正确跳过自动后端选择逻辑
    raise ImportError(e) from e

from gi.repository import Gio, GLib, GObject, Gtk, Gdk  # 从 gi.repository 中导入多个 GTK 相关的模块
from . import _backend_gtk  # 导入当前包下的 _backend_gtk 模块
from ._backend_gtk import (  # 从 _backend_gtk 模块中导入多个类和对象
    _BackendGTK, _FigureCanvasGTK, _FigureManagerGTK, _NavigationToolbar2GTK,
    TimerGTK as TimerGTK3,
)

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@functools.cache
def _mpl_to_gtk_cursor(mpl_cursor):
    """
    使用 functools.cache 装饰器缓存函数结果，将 matplotlib 的光标名映射为 GTK 中的光标对象。

    Args:
        mpl_cursor (str): matplotlib 中的光标名称。

    Returns:
        Gdk.Cursor: GTK 中对应的光标对象。
    """
    return Gdk.Cursor.new_from_name(
        Gdk.Display.get_default(),
        _backend_gtk.mpl_to_gtk_cursor_name(mpl_cursor))


class FigureCanvasGTK3(_FigureCanvasGTK, Gtk.DrawingArea):
    """
    继承自 _FigureCanvasGTK 和 Gtk.DrawingArea，实现 GTK3 版本的图形画布。

    Attributes:
        required_interactive_framework (str): 所需的交互式框架为 "gtk3"。
        manager_class (classproperty): 管理器类属性，指定为 FigureManagerGTK3。
        event_mask (Gdk.EventMask): 事件掩码，包括各种 GDK 事件类型的掩码。
    """

    required_interactive_framework = "gtk3"
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK3)
    event_mask = (Gdk.EventMask.BUTTON_PRESS_MASK
                  | Gdk.EventMask.BUTTON_RELEASE_MASK
                  | Gdk.EventMask.EXPOSURE_MASK
                  | Gdk.EventMask.KEY_PRESS_MASK
                  | Gdk.EventMask.KEY_RELEASE_MASK
                  | Gdk.EventMask.ENTER_NOTIFY_MASK
                  | Gdk.EventMask.LEAVE_NOTIFY_MASK
                  | Gdk.EventMask.POINTER_MOTION_MASK
                  | Gdk.EventMask.SCROLL_MASK)
    # 初始化方法，可选参数 figure 用于初始化父类
    def __init__(self, figure=None):
        # 调用父类的初始化方法
        super().__init__(figure=figure)

        # 初始化私有属性 _idle_draw_id，用于存储闲置绘制的标识符
        self._idle_draw_id = 0
        # 初始化 _rubberband_rect，用于存储橡皮筋矩形的位置信息，初始为 None
        self._rubberband_rect = None

        # 连接各种 GTK 事件到相应的处理方法
        self.connect('scroll_event',         self.scroll_event)
        self.connect('button_press_event',   self.button_press_event)
        self.connect('button_release_event', self.button_release_event)
        self.connect('configure_event',      self.configure_event)
        self.connect('screen-changed',       self._update_device_pixel_ratio)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)
        self.connect('draw',                 self.on_draw_event)
        self.connect('draw',                 self._post_draw)
        self.connect('key_press_event',      self.key_press_event)
        self.connect('key_release_event',    self.key_release_event)
        self.connect('motion_notify_event',  self.motion_notify_event)
        self.connect('enter_notify_event',   self.enter_notify_event)
        self.connect('leave_notify_event',   self.leave_notify_event)
        self.connect('size_allocate',        self.size_allocate)

        # 设置事件掩码，指示组件处理的事件类型
        self.set_events(self.__class__.event_mask)

        # 设置组件可获取焦点
        self.set_can_focus(True)

        # 创建并加载 CSS 提供者，设置组件的背景颜色为白色
        css = Gtk.CssProvider()
        css.load_from_data(b".matplotlib-canvas { background-color: white; }")
        # 获取组件的样式上下文，添加 CSS 提供者
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        # 为组件添加 CSS 类名 "matplotlib-canvas"
        style_ctx.add_class("matplotlib-canvas")

    # 销毁方法，处理关闭事件
    def destroy(self):
        # 创建关闭事件对象，执行关闭事件处理
        CloseEvent("close_event", self)._process()

    # 设置鼠标光标的方法，继承自父类的文档字符串
    def set_cursor(self, cursor):
        # 获取组件的窗口属性
        window = self.get_property("window")
        # 如果窗口不为 None，则设置窗口的光标为经转换后的 GTK 光标
        if window is not None:
            window.set_cursor(_mpl_to_gtk_cursor(cursor))
            # 获取默认的 GLib 主上下文，执行主循环的迭代
            context = GLib.MainContext.default()
            context.iteration(True)

    # 将 GTK 事件的位置或当前光标位置（如果 event 为 None）转换为 Matplotlib 坐标的方法
    def _mpl_coords(self, event=None):
        """
        Convert the position of a GTK event, or of the current cursor position
        if *event* is None, to Matplotlib coordinates.

        GTK use logical pixels, but the figure is scaled to physical pixels for
        rendering.  Transform to physical pixels so that all of the down-stream
        transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        # 如果 event 为 None，则获取组件的窗口，并获取设备的位置信息
        if event is None:
            window = self.get_window()
            t, x, y, state = window.get_device_position(
                window.get_display().get_device_manager().get_client_pointer())
        else:
            # 否则，使用 event 参数的 x 和 y 值
            x, y = event.x, event.y
        # 将 x 坐标乘以设备像素比例，以转换为物理像素
        x = x * self.device_pixel_ratio
        # 翻转 y 坐标，使得 y=0 位于画布底部
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        # 返回转换后的 x, y 坐标
        return x, y
    # 处理滚动事件的回调函数，根据滚动方向确定步进值
    def scroll_event(self, widget, event):
        step = 1 if event.direction == Gdk.ScrollDirection.UP else -1
        # 创建 MouseEvent 实例并处理滚动事件
        MouseEvent("scroll_event", self,
                   *self._mpl_coords(event), step=step,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # 停止事件传播？

    # 处理鼠标按下事件的回调函数
    def button_press_event(self, widget, event):
        # 创建 MouseEvent 实例并处理鼠标按下事件
        MouseEvent("button_press_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # 停止事件传播？

    # 处理鼠标释放事件的回调函数
    def button_release_event(self, widget, event):
        # 创建 MouseEvent 实例并处理鼠标释放事件
        MouseEvent("button_release_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # 停止事件传播？

    # 处理键盘按下事件的回调函数
    def key_press_event(self, widget, event):
        # 创建 KeyEvent 实例并处理键盘按下事件
        KeyEvent("key_press_event", self,
                 self._get_key(event), *self._mpl_coords(event),
                 guiEvent=event)._process()
        return True  # 停止事件传播

    # 处理键盘释放事件的回调函数
    def key_release_event(self, widget, event):
        # 创建 KeyEvent 实例并处理键盘释放事件
        KeyEvent("key_release_event", self,
                 self._get_key(event), *self._mpl_coords(event),
                 guiEvent=event)._process()
        return True  # 停止事件传播

    # 处理鼠标移动事件的回调函数
    def motion_notify_event(self, widget, event):
        # 创建 MouseEvent 实例并处理鼠标移动事件
        MouseEvent("motion_notify_event", self, *self._mpl_coords(event),
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # 停止事件传播？

    # 处理鼠标进入事件的回调函数
    def enter_notify_event(self, widget, event):
        # 获取当前 GTK 窗口的修饰键状态
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        # 创建 LocationEvent 实例并处理鼠标进入事件
        LocationEvent("figure_enter_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    # 处理鼠标离开事件的回调函数
    def leave_notify_event(self, widget, event):
        # 获取当前 GTK 窗口的修饰键状态
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        # 创建 LocationEvent 实例并处理鼠标离开事件
        LocationEvent("figure_leave_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    # 处理窗口尺寸分配变化的回调函数
    def size_allocate(self, widget, allocation):
        # 获取图形的 DPI 值
        dpival = self.figure.dpi
        # 根据窗口分配的尺寸和设备像素比例设置图形尺寸
        winch = allocation.width * self.device_pixel_ratio / dpival
        hinch = allocation.height * self.device_pixel_ratio / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        # 创建 ResizeEvent 实例并处理窗口尺寸变化事件
        ResizeEvent("resize_event", self)._process()
        self.draw_idle()
    # 定义私有方法 `_mpl_modifiers`，用于根据事件状态获取修饰键列表，可以排除指定的修饰键。
    def _mpl_modifiers(event_state, *, exclude=None):
        # 定义修饰键和其对应的 Gdk.ModifierType 控制位和键名的列表
        modifiers = [
            ("ctrl", Gdk.ModifierType.CONTROL_MASK, "control"),
            ("alt", Gdk.ModifierType.MOD1_MASK, "alt"),
            ("shift", Gdk.ModifierType.SHIFT_MASK, "shift"),
            ("super", Gdk.ModifierType.MOD4_MASK, "super"),
        ]
        # 返回修饰键列表，排除了指定的修饰键
        return [name for name, mask, key in modifiers
                if exclude != key and event_state & mask]
    
    # 定义私有方法 `_get_key`，用于获取事件对应的键名字符串，考虑修饰键的影响
    def _get_key(self, event):
        # 获取按键的 unicode 字符串
        unikey = chr(Gdk.keyval_to_unicode(event.keyval))
        # 将 unicode 字符串转换为 matplotlib 的键名
        key = cbook._unikey_or_keysym_to_mplkey(
            unikey, Gdk.keyval_name(event.keyval))
        # 获取修饰键列表，排除当前键名
        mods = self._mpl_modifiers(event.state, exclude=key)
        # 如果修饰键列表中包含 "shift" 并且当前键是可打印字符，则移除 "shift"
        if "shift" in mods and unikey.isprintable():
            mods.remove("shift")
        # 返回修饰键和键名组成的字符串
        return "+".join([*mods, key])
    
    # 定义私有方法 `_update_device_pixel_ratio`，用于更新设备像素比率相关逻辑
    def _update_device_pixel_ratio(self, *args, **kwargs):
        # 如果设置设备像素比率成功，则重新调整画布大小并重绘
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            # 通过触发 resize 事件来重新调整画布大小
            self.queue_resize()
            # 重新绘制画布
            self.queue_draw()
    
    # 定义 configure_event 方法，处理 GTK 配置事件
    def configure_event(self, widget, event):
        # 如果 widget 的窗口属性为空，则直接返回
        if widget.get_property("window") is None:
            return
        # 根据设备像素比率调整宽度和高度
        w = event.width * self.device_pixel_ratio
        h = event.height * self.device_pixel_ratio
        # 如果宽度或高度小于 3，则返回，不执行后续操作
        if w < 3 or h < 3:
            return  # empty fig
        # 计算 DPI
        dpi = self.figure.dpi
        # 设置画布大小（单位为英寸）
        self.figure.set_size_inches(w / dpi, h / dpi, forward=False)
        return False  # 是否继续传播事件？
    
    # 定义私有方法 `_draw_rubberband`，用于绘制选择框
    def _draw_rubberband(self, rect):
        # 设置选择框的矩形区域
        self._rubberband_rect = rect
        # TODO: 只更新选择框区域
        self.queue_draw()
    
    # 定义私有方法 `_post_draw`，用于在绘制后进行一些操作
    def _post_draw(self, widget, ctx):
        # 如果选择框的矩形区域为空，则直接返回
        if self._rubberband_rect is None:
            return
        # 根据设备像素比率计算选择框的坐标和尺寸
        x0, y0, w, h = (dim / self.device_pixel_ratio
                        for dim in self._rubberband_rect)
        x1 = x0 + w
        y1 = y0 + h
        # 在 ctx 上绘制选择框的边框线条，以避免放大时虚线跳跃
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)
    
        # 设置抗锯齿效果、线宽和颜色，并绘制边框
        ctx.set_antialias(1)
        ctx.set_line_width(1)
        ctx.set_dash((3, 3), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()
    
        # 设置虚线样式和颜色，并绘制虚线
        ctx.set_dash((3, 3), 3)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()
    
    # 定义 on_draw_event 方法，用于绘制事件的空方法，由 GTK3Agg 或 GTK3Cairo 类覆盖实现
    def on_draw_event(self, widget, ctx):
        # 由 GTK3Agg 或 GTK3Cairo 类覆盖实现
        pass
    
    # 定义 draw 方法，绘制图形
    def draw(self):
        # 继承的文档字符串
        if self.is_drawable():
            # 请求重新绘制
            self.queue_draw()
    # 绘图空闲时调用的方法，用于在空闲状态下绘制界面
    def draw_idle(self):
        # 继承的文档字符串
        # 如果已经有绘图空闲标识，则直接返回
        if self._idle_draw_id != 0:
            return
        
        # 定义一个在空闲时执行绘图的函数
        def idle_draw(*args):
            try:
                self.draw()  # 调用本对象的绘图方法
            finally:
                self._idle_draw_id = 0  # 将绘图空闲标识重置为0
            return False
        
        # 将空闲绘图函数添加到GLib的空闲调度器中，并记录其标识
        self._idle_draw_id = GLib.idle_add(idle_draw)

    # 刷新事件的方法，处理未处理的事件直到全部处理完毕
    def flush_events(self):
        # 继承的文档字符串
        # 获取默认的主上下文
        context = GLib.MainContext.default()
        
        # 当主上下文中有未处理的事件时，持续迭代处理
        while context.pending():
            context.iteration(True)
class NavigationToolbar2GTK3(_NavigationToolbar2GTK, Gtk.Toolbar):
    # 定义 NavigationToolbar2GTK3 类，继承自 _NavigationToolbar2GTK 和 Gtk.Toolbar
    def __init__(self, canvas):
        # 初始化方法，接受一个 canvas 参数
        GObject.GObject.__init__(self)

        # 设置工具栏的样式为带图标的样式
        self.set_style(Gtk.ToolbarStyle.ICONS)

        # 用于存储每个工具项的 GTK ID 的字典
        self._gtk_ids = {}

        # 遍历每个工具项，依次添加到工具栏中
        for text, tooltip_text, image_file, callback in self.toolitems:
            # 如果工具项的文本为 None，则插入分隔符工具项并继续下一轮循环
            if text is None:
                self.insert(Gtk.SeparatorToolItem(), -1)
                continue
            
            # 根据图标文件名构建 Gtk.Image 对象
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(
                    str(cbook._get_data_path('images',
                                             f'{image_file}-symbolic.svg'))),
                Gtk.IconSize.LARGE_TOOLBAR)
            
            # 根据回调函数名确定创建的按钮类型
            self._gtk_ids[text] = button = (
                Gtk.ToggleToolButton() if callback in ['zoom', 'pan'] else
                Gtk.ToolButton())
            
            # 设置按钮的标签文本和图标
            button.set_label(text)
            button.set_icon_widget(image)
            
            # 连接按钮的 'clicked' 信号到相应的回调函数，并保存信号处理程序的 ID
            button._signal_handler = button.connect(
                'clicked', getattr(self, callback))
            
            # 设置按钮的工具提示文本
            button.set_tooltip_text(tooltip_text)
            
            # 将按钮插入工具栏中
            self.insert(button, -1)

        # 插入填充项，确保工具栏至少有两行高度，以避免鼠标悬停在图像上时导致的重绘问题
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        label = Gtk.Label()
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        toolitem.set_expand(True)  # 将真实消息推送到右侧
        toolitem.add(label)

        # 插入消息显示的工具项
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        toolitem.add(self.message)

        # 显示所有工具项
        self.show_all()

        # 调用父类 _NavigationToolbar2GTK 的初始化方法，传入 canvas 参数
        _NavigationToolbar2GTK.__init__(self, canvas)
    # 定义一个方法，用于保存图形到文件
    def save_figure(self, *args):
        # 创建一个文件选择对话框
        dialog = Gtk.FileChooserDialog(
            title="Save the figure",  # 对话框标题
            parent=self.canvas.get_toplevel(),  # 设置父窗口为当前画布的顶层窗口
            action=Gtk.FileChooserAction.SAVE,  # 设置对话框操作为保存文件
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,  # 设置对话框按钮
                     Gtk.STOCK_SAVE,   Gtk.ResponseType.OK),
        )
        
        # 遍历画布支持的文件类型，并添加过滤器到对话框中
        for name, fmts \
                in self.canvas.get_supported_filetypes_grouped().items():
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            # 如果当前默认文件类型在支持的文件类型列表中，则将该过滤器设为默认过滤器
            if self.canvas.get_default_filetype() in fmts:
                dialog.set_filter(ff)

        # 定义一个回调函数，用于在过滤器变化时更新文件名的后缀
        @functools.partial(dialog.connect, "notify::filter")
        def on_notify_filter(*args):
            name = dialog.get_filter().get_name()
            fmt = self.canvas.get_supported_filetypes_grouped()[name][0]
            dialog.set_current_name(
                str(Path(dialog.get_current_name()).with_suffix(f'.{fmt}')))

        # 设置对话框打开时的初始文件夹为保存图形的默认文件夹
        dialog.set_current_folder(mpl.rcParams["savefig.directory"])
        # 设置对话框显示的初始文件名为画布的默认文件名
        dialog.set_current_name(self.canvas.get_default_filename())
        # 设置是否进行覆盖确认
        dialog.set_do_overwrite_confirmation(True)

        # 运行对话框并等待用户响应
        response = dialog.run()
        # 获取用户选择的文件名
        fname = dialog.get_filename()
        # 获取当前选择的文件过滤器
        ff = dialog.get_filter()
        # 根据过滤器获取选择的文件格式
        fmt = self.canvas.get_supported_filetypes_grouped()[ff.get_name()][0]
        # 销毁对话框
        dialog.destroy()
        
        # 如果用户没有确认保存操作，则直接返回
        if response != Gtk.ResponseType.OK:
            return
        
        # 如果设置了保存图形的默认文件夹路径，则更新该路径为保存图形时的目录
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
        
        try:
            # 尝试将画布中的图形保存为指定格式的文件
            self.canvas.figure.savefig(fname, format=fmt)
        except Exception as e:
            # 如果保存失败，则创建一个错误消息对话框显示异常信息
            dialog = Gtk.MessageDialog(
                parent=self.canvas.get_toplevel(), message_format=str(e),
                type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK)
            dialog.run()
            dialog.destroy()
# 定义一个名为 ToolbarGTK3 的类，继承自 ToolContainerBase 和 Gtk.Box
class ToolbarGTK3(ToolContainerBase, Gtk.Box):
    # 类变量，图标文件扩展名
    _icon_extension = '-symbolic.svg'

    # 初始化方法，接受 toolmanager 参数
    def __init__(self, toolmanager):
        # 调用 ToolContainerBase 类的初始化方法
        ToolContainerBase.__init__(self, toolmanager)
        # 调用 Gtk.Box 类的初始化方法
        Gtk.Box.__init__(self)
        # 设置盒子的属性为水平方向
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        # 创建一个 Gtk.Label 对象用于显示消息，并设置右对齐
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)
        # 将消息标签放在盒子的末尾
        self.pack_end(self._message, False, False, 0)
        # 显示所有子部件
        self.show_all()
        # 初始化工具组和工具项目字典
        self._groups = {}
        self._toolitems = {}

    # 添加工具项方法，接受名称、组、位置、图像文件、描述和开关标志作为参数
    def add_toolitem(self, name, group, position, image_file, description, toggle):
        # 根据开关标志选择创建 Gtk.ToggleToolButton 或 Gtk.ToolButton
        if toggle:
            button = Gtk.ToggleToolButton()
        else:
            button = Gtk.ToolButton()
        # 设置按钮的标签为名称
        button.set_label(name)

        # 如果图像文件不为 None，则创建一个 Gtk.Image 对象作为按钮的图标
        if image_file is not None:
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(image_file),
                Gtk.IconSize.LARGE_TOOLBAR)
            button.set_icon_widget(image)

        # 如果位置为 None，则设为 -1
        if position is None:
            position = -1

        # 调用内部方法将按钮添加到工具栏
        self._add_button(button, group, position)
        # 连接按钮的 'clicked' 信号到 _call_tool 方法，并设置工具提示文本为描述
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)
        button.show_all()
        # 将按钮和信号的元组添加到对应名称的工具项目列表中
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    # 内部方法，将按钮添加到指定组的工具栏中的指定位置
    def _add_button(self, button, group, position):
        # 如果组不在 _groups 字典中，则创建一个新的 Gtk.Toolbar 并添加到盒子中
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            toolbar = Gtk.Toolbar()
            toolbar.set_style(Gtk.ToolbarStyle.ICONS)
            self.pack_start(toolbar, False, False, 0)
            toolbar.show_all()
            self._groups[group] = toolbar
        # 将按钮插入到对应组的工具栏中的指定位置
        self._groups[group].insert(button, position)

    # 内部方法，按钮点击时调用的方法，触发对应名称的工具
    def _call_tool(self, btn, name):
        self.trigger_tool(name)

    # 切换工具项方法，接受名称和切换状态作为参数
    def toggle_toolitem(self, name, toggled):
        # 如果名称不在 _toolitems 中，则返回
        if name not in self._toolitems:
            return
        # 遍历名称对应的所有工具项目，阻塞信号并设置活动状态，最后解除信号阻塞
        for toolitem, signal in self._toolitems[name]:
            toolitem.handler_block(signal)
            toolitem.set_active(toggled)
            toolitem.handler_unblock(signal)

    # 移除工具项方法，接受名称作为参数
    def remove_toolitem(self, name):
        # 遍历名称对应的所有工具项目，从所有工具组中移除对应的工具项
        for toolitem, _signal in self._toolitems.pop(name, []):
            for group in self._groups:
                if toolitem in self._groups[group]:
                    self._groups[group].remove(toolitem)

    # 内部方法，向盒子中添加垂直分隔条
    def _add_separator(self):
        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.VERTICAL)
        self.pack_start(sep, False, True, 0)
        sep.show_all()

    # 设置消息文本的方法，接受字符串 s 作为参数
    def set_message(self, s):
        self._message.set_label(s)


# 使用 backend_tools._register_tool_class 注册 FigureCanvasGTK3 类的工具类
@backend_tools._register_tool_class(FigureCanvasGTK3)
class SaveFigureGTK3(backend_tools.SaveFigureBase):
    # 触发方法，接受任意位置和关键字参数
    def trigger(self, *args, **kwargs):
        # 调用 NavigationToolbar2GTK3 类的 save_figure 方法，返回经典样式伪工具栏的对象
        NavigationToolbar2GTK3.save_figure(
            self._make_classic_style_pseudo_toolbar())


# 使用 backend_tools._register_tool_class 注册 FigureCanvasGTK3 类的帮助工具类
@backend_tools._register_tool_class(FigureCanvasGTK3)
class HelpGTK3(backend_tools.ToolHelpBase):
    pass
    def _normalize_shortcut(self, key):
        """
        Convert Matplotlib key presses to GTK+ accelerator identifiers.

        Related to `FigureCanvasGTK3._get_key`.
        """
        # 定义特殊键的映射关系
        special = {
            'backspace': 'BackSpace',
            'pagedown': 'Page_Down',
            'pageup': 'Page_Up',
            'scroll_lock': 'Scroll_Lock',
        }

        # 将快捷键按加号分割为不同的部分
        parts = key.split('+')
        # 将除了最后一个部分之外的其他部分格式化为 GTK+ 的修饰键格式
        mods = ['<' + mod + '>' for mod in parts[:-1]]
        # 取最后一个部分作为按键
        key = parts[-1]

        # 根据特殊键的映射替换按键名
        if key in special:
            key = special[key]
        elif len(key) > 1:
            key = key.capitalize()
        elif key.isupper():
            mods += ['<shift>']

        # 返回格式化后的 GTK+ 加速键标识符
        return ''.join(mods) + key

    def _is_valid_shortcut(self, key):
        """
        Check for a valid shortcut to be displayed.

        - GTK will never send 'cmd+' (see `FigureCanvasGTK3._get_key`).
        - The shortcut window only shows keyboard shortcuts, not mouse buttons.
        """
        # 检查快捷键是否有效显示
        return 'cmd+' not in key and not key.startswith('MouseButton.')

    def _show_shortcuts_window(self):
        # 创建一个新的快捷键部分
        section = Gtk.ShortcutsSection()

        # 遍历所有工具并按名称排序
        for name, tool in sorted(self.toolmanager.tools.items()):
            # 如果工具没有描述则跳过
            if not tool.description:
                continue

            # 创建一个新的快捷键组
            group = Gtk.ShortcutsGroup()
            section.add(group)
            # 由于没有组命名，这是一个删除标题的Hack
            group.forall(lambda widget, data: widget.set_visible(False), None)

            # 创建一个新的快捷键对象
            shortcut = Gtk.ShortcutsShortcut(
                accelerator=' '.join(
                    self._normalize_shortcut(key)
                    for key in self.toolmanager.get_tool_keymap(name)
                    if self._is_valid_shortcut(key)),
                title=tool.name,
                subtitle=tool.description)
            group.add(shortcut)

        # 创建一个新的快捷键窗口
        window = Gtk.ShortcutsWindow(
            title='Help',
            modal=True,
            transient_for=self._figure.canvas.get_toplevel())
        # 显示快捷键部分，必须在添加前显式调用
        section.show()
        window.add(section)

        # 显示所有的窗口内容
        window.show_all()

    def _show_shortcuts_dialog(self):
        # 创建一个新的消息对话框
        dialog = Gtk.MessageDialog(
            self._figure.canvas.get_toplevel(),
            0, Gtk.MessageType.INFO, Gtk.ButtonsType.OK, self._get_help_text(),
            title="Help")
        dialog.run()
        dialog.destroy()

    def trigger(self, *args):
        # 检查 GTK 版本是否大于或等于 3.20.0
        if Gtk.check_version(3, 20, 0) is None:
            # 显示快捷键窗口
            self._show_shortcuts_window()
        else:
            # 显示快捷键对话框
            self._show_shortcuts_dialog()
@backend_tools._register_tool_class(FigureCanvasGTK3)
# 使用 FigureCanvasGTK3 注册工具类到 backend_tools 模块
class ToolCopyToClipboardGTK3(backend_tools.ToolCopyToClipboardBase):
    # ToolCopyToClipboardGTK3 类继承自 backend_tools.ToolCopyToClipboardBase 类
    def trigger(self, *args, **kwargs):
        # 触发方法，复制图像到剪贴板
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        # 获取剪贴板对象
        window = self.canvas.get_window()
        # 获取画布窗口对象
        x, y, width, height = window.get_geometry()
        # 获取窗口的几何信息：位置和大小
        pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
        # 从窗口获取 GdkPixbuf 对象
        clipboard.set_image(pb)
        # 将 GdkPixbuf 对象设置到剪贴板


Toolbar = ToolbarGTK3
# 将 ToolbarGTK3 赋值给 Toolbar 变量
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.ConfigureSubplotsGTK)
# 使用 FigureCanvasGTK3 和 _backend_gtk.ConfigureSubplotsGTK 注册工具类到 backend_tools 模块
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.RubberbandGTK)
# 使用 FigureCanvasGTK3 和 _backend_gtk.RubberbandGTK 注册工具类到 backend_tools 模块


class FigureManagerGTK3(_FigureManagerGTK):
    # FigureManagerGTK3 类继承自 _FigureManagerGTK 类
    _toolbar2_class = NavigationToolbar2GTK3
    # 设置 _toolbar2_class 属性为 NavigationToolbar2GTK3
    _toolmanager_toolbar_class = ToolbarGTK3
    # 设置 _toolmanager_toolbar_class 属性为 ToolbarGTK3


@_BackendGTK.export
# 导出 _BackendGTK3 类到 _BackendGTK 模块
class _BackendGTK3(_BackendGTK):
    # _BackendGTK3 类继承自 _BackendGTK 类
    FigureCanvas = FigureCanvasGTK3
    # 设置 FigureCanvas 属性为 FigureCanvasGTK3
    FigureManager = FigureManagerGTK3
    # 设置 FigureManager 属性为 FigureManagerGTK3
```