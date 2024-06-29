# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_gtk4.py`

```py
# 导入必要的模块和库
import functools
import io
import os

import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
    ToolContainerBase, KeyEvent, LocationEvent, MouseEvent, ResizeEvent,
    CloseEvent)

try:
    import gi
except ImportError as err:
    # 如果导入失败，抛出自定义的 ImportError
    raise ImportError("The GTK4 backends require PyGObject") from err

try:
    # 检查并要求特定版本的 Gtk
    gi.require_version("Gtk", "4.0")
except ValueError as e:
    # 如果版本要求不满足，将 ValueError 转换为 ImportError
    raise ImportError(e) from e

# 导入需要的 GTK 相关模块
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (
    _BackendGTK, _FigureCanvasGTK, _FigureManagerGTK, _NavigationToolbar2GTK,
    TimerGTK as TimerGTK4,
)


class FigureCanvasGTK4(_FigureCanvasGTK, Gtk.DrawingArea):
    # 指定所需的交互框架为 gtk4
    required_interactive_framework = "gtk4"
    # 不支持位图传输
    supports_blit = False
    # 管理器类为 FigureManagerGTK4
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK4)
    # 上下文是否被缩放，默认为 False
    _context_is_scaled = False

    def __init__(self, figure=None):
        super().__init__(figure=figure)

        # 设置水平和垂直扩展属性为 True
        self.set_hexpand(True)
        self.set_vexpand(True)

        # 初始化空闲绘制 ID 和橡皮筋矩形
        self._idle_draw_id = 0
        self._rubberband_rect = None

        # 设置绘制函数为 _draw_func，并连接 resize 和 scale-factor 通知事件
        self.set_draw_func(self._draw_func)
        self.connect('resize', self.resize_event)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)

        # 添加点击手势控制器，连接按钮按下和释放事件
        click = Gtk.GestureClick()
        click.set_button(0)  # 所有按钮
        click.connect('pressed', self.button_press_event)
        click.connect('released', self.button_release_event)
        self.add_controller(click)

        # 添加键盘事件控制器，连接键盘按下和释放事件
        key = Gtk.EventControllerKey()
        key.connect('key-pressed', self.key_press_event)
        key.connect('key-released', self.key_release_event)
        self.add_controller(key)

        # 添加鼠标移动事件控制器，连接鼠标移动、进入和离开事件
        motion = Gtk.EventControllerMotion()
        motion.connect('motion', self.motion_notify_event)
        motion.connect('enter', self.enter_notify_event)
        motion.connect('leave', self.leave_notify_event)
        self.add_controller(motion)

        # 添加滚轮滚动事件控制器，连接滚轮滚动事件
        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect('scroll', self.scroll_event)
        self.add_controller(scroll)

        # 设置可聚焦性为 True

        self.set_focusable(True)

        # 设置背景样式为白色
        css = Gtk.CssProvider()
        style = '.matplotlib-canvas { background-color: white; }'
        if Gtk.check_version(4, 9, 3) is None:
            css.load_from_data(style, -1)
        else:
            css.load_from_data(style.encode('utf-8'))
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        style_ctx.add_class("matplotlib-canvas")

    # 销毁方法，处理关闭事件
    def destroy(self):
        CloseEvent("close_event", self)._process()
    def set_cursor(self, cursor):
        # 继承了文档字符串的功能，设置鼠标光标
        self.set_cursor_from_name(_backend_gtk.mpl_to_gtk_cursor_name(cursor))

    def _mpl_coords(self, xy=None):
        """
        将 GTK 事件的 *xy* 位置或者当前光标位置（如果 *xy* 为 None）转换为 Matplotlib 坐标。

        GTK 使用逻辑像素，但是图形将其缩放到物理像素以进行渲染。需要转换为物理像素，以确保所有下游的变换按预期工作。

        另外，坐标原点不同，需要进行修正。
        """
        if xy is None:
            surface = self.get_native().get_surface()
            is_over, x, y, mask = surface.get_device_position(
                self.get_display().get_default_seat().get_pointer())
        else:
            x, y = xy
        x = x * self.device_pixel_ratio
        # 翻转 y 轴，使得 y=0 位于画布底部
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        return x, y

    def scroll_event(self, controller, dx, dy):
        MouseEvent(
            "scroll_event", self, *self._mpl_coords(), step=dy,
            modifiers=self._mpl_modifiers(controller),
        )._process()
        return True

    def button_press_event(self, controller, n_press, x, y):
        MouseEvent(
            "button_press_event", self, *self._mpl_coords((x, y)),
            controller.get_current_button(),
            modifiers=self._mpl_modifiers(controller),
        )._process()
        self.grab_focus()

    def button_release_event(self, controller, n_press, x, y):
        MouseEvent(
            "button_release_event", self, *self._mpl_coords((x, y)),
            controller.get_current_button(),
            modifiers=self._mpl_modifiers(controller),
        )._process()

    def key_press_event(self, controller, keyval, keycode, state):
        KeyEvent(
            "key_press_event", self, self._get_key(keyval, keycode, state),
            *self._mpl_coords(),
        )._process()
        return True

    def key_release_event(self, controller, keyval, keycode, state):
        KeyEvent(
            "key_release_event", self, self._get_key(keyval, keycode, state),
            *self._mpl_coords(),
        )._process()
        return True

    def motion_notify_event(self, controller, x, y):
        MouseEvent(
            "motion_notify_event", self, *self._mpl_coords((x, y)),
            modifiers=self._mpl_modifiers(controller),
        )._process()

    def enter_notify_event(self, controller, x, y):
        LocationEvent(
            "figure_enter_event", self, *self._mpl_coords((x, y)),
            modifiers=self._mpl_modifiers(),
        )._process()

    def leave_notify_event(self, controller):
        LocationEvent(
            "figure_leave_event", self, *self._mpl_coords(),
            modifiers=self._mpl_modifiers(),
        )._process()
    # 当窗口大小发生变化时调整图形的大小
    def resize_event(self, area, width, height):
        # 更新设备像素比例以适应高分辨率显示
        self._update_device_pixel_ratio()
        # 获取图形对象的 DPI
        dpi = self.figure.dpi
        # 计算调整后的宽度和高度（英寸）
        winch = width * self.device_pixel_ratio / dpi
        hinch = height * self.device_pixel_ratio / dpi
        # 设置图形对象的尺寸
        self.figure.set_size_inches(winch, hinch, forward=False)
        # 触发 ResizeEvent 事件
        ResizeEvent("resize_event", self)._process()
        # 重新绘制图形
        self.draw_idle()

    # 获取当前按下的修饰键列表
    def _mpl_modifiers(self, controller=None):
        if controller is None:
            # 获取本地平台的表面并获取指针的位置和状态
            surface = self.get_native().get_surface()
            is_over, x, y, event_state = surface.get_device_position(
                self.get_display().get_default_seat().get_pointer())
        else:
            # 获取控制器的当前事件状态
            event_state = controller.get_current_event_state()
        # 定义修饰键与对应的 Gdk.ModifierType 的映射表
        mod_table = [
            ("ctrl", Gdk.ModifierType.CONTROL_MASK),
            ("alt", Gdk.ModifierType.ALT_MASK),
            ("shift", Gdk.ModifierType.SHIFT_MASK),
            ("super", Gdk.ModifierType.SUPER_MASK),
        ]
        # 返回当前按下的修饰键列表
        return [name for name, mask in mod_table if event_state & mask]

    # 根据键值、键码和状态获取按键的字符串表示（包含修饰键）
    def _get_key(self, keyval, keycode, state):
        # 获取键对应的 Unicode 字符
        unikey = chr(Gdk.keyval_to_unicode(keyval))
        # 使用 matplotlib 工具函数获取按键名称
        key = cbook._unikey_or_keysym_to_mplkey(
            unikey,
            Gdk.keyval_name(keyval))
        # 定义修饰键与对应的 Gdk.ModifierType 及字符串表示的映射表
        modifiers = [
            ("ctrl", Gdk.ModifierType.CONTROL_MASK, "control"),
            ("alt", Gdk.ModifierType.ALT_MASK, "alt"),
            ("shift", Gdk.ModifierType.SHIFT_MASK, "shift"),
            ("super", Gdk.ModifierType.SUPER_MASK, "super"),
        ]
        # 获取当前按下的修饰键列表（不包含当前按下的键）
        mods = [
            mod for mod, mask, mod_key in modifiers
            if (mod_key != key and state & mask
                and not (mod == "shift" and unikey.isprintable()))]
        # 返回包含修饰键和键的字符串表示的列表
        return "+".join([*mods, key])

    # 更新设备像素比例的方法
    def _update_device_pixel_ratio(self, *args, **kwargs):
        # 如果设备像素比例发生变化，则重新绘制图形
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            self.draw()

    # 绘制橡皮筋效果（虚线矩形）
    def _draw_rubberband(self, rect):
        # 设置橡皮筋的矩形区域
        self._rubberband_rect = rect
        # 刷新绘图区域以显示橡皮筋效果
        self.queue_draw()

    # 绘制函数的回调方法
    def _draw_func(self, drawing_area, ctx, width, height):
        # 执行绘制事件的回调方法
        self.on_draw_event(self, ctx)
        # 执行绘制后的处理方法
        self._post_draw(self, ctx)
    def _post_draw(self, widget, ctx):
        # 如果没有定义选取框的位置信息，直接返回
        if self._rubberband_rect is None:
            return

        # 初始化线条宽度和虚线间隔
        lw = 1
        dash = 3

        # 如果上下文没有进行缩放
        if not self._context_is_scaled:
            # 将选取框的位置信息转换为设备像素比例下的值
            x0, y0, w, h = (dim / self.device_pixel_ratio
                            for dim in self._rubberband_rect)
        else:
            # 否则直接使用选取框的位置信息
            x0, y0, w, h = self._rubberband_rect
            # 根据设备像素比例调整线条宽度和虚线间隔
            lw *= self.device_pixel_ratio
            dash *= self.device_pixel_ratio

        # 计算选取框的右下角坐标
        x1 = x0 + w
        y1 = y0 + h

        # 从左上角到右下角绘制四条线段，避免缩放时虚线“跳跃”
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)

        # 设置反锯齿效果、线条宽度、虚线样式、绘制颜色并描边
        ctx.set_antialias(1)
        ctx.set_line_width(lw)
        ctx.set_dash((dash, dash), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()

        # 修改虚线偏移并设置新的绘制颜色再次描边
        ctx.set_dash((dash, dash), dash)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

    def on_draw_event(self, widget, ctx):
        # 由GTK4Agg或GTK4Cairo进行覆盖
        pass

    def draw(self):
        # 绘制方法继承的文档字符串
        if self.is_drawable():
            # 请求重绘
            self.queue_draw()

    def draw_idle(self):
        # 绘制方法继承的文档字符串
        if self._idle_draw_id != 0:
            return

        # 定义空闲绘制函数
        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False

        # 使用GLib调度空闲绘制
        self._idle_draw_id = GLib.idle_add(idle_draw)

    def flush_events(self):
        # 刷新事件方法继承的文档字符串
        context = GLib.MainContext.default()
        # 处理待处理事件，直到为空
        while context.pending():
            context.iteration(True)
# 创建一个名为 NavigationToolbar2GTK4 的类，它继承自 _NavigationToolbar2GTK 和 Gtk.Box
class NavigationToolbar2GTK4(_NavigationToolbar2GTK, Gtk.Box):
    # 初始化方法，接受一个 canvas 参数
    def __init__(self, canvas):
        # 调用 Gtk.Box 的初始化方法
        Gtk.Box.__init__(self)

        # 为工具栏添加 CSS 类 'toolbar'
        self.add_css_class('toolbar')

        # 存储各个工具项的 GTK ID
        self._gtk_ids = {}
        # 遍历工具栏中的每个工具项
        for text, tooltip_text, image_file, callback in self.toolitems:
            # 如果工具项的文本为 None，则添加一个分隔符并继续下一次循环
            if text is None:
                self.append(Gtk.Separator())
                continue
            # 使用 Gtk.Image.new_from_gicon 创建一个图像，图像的来源是指定路径下的 SVG 图标
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(
                    str(cbook._get_data_path('images',
                                             f'{image_file}-symbolic.svg'))))
            # 根据回调函数类型创建对应的按钮，ToggleButton 或者 Button
            self._gtk_ids[text] = button = (
                Gtk.ToggleButton() if callback in ['zoom', 'pan'] else
                Gtk.Button())
            # 设置按钮的子部件为图像
            button.set_child(image)
            # 为按钮添加 CSS 类 'flat' 和 'image-button'
            button.add_css_class('flat')
            button.add_css_class('image-button')
            # 将按钮的信号处理器 ID 存储起来，以便需要时可以阻止它
            button._signal_handler = button.connect(
                'clicked', getattr(self, callback))
            # 设置按钮的工具提示文本
            button.set_tooltip_text(tooltip_text)
            # 将按钮添加到工具栏中
            self.append(button)

        # 这个填充项确保工具栏至少有两行文本高度。
        # 否则，在鼠标悬停在图像上时，由于使用了两行消息，会导致工具栏重新绘制。
        label = Gtk.Label()
        # 设置标签的标记文本，使用小号字体确保空白字符占用的高度
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        # 设置标签可以水平扩展，以将真实的消息推到右侧
        label.set_hexpand(True)
        # 将填充标签添加到工具栏中
        self.append(label)

        # 创建一个消息标签，用于显示右对齐的文本消息
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        # 将消息标签添加到工具栏中
        self.append(self.message)

        # 调用 _NavigationToolbar2GTK 的初始化方法，传递 canvas 参数
        _NavigationToolbar2GTK.__init__(self, canvas)
    # 定义一个保存图形的方法，接受可变数量的参数
    def save_figure(self, *args):
        # 创建一个本地文件选择器对话框
        dialog = Gtk.FileChooserNative(
            title='Save the figure',
            transient_for=self.canvas.get_root(),
            action=Gtk.FileChooserAction.SAVE,
            modal=True)
        # 必须保持对话框的引用
        self._save_dialog = dialog

        # 创建一个文件过滤器，设置名称为'All files'，并添加所有文件的匹配模式
        ff = Gtk.FileFilter()
        ff.set_name('All files')
        ff.add_pattern('*')
        dialog.add_filter(ff)
        dialog.set_filter(ff)

        # 获取图形支持的文件格式，并添加到文件过滤器中
        formats = []
        default_format = None
        for i, (name, fmts) in enumerate(
                self.canvas.get_supported_filetypes_grouped().items()):
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            formats.append(name)
            if self.canvas.get_default_filetype() in fmts:
                default_format = i
        # 设置默认格式为第一个，并将其移动到列表的开头
        formats = [formats[default_format], *formats[:default_format],
                   *formats[default_format+1:]]
        dialog.add_choice('format', 'File format', formats, formats)
        dialog.set_choice('format', formats[0])

        # 设置当前文件夹为保存图形的默认文件夹
        dialog.set_current_folder(Gio.File.new_for_path(
            os.path.expanduser(mpl.rcParams['savefig.directory'])))
        # 设置默认文件名
        dialog.set_current_name(self.canvas.get_default_filename())

        # 响应对话框的事件
        @functools.partial(dialog.connect, 'response')
        def on_response(dialog, response):
            file = dialog.get_file()
            fmt = dialog.get_choice('format')
            fmt = self.canvas.get_supported_filetypes_grouped()[fmt][0]
            dialog.destroy()
            self._save_dialog = None
            if response != Gtk.ResponseType.ACCEPT:
                return
            # 保存图形到指定文件路径
            if mpl.rcParams['savefig.directory']:
                parent = file.get_parent()
                mpl.rcParams['savefig.directory'] = parent.get_path()
            try:
                self.canvas.figure.savefig(file.get_path(), format=fmt)
            except Exception as e:
                # 显示保存失败的错误消息
                msg = Gtk.MessageDialog(
                    transient_for=self.canvas.get_root(),
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK, modal=True,
                    text=str(e))
                msg.show()

        # 显示文件选择器对话框
        dialog.show()
# 定义一个名为 ToolbarGTK4 的类，继承自 ToolContainerBase 和 Gtk.Box
class ToolbarGTK4(ToolContainerBase, Gtk.Box):
    # 定义类变量 _icon_extension，其值为 '-symbolic.svg'
    _icon_extension = '-symbolic.svg'

    # 初始化方法，接受 toolmanager 参数
    def __init__(self, toolmanager):
        # 调用 ToolContainerBase 的初始化方法
        ToolContainerBase.__init__(self, toolmanager)
        # 调用 Gtk.Box 的初始化方法
        Gtk.Box.__init__(self)
        # 设置当前对象的属性 'orientation' 为水平方向
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)

        # 创建一个 Gtk.Box 对象，用于存放工具项
        self._tool_box = Gtk.Box()
        self.append(self._tool_box)  # 将 _tool_box 添加到当前对象中
        self._groups = {}  # 初始化空字典 _groups，用于存放分组信息
        self._toolitems = {}  # 初始化空字典 _toolitems，用于存放工具项信息

        # 创建一个填充项，确保工具栏至少有两行文本高度。
        # 这样可以避免鼠标悬停在图像上时导致重新绘制画布，因为图像使用两行消息，可能会改变工具栏的大小。
        label = Gtk.Label()
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        label.set_hexpand(True)  # 设置为扩展宽度，将实际消息推到右侧。
        self.append(label)  # 将 label 添加到当前对象中

        # 创建一个 Gtk.Label 对象 _message，用于显示右对齐的消息
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)  # 设置消息右对齐
        self.append(self._message)  # 将 _message 添加到当前对象中

    # 添加工具项的方法，接受 name、group、position、image_file、description 和 toggle 参数
    def add_toolitem(self, name, group, position, image_file, description,
                     toggle):
        # 根据 toggle 参数决定创建 Gtk.ToggleButton 或 Gtk.Button 对象
        if toggle:
            button = Gtk.ToggleButton()
        else:
            button = Gtk.Button()
        button.set_label(name)  # 设置按钮的标签为 name
        button.add_css_class('flat')  # 添加 CSS 类 'flat' 到按钮

        # 如果提供了 image_file，创建一个 Gtk.Image 对象并设置为按钮的子对象，同时添加 CSS 类 'image-button'
        if image_file is not None:
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(image_file))
            button.set_child(image)
            button.add_css_class('image-button')

        # 如果 position 为 None，则设置其为 -1
        if position is None:
            position = -1

        # 调用 _add_button 方法添加按钮到工具栏中
        self._add_button(button, group, position)
        # 连接按钮的 'clicked' 信号到 _call_tool 方法，并传递 name 参数
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)  # 设置按钮的工具提示文本为 description
        # 将按钮和其信号存储到 _toolitems 字典中
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    # 在指定的 group 和 position 处查找子组件的私有方法
    def _find_child_at_position(self, group, position):
        # 初始化 children 列表，将 None 加入其中
        children = [None]
        # 获取指定 group 的第一个子组件
        child = self._groups[group].get_first_child()
        # 遍历该 group 的所有子组件，将它们加入 children 列表
        while child is not None:
            children.append(child)
            child = child.get_next_sibling()
        # 返回指定 position 的子组件
        return children[position]

    # 添加按钮到工具栏中的私有方法，接受 button、group 和 position 参数
    def _add_button(self, button, group, position):
        # 如果 group 不在 _groups 字典中，则创建新的分组
        if group not in self._groups:
            # 如果 _groups 不为空，调用 _add_separator 方法添加分隔符
            if self._groups:
                self._add_separator()
            # 创建一个新的 Gtk.Box 对象作为分组容器，将其添加到 _tool_box 中
            group_box = Gtk.Box()
            self._tool_box.append(group_box)
            self._groups[group] = group_box  # 将分组信息存储到 _groups 字典中
        # 在指定的位置后面插入按钮到对应的分组中
        self._groups[group].insert_child_after(
            button, self._find_child_at_position(group, position))

    # 调用工具的私有方法，接受 btn 和 name 参数
    def _call_tool(self, btn, name):
        # 调用 trigger_tool 方法，触发指定名称的工具
        self.trigger_tool(name)

    # 切换工具项状态的方法，接受 name 和 toggled 参数
    def toggle_toolitem(self, name, toggled):
        # 如果 name 不在 _toolitems 中，直接返回
        if name not in self._toolitems:
            return
        # 遍历 _toolitems 中指定 name 的工具项，并设置其状态为 toggled
        for toolitem, signal in self._toolitems[name]:
            toolitem.handler_block(signal)  # 阻塞按钮的信号处理
            toolitem.set_active(toggled)  # 设置按钮的激活状态为 toggled
            toolitem.handler_unblock(signal)  # 解除按钮的信号处理阻塞
    # 定义一个方法，用于移除工具项
    def remove_toolitem(self, name):
        # 弹出指定名称的工具项及其关联的信号处理函数列表
        for toolitem, _signal in self._toolitems.pop(name, []):
            # 遍历所有分组
            for group in self._groups:
                # 如果工具项在当前分组中
                if toolitem in self._groups[group]:
                    # 移除工具项
                    self._groups[group].remove(toolitem)

    # 定义一个方法，用于向工具箱中添加分隔符
    def _add_separator(self):
        # 创建一个垂直方向的分隔符对象
        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.VERTICAL)
        # 将分隔符添加到工具箱中
        self._tool_box.append(sep)

    # 定义一个方法，用于设置消息显示内容
    def set_message(self, s):
        # 设置消息标签的文本内容为参数 s
        self._message.set_label(s)
# 将 SaveFigureGTK4 类注册为 FigureCanvasGTK4 的工具类
@backend_tools._register_tool_class(FigureCanvasGTK4)
class SaveFigureGTK4(backend_tools.SaveFigureBase):
    # 触发保存图形操作，调用 NavigationToolbar2GTK4 的 save_figure 方法
    def trigger(self, *args, **kwargs):
        NavigationToolbar2GTK4.save_figure(
            self._make_classic_style_pseudo_toolbar())


# 将 HelpGTK4 类注册为 FigureCanvasGTK4 的工具类
@backend_tools._register_tool_class(FigureCanvasGTK4)
class HelpGTK4(backend_tools.ToolHelpBase):
    # 将 Matplotlib 键盘按键转换为 GTK+ 加速器标识符
    def _normalize_shortcut(self, key):
        """
        Convert Matplotlib key presses to GTK+ accelerator identifiers.

        Related to `FigureCanvasGTK4._get_key`.
        """
        special = {
            'backspace': 'BackSpace',
            'pagedown': 'Page_Down',
            'pageup': 'Page_Up',
            'scroll_lock': 'Scroll_Lock',
        }

        parts = key.split('+')
        mods = ['<' + mod + '>' for mod in parts[:-1]]  # 提取修饰键
        key = parts[-1]  # 提取主键

        if key in special:  # 处理特殊键
            key = special[key]
        elif len(key) > 1:  # 大写非特殊键
            key = key.capitalize()
        elif key.isupper():  # 大写字母
            mods += ['<shift>']

        return ''.join(mods) + key

    # 检查是否是有效的快捷键以便显示
    def _is_valid_shortcut(self, key):
        """
        Check for a valid shortcut to be displayed.

        - GTK will never send 'cmd+' (see `FigureCanvasGTK4._get_key`).
        - The shortcut window only shows keyboard shortcuts, not mouse buttons.
        """
        return 'cmd+' not in key and not key.startswith('MouseButton.')

    # 触发帮助操作
    def trigger(self, *args):
        section = Gtk.ShortcutsSection()  # 创建快捷键部分

        for name, tool in sorted(self.toolmanager.tools.items()):  # 遍历工具列表
            if not tool.description:  # 跳过没有描述的工具
                continue

            # 将每个工具放入独立的组，允许 GTK 自动分割成不同的列/页面，特别是当有很多快捷键且部分键很宽时
            group = Gtk.ShortcutsGroup()  # 创建快捷键组
            section.append(group)  # 将组添加到部分

            # 一个小技巧，移除组的标题，因为我们没有组命名
            child = group.get_first_child()
            while child is not None:
                child.set_visible(False)
                child = child.get_next_sibling()

            # 创建快捷方式对象并添加到组中
            shortcut = Gtk.ShortcutsShortcut(
                accelerator=' '.join(
                    self._normalize_shortcut(key)
                    for key in self.toolmanager.get_tool_keymap(name)
                    if self._is_valid_shortcut(key)),
                title=tool.name,
                subtitle=tool.description)
            group.append(shortcut)

        # 创建帮助窗口并显示
        window = Gtk.ShortcutsWindow(
            title='Help',
            modal=True,
            transient_for=self._figure.canvas.get_root())
        window.set_child(section)
        window.show()


# 将 ToolCopyToClipboardGTK4 类注册为 FigureCanvasGTK4 的工具类
@backend_tools._register_tool_class(FigureCanvasGTK4)
class ToolCopyToClipboardGTK4(backend_tools.ToolCopyToClipboardBase):
    # 定义触发器方法，接受任意位置参数和关键字参数
    def trigger(self, *args, **kwargs):
        # 使用字节流对象f来保存绘制的RGBA数据
        with io.BytesIO() as f:
            # 将画布的RGBA数据输出到字节流f中
            self.canvas.print_rgba(f)
            # 获取画布的宽度和高度
            w, h = self.canvas.get_width_height()
            # 根据字节流f的缓冲区数据创建Pixbuf对象pb
            pb = GdkPixbuf.Pixbuf.new_from_data(f.getbuffer(),
                                                GdkPixbuf.Colorspace.RGB, True,
                                                8, w, h, w*4)
        # 获取画布的剪贴板对象
        clipboard = self.canvas.get_clipboard()
        # 将Pixbuf对象pb设置到剪贴板中
        clipboard.set(pb)
# 将 FigureCanvasGTK4 类和 _backend_gtk.ConfigureSubplotsGTK 函数注册为工具类
backend_tools._register_tool_class(
    FigureCanvasGTK4, _backend_gtk.ConfigureSubplotsGTK)

# 将 FigureCanvasGTK4 类和 _backend_gtk.RubberbandGTK 函数注册为工具类
backend_tools._register_tool_class(
    FigureCanvasGTK4, _backend_gtk.RubberbandGTK)

# 将 ToolbarGTK4 类赋值给 Toolbar 变量
Toolbar = ToolbarGTK4

# 定义 FigureManagerGTK4 类，继承自 _FigureManagerGTK 类
class FigureManagerGTK4(_FigureManagerGTK):
    # 设置 FigureManagerGTK4 类的工具栏2（_toolbar2_class）为 NavigationToolbar2GTK4 类
    _toolbar2_class = NavigationToolbar2GTK4
    # 设置 FigureManagerGTK4 类的工具管理器工具栏类（_toolmanager_toolbar_class）为 ToolbarGTK4 类

# 导出 _BackendGTK 类的子类 _BackendGTK4
@_BackendGTK.export
class _BackendGTK4(_BackendGTK):
    # 设置 _BackendGTK4 类的 FigureCanvas 属性为 FigureCanvasGTK4 类
    FigureCanvas = FigureCanvasGTK4
    # 设置 _BackendGTK4 类的 FigureManager 属性为 FigureManagerGTK4 类
    FigureManager = FigureManagerGTK4
```