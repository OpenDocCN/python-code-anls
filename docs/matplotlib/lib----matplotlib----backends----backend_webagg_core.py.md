# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_webagg_core.py`

```
"""Displays Agg images in the browser, with interactivity."""
# 导入所需的模块和库
import asyncio  # 异步IO库，用于异步操作
import datetime  # 日期和时间库，用于处理时间相关操作
from io import BytesIO, StringIO  # 导入字节流和字符串IO处理类
import json  # JSON格式数据处理库
import logging  # 日志记录模块
import os  # 系统操作相关库
from pathlib import Path  # 处理路径相关操作的库

import numpy as np  # 数值计算库
from PIL import Image  # Python Imaging Library，图像处理库

from matplotlib import _api, backend_bases, backend_tools  # Matplotlib相关模块
from matplotlib.backends import backend_agg  # Matplotlib Agg后端
from matplotlib.backend_bases import (
    _Backend, KeyEvent, LocationEvent, MouseEvent, ResizeEvent)  # Matplotlib基础后端相关事件

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_SPECIAL_KEYS_LUT = {'Alt': 'alt',  # 特殊按键映射表，将特殊按键映射到简化的形式
                     'AltGraph': 'alt',
                     'CapsLock': 'caps_lock',
                     'Control': 'control',
                     'Meta': 'meta',
                     'NumLock': 'num_lock',
                     'ScrollLock': 'scroll_lock',
                     'Shift': 'shift',
                     'Super': 'super',
                     'Enter': 'enter',
                     'Tab': 'tab',
                     'ArrowDown': 'down',
                     'ArrowLeft': 'left',
                     'ArrowRight': 'right',
                     'ArrowUp': 'up',
                     'End': 'end',
                     'Home': 'home',
                     'PageDown': 'pagedown',
                     'PageUp': 'pageup',
                     'Backspace': 'backspace',
                     'Delete': 'delete',
                     'Insert': 'insert',
                     'Escape': 'escape',
                     'Pause': 'pause',
                     'Select': 'select',
                     'Dead': 'dead',
                     'F1': 'f1',
                     'F2': 'f2',
                     'F3': 'f3',
                     'F4': 'f4',
                     'F5': 'f5',
                     'F6': 'f6',
                     'F7': 'f7',
                     'F8': 'f8',
                     'F9': 'f9',
                     'F10': 'f10',
                     'F11': 'f11',
                     'F12': 'f12'}

def _handle_key(key):
    """处理键盘事件中的键值"""
    # 从键名中提取键值
    value = key[key.index('k') + 1:]
    # 处理带有shift的特殊键
    if 'shift+' in key:
        if len(value) == 1:
            key = key.replace('shift+', '')
    # 根据映射表替换特殊键名
    if value in _SPECIAL_KEYS_LUT:
        value = _SPECIAL_KEYS_LUT[value]
    # 返回处理后的键值
    key = key[:key.index('k')] + value
    return key

class TimerTornado(backend_bases.TimerBase):
    """基于Tornado框架的定时器类"""
    def __init__(self, *args, **kwargs):
        self._timer = None  # 初始化定时器为None
        super().__init__(*args, **kwargs)  # 调用父类初始化函数
    # 启动定时器功能
    def _timer_start(self):
        import tornado  # 导入tornado库

        self._timer_stop()  # 停止当前的定时器（如果存在）

        if self._single:
            ioloop = tornado.ioloop.IOLoop.instance()  # 获取tornado的IOLoop实例
            # 添加一个单次定时器，在指定的时间间隔后执行self._on_timer方法
            self._timer = ioloop.add_timeout(
                datetime.timedelta(milliseconds=self.interval),
                self._on_timer)
        else:
            # 创建一个周期性定时器，以最大间隔时间（self.interval或1微秒）调用self._on_timer方法
            self._timer = tornado.ioloop.PeriodicCallback(
                self._on_timer,
                max(self.interval, 1e-6))
            self._timer.start()  # 启动定时器

    # 停止定时器功能
    def _timer_stop(self):
        import tornado  # 导入tornado库

        if self._timer is None:
            return  # 如果定时器为None，直接返回

        elif self._single:
            ioloop = tornado.ioloop.IOLoop.instance()  # 获取tornado的IOLoop实例
            ioloop.remove_timeout(self._timer)  # 移除单次定时器

        else:
            self._timer.stop()  # 停止周期性定时器

        self._timer = None  # 将定时器对象置为None，表示已停止

    # 设置定时器的间隔
    def _timer_set_interval(self):
        # 如果定时器已经启动，则先停止再重新启动
        if self._timer is not None:
            self._timer_stop()  # 停止定时器
            self._timer_start()  # 启动定时器
class TimerAsyncio(backend_bases.TimerBase):
    # 继承自 TimerBase 的异步定时器类

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置和关键字参数
        self._task = None
        super().__init__(*args, **kwargs)

    async def _timer_task(self, interval):
        # 异步定时任务，每隔一段时间执行一次
        while True:
            try:
                await asyncio.sleep(interval)  # 等待一段时间
                self._on_timer()  # 触发定时器事件

                if self._single:
                    break  # 如果是单次定时器，执行后就退出循环
            except asyncio.CancelledError:
                break  # 如果取消了定时器任务，退出循环

    def _timer_start(self):
        # 启动定时器
        self._timer_stop()  # 停止已有的定时器

        self._task = asyncio.ensure_future(
            self._timer_task(max(self.interval / 1_000., 1e-6))
        )  # 创建新的定时器任务并异步执行

    def _timer_stop(self):
        # 停止定时器
        if self._task is not None:
            self._task.cancel()  # 取消定时器任务
        self._task = None

    def _timer_set_interval(self):
        # 设置定时器的时间间隔
        # 如果定时器已经启动，先停止再重新启动
        if self._task is not None:
            self._timer_stop()
            self._timer_start()


class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    manager_class = _api.classproperty(lambda cls: FigureManagerWebAgg)
    _timer_cls = TimerAsyncio
    # Webagg and friends having the right methods, but still
    # having bugs in practice.  Do not advertise that it works until
    # we can debug this.
    supports_blit = False
    # Webagg及其相关方法理论上正确，但实际上还存在一些bug。
    # 在我们调试这些问题之前，不要宣传其正常工作。

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置和关键字参数
        super().__init__(*args, **kwargs)
        # 当渲染器包含比 PNG 缓冲区更新的数据时，设为 True
        self._png_is_old = True
        # 通过 "refresh" 消息设置为 True，以便下一个发送给客户端的帧将是完整帧
        self._force_full = True
        # 上一个缓冲区，用于差分模式
        self._last_buff = np.empty((0, 0))
        # 存储当前图像模式，以便客户端随时请求信息
        # 调用 self.set_image_mode(mode) 可以改变这个值，通知连接的客户端
        self._current_image_mode = 'full'
        # 跟踪鼠标事件以记录关键事件的 x、y 位置
        self._last_mouse_xy = (None, None)

    def show(self):
        # 显示图形窗口
        from matplotlib.pyplot import show
        show()

    def draw(self):
        # 绘制图形
        self._png_is_old = True  # 设为 True 表示 PNG 缓冲区已过时
        try:
            super().draw()  # 调用父类方法进行绘制
        finally:
            self.manager.refresh_all()  # 刷新所有管理器

    def blit(self, bbox=None):
        # 在指定的边界框内进行部分绘制
        self._png_is_old = True  # 设为 True 表示 PNG 缓冲区已过时
        self.manager.refresh_all()  # 刷新所有管理器

    def draw_idle(self):
        # 发送 "draw" 事件
        self.send_event("draw")
    def set_cursor(self, cursor):
        """
        Set the cursor type based on the provided cursor parameter and send
        the corresponding event to clients.

        Args:
            cursor (str): The cursor type to be set.

        Returns:
            None
        """
        # 使用 _api 检查并获取指定 cursor 对应的具体光标类型
        cursor = _api.check_getitem({
            backend_tools.Cursors.HAND: 'pointer',
            backend_tools.Cursors.POINTER: 'default',
            backend_tools.Cursors.SELECT_REGION: 'crosshair',
            backend_tools.Cursors.MOVE: 'move',
            backend_tools.Cursors.WAIT: 'wait',
            backend_tools.Cursors.RESIZE_HORIZONTAL: 'ew-resize',
            backend_tools.Cursors.RESIZE_VERTICAL: 'ns-resize',
        }, cursor=cursor)
        # 发送名为 'cursor' 的事件，指定 cursor 类型
        self.send_event('cursor', cursor=cursor)

    def set_image_mode(self, mode):
        """
        Set the mode for subsequent images to be sent to clients.

        Args:
            mode (str): The mode to set, either 'full' or 'diff'.

        Returns:
            None

        Raises:
            ValueError: If mode is not one of 'full' or 'diff'.
        """
        # 检查 mode 是否在允许的列表中
        _api.check_in_list(['full', 'diff'], mode=mode)
        # 如果当前的图像模式不等于指定的 mode，则更新当前图像模式
        if self._current_image_mode != mode:
            self._current_image_mode = mode
            # 处理发送图像模式改变的操作
            self.handle_send_image_mode(None)

    def get_diff_image(self):
        """
        Generate and return a difference image based on current and previous
        buffer states.

        Returns:
            bytes: PNG image data representing the difference image.
        """
        # 如果 PNG 已经过时，则获取当前渲染器并将其转换为像素数组
        if self._png_is_old:
            renderer = self.get_renderer()

            pixels = np.asarray(renderer.buffer_rgba())
            # 缓冲区的像素类型为 uint32，以便可以一次性比较整个像素，而不需要分别比较每个平面
            buff = pixels.view(np.uint32).squeeze(2)

            # 如果需要完全重绘（force_full 为真），或者像素具有透明度（需要使用完整模式）
            if (self._force_full
                    or buff.shape != self._last_buff.shape
                    or (pixels[:, :, 3] != 255).any()):
                self.set_image_mode('full')
                output = buff
            else:
                self.set_image_mode('diff')
                diff = buff != self._last_buff
                output = np.where(diff, buff, 0)

            # 存储当前缓冲区，以便计算下一个差异
            self._last_buff = buff.copy()
            self._force_full = False
            self._png_is_old = False

            # 将输出数据转换为 PNG 格式并返回其字节表示
            data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
            with BytesIO() as png:
                Image.fromarray(data).save(png, format="png")
                return png.getvalue()

    def handle_event(self, event):
        """
        Determine and invoke the appropriate handler method for the given event type.

        Args:
            event (dict): The event dictionary containing 'type' as the key.

        Returns:
            Depends on the specific handler method called.
        """
        # 获取事件类型
        e_type = event['type']
        # 根据事件类型动态获取对应的处理方法，若不存在则使用默认的处理方法
        handler = getattr(self, f'handle_{e_type}',
                          self.handle_unknown_event)
        return handler(event)

    def handle_unknown_event(self, event):
        """
        Handle an unknown event type by logging a warning.

        Args:
            event (dict): The event dictionary containing 'type' as the key.

        Returns:
            None
        """
        # 记录未处理消息类型的警告日志
        _log.warning('Unhandled message type %s. %s', event["type"], event)
    # 处理接收到的确认消息，用于网络流量双向传输，浏览器在接收到每个图像帧后发送一个“ack”消息
    def handle_ack(self, event):
        pass

    # 处理绘制事件，调用draw方法
    def handle_draw(self, event):
        self.draw()

    # 处理鼠标事件，获取鼠标位置和按钮信息，转换坐标系，触发相应的鼠标事件
    def _handle_mouse(self, event):
        x = event['x']
        y = event['y']
        y = self.get_renderer().height - y
        self._last_mouse_xy = x, y
        button = event['button'] + 1

        e_type = event['type']
        modifiers = event['modifiers']
        guiEvent = event.get('guiEvent')
        if e_type in ['button_press', 'button_release']:
            MouseEvent(e_type + '_event', self, x, y, button,
                       modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'dblclick':
            MouseEvent('button_press_event', self, x, y, button, dblclick=True,
                       modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'scroll':
            MouseEvent('scroll_event', self, x, y, step=event['step'],
                       modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'motion_notify':
            MouseEvent(e_type + '_event', self, x, y,
                       modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type in ['figure_enter', 'figure_leave']:
            LocationEvent(e_type + '_event', self, x, y,
                          modifiers=modifiers, guiEvent=guiEvent)._process()
    handle_button_press = handle_button_release = handle_dblclick = \
        handle_figure_enter = handle_figure_leave = handle_motion_notify = \
        handle_scroll = _handle_mouse

    # 处理键盘事件，触发相应的键盘事件
    def _handle_key(self, event):
        KeyEvent(event['type'] + '_event', self,
                 _handle_key(event['key']), *self._last_mouse_xy,
                 guiEvent=event.get('guiEvent'))._process()
    handle_key_press = handle_key_release = _handle_key

    # 处理工具栏按钮事件，调用工具栏对应的方法
    def handle_toolbar_button(self, event):
        getattr(self.toolbar, event['name'])()

    # 处理刷新事件，更新图形标签，强制全面刷新，设置工具栏历史按钮，触发绘制
    def handle_refresh(self, event):
        figure_label = self.figure.get_label()
        if not figure_label:
            figure_label = f"Figure {self.manager.num}"
        self.send_event('figure_label', label=figure_label)
        self._force_full = True
        if self.toolbar:
            self.toolbar.set_history_buttons()
        self.draw_idle()
    # 处理窗口大小改变事件，根据事件提供的宽度和高度调整图形大小
    def handle_resize(self, event):
        # 计算调整后的图形宽度（以设备像素比例调整）
        x = int(event.get('width', 800)) * self.device_pixel_ratio
        # 计算调整后的图形高度（以设备像素比例调整）
        y = int(event.get('height', 800)) * self.device_pixel_ratio
        # 获取当前图形对象
        fig = self.figure
        # 根据像素密度调整图形尺寸，以英寸为单位
        fig.set_size_inches(x / fig.dpi, y / fig.dpi, forward=False)
        # 标记PNG图像已过时，需更新视图器以匹配新的图形尺寸
        self._png_is_old = True
        # 通知图形管理器调整视图大小，不执行前向计算
        self.manager.resize(*fig.bbox.size, forward=False)
        # 处理图形大小改变事件，进一步处理
        ResizeEvent('resize_event', self)._process()
        # 更新图形显示
        self.draw_idle()

    # 处理发送图像模式请求事件，向客户端发送当前图像模式信息
    def handle_send_image_mode(self, event):
        # 向客户端发送当前图像模式信息请求
        self.send_event('image_mode', mode=self._current_image_mode)

    # 处理设置设备像素比率事件，根据事件设定的像素比率进行处理
    def handle_set_device_pixel_ratio(self, event):
        # 调用内部方法处理设置设备像素比率事件，获取设备像素比率，默认为1
        self._handle_set_device_pixel_ratio(event.get('device_pixel_ratio', 1))

    # 处理设置DPI比率事件，用于向后兼容旧版ipympl
    def handle_set_dpi_ratio(self, event):
        # 调用内部方法处理设置设备像素比率事件，获取DPI比率，默认为1
        self._handle_set_device_pixel_ratio(event.get('dpi_ratio', 1))

    # 处理设置设备像素比率的内部方法，根据提供的设备像素比率进行处理
    def _handle_set_device_pixel_ratio(self, device_pixel_ratio):
        # 如果设备像素比率发生变化，设置强制完整更新标志，并请求重新绘制图形
        if self._set_device_pixel_ratio(device_pixel_ratio):
            self._force_full = True
            self.draw_idle()

    # 发送自定义事件到图形管理器，通知特定事件类型和相关参数
    def send_event(self, event_type, **kwargs):
        # 如果存在图形管理器，发送指定事件类型和参数到管理器
        if self.manager:
            self.manager._send_event(event_type, **kwargs)
# 允许的工具项集合，包括一些常规操作和一个下载按钮，以及一个空值
_ALLOWED_TOOL_ITEMS = {
    'home',
    'back',
    'forward',
    'pan',
    'zoom',
    'download',
    None,
}

# 定义一个自定义的 Matplotlib WebAgg 导航工具栏类，继承自基类 NavigationToolbar2
class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):

    # 工具栏按钮项列表，包括标准工具栏项和额外的下载按钮
    toolitems = [
        (text, tooltip_text, image_file, name_of_method)
        for text, tooltip_text, image_file, name_of_method
        in (*backend_bases.NavigationToolbar2.toolitems,
            ('Download', 'Download plot', 'filesave', 'download'))
        if name_of_method in _ALLOWED_TOOL_ITEMS
    ]

    # 初始化方法，接受画布对象并设置空消息
    def __init__(self, canvas):
        self.message = ''
        super().__init__(canvas)

    # 设置消息方法，如果消息内容有变化则发送 "message" 事件
    def set_message(self, message):
        if message != self.message:
            self.canvas.send_event("message", message=message)
        self.message = message

    # 绘制选区方法，发送 "rubberband" 事件
    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.send_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    # 移除选区方法，发送 "rubberband" 事件来清除选区
    def remove_rubberband(self):
        self.canvas.send_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    # 保存图形方法，发送 'save' 事件
    def save_figure(self, *args):
        self.canvas.send_event('save')

    # 平移方法，调用基类的平移方法后发送 'navigate_mode' 事件
    def pan(self):
        super().pan()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    # 缩放方法，调用基类的缩放方法后发送 'navigate_mode' 事件
    def zoom(self):
        super().zoom()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    # 设置历史记录按钮状态方法，根据导航栈的状态发送 'history_buttons' 事件
    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        self.canvas.send_event('history_buttons',
                               Back=can_backward, Forward=can_forward)


# 定义一个自定义的 Matplotlib WebAgg 图形管理器类，继承自基类 FigureManagerBase
class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    # 必须设为 None 以避免破坏 ipympl
    _toolbar2_class = None
    # 指定使用的工具栏类为前面定义的 NavigationToolbar2WebAgg
    ToolbarCls = NavigationToolbar2WebAgg
    # 窗口标题
    _window_title = "Matplotlib"

    # 初始化方法，接受画布对象和编号，初始化 Web Socket 集合
    def __init__(self, canvas, num):
        self.web_sockets = set()
        super().__init__(canvas, num)

    # 不执行任何操作的 show 方法
    def show(self):
        pass

    # 调整大小方法，发送 'resize' 事件
    def resize(self, w, h, forward=True):
        self._send_event(
            'resize',
            size=(w / self.canvas.device_pixel_ratio,
                  h / self.canvas.device_pixel_ratio),
            forward=forward)

    # 设置窗口标题方法，发送 'figure_label' 事件，并更新窗口标题属性
    def set_window_title(self, title):
        self._send_event('figure_label', label=title)
        self._window_title = title

    # 获取窗口标题方法，返回当前窗口标题属性值
    def get_window_title(self):
        return self._window_title

    # 添加 Web Socket 方法，确保 Web Socket 对象有必要的方法，并发送 'refresh' 事件
    def add_web_socket(self, web_socket):
        assert hasattr(web_socket, 'send_binary')
        assert hasattr(web_socket, 'send_json')
        self.web_sockets.add(web_socket)
        self.resize(*self.canvas.figure.bbox.size)
        self._send_event('refresh')

    # 移除 Web Socket 方法，从集合中移除指定的 Web Socket 对象
    def remove_web_socket(self, web_socket):
        self.web_sockets.remove(web_socket)

    # 处理 JSON 数据方法，将其传递给画布对象处理
    def handle_json(self, content):
        self.canvas.handle_event(content)
    # 刷新所有连接的 WebSocket 实例，发送 Canvas 对象的差异图像
    def refresh_all(self):
        # 如果存在 WebSocket 实例
        if self.web_sockets:
            # 获取 Canvas 对象的差异图像
            diff = self.canvas.get_diff_image()
            # 如果差异图像不为空
            if diff is not None:
                # 遍历所有 WebSocket 实例，发送二进制差异图像数据
                for s in self.web_sockets:
                    s.send_binary(diff)

    @classmethod
    # 返回 JavaScript 脚本内容
    def get_javascript(cls, stream=None):
        # 如果未提供流对象，则创建一个 StringIO 对象作为输出流
        if stream is None:
            output = StringIO()
        else:
            output = stream

        # 将 mpl.js 文件的内容写入输出流
        output.write((Path(__file__).parent / "web_backend/js/mpl.js")
                     .read_text(encoding="utf-8"))

        # 获取工具栏项列表，将其转换为 JSON 字符串后写入输出流
        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(['', '', '', ''])
            else:
                toolitems.append([name, tooltip, image, method])
        output.write(f"mpl.toolbar_items = {json.dumps(toolitems)};\n\n")

        # 获取支持的文件扩展名列表，将其转换为 JSON 字符串后写入输出流
        extensions = []
        for filetype, ext in sorted(FigureCanvasWebAggCore.
                                    get_supported_filetypes_grouped().
                                    items()):
            extensions.append(ext[0])
        output.write(f"mpl.extensions = {json.dumps(extensions)};\n\n")

        # 写入默认文件扩展名到输出流
        output.write("mpl.default_extension = {};".format(
            json.dumps(FigureCanvasWebAggCore.get_default_filetype())))

        # 如果未提供流对象，则返回输出流的内容
        if stream is None:
            return output.getvalue()

    @classmethod
    # 返回静态文件路径，即 web_backend 目录的路径
    def get_static_file_path(cls):
        return os.path.join(os.path.dirname(__file__), 'web_backend')

    # 发送事件到所有 WebSocket 实例
    def _send_event(self, event_type, **kwargs):
        # 构建事件的 payload 对象
        payload = {'type': event_type, **kwargs}
        # 将 payload 对象以 JSON 格式发送到所有 WebSocket 实例
        for s in self.web_sockets:
            s.send_json(payload)
# 将 _BackendWebAggCoreAgg 类导出为模块的一部分，使其可被外部访问
@_Backend.export
# 定义 _BackendWebAggCoreAgg 类，继承自 _Backend
class _BackendWebAggCoreAgg(_Backend):
    # 设置 FigureCanvas 类型为 FigureCanvasWebAggCore
    FigureCanvas = FigureCanvasWebAggCore
    # 设置 FigureManager 类型为 FigureManagerWebAgg
    FigureManager = FigureManagerWebAgg
```