# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_nbagg.py`

```py
"""Interactive figures in the IPython notebook."""
# 导入所需的模块和类

from base64 import b64encode  # 导入base64编码函数
import io  # 导入io模块
import json  # 导入json模块
import pathlib  # 导入pathlib模块
import uuid  # 导入uuid模块

from ipykernel.comm import Comm  # 从ipykernel.comm中导入Comm类
from IPython.display import display, Javascript, HTML  # 从IPython.display中导入display、Javascript和HTML类

from matplotlib import is_interactive  # 从matplotlib中导入is_interactive函数
from matplotlib._pylab_helpers import Gcf  # 从matplotlib._pylab_helpers中导入Gcf类
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2  # 从matplotlib.backend_bases中导入_Backend、CloseEvent和NavigationToolbar2类
from .backend_webagg_core import (  # 从当前包的backend_webagg_core模块导入FigureCanvasWebAggCore、FigureManagerWebAgg和NavigationToolbar2WebAgg类
    FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg)
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
    TimerTornado, TimerAsyncio)


def connection_info():
    """
    Return a string showing the figure and connection status for the backend.

    This is intended as a diagnostic tool, and not for general use.
    """
    # 构造用于显示图形和连接状态的字符串列表
    result = [
        '{fig} - {socket}'.format(
            fig=(manager.canvas.figure.get_label()
                 or f"Figure {manager.num}"),
            socket=manager.web_sockets)
        for manager in Gcf.get_all_fig_managers()
    ]
    # 如果不是交互模式，添加待显示的图形数量信息
    if not is_interactive():
        result.append(f'Figures pending show: {len(Gcf.figs)}')
    return '\n'.join(result)


_FONT_AWESOME_CLASSES = {  # font-awesome 4 names
    'home': 'fa fa-home',
    'back': 'fa fa-arrow-left',
    'forward': 'fa fa-arrow-right',
    'zoom_to_rect': 'fa fa-square-o',
    'move': 'fa fa-arrows',
    'download': 'fa fa-floppy-o',
    None: None
}


class NavigationIPy(NavigationToolbar2WebAgg):
    """
    Custom navigation toolbar for IPython notebook.

    Extends NavigationToolbar2WebAgg with additional toolitems.
    """

    # 使用标准的工具栏项目 + 下载按钮
    toolitems = [(text, tooltip_text,
                  _FONT_AWESOME_CLASSES[image_file], name_of_method)
                 for text, tooltip_text, image_file, name_of_method
                 in (NavigationToolbar2.toolitems +
                     (('Download', 'Download plot', 'download', 'download'),))
                 if image_file in _FONT_AWESOME_CLASSES]


class FigureManagerNbAgg(FigureManagerWebAgg):
    """
    Figure manager for IPython notebook backend.

    Extends FigureManagerWebAgg to customize toolbar.
    """
    
    _toolbar2_class = ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        """
        Initialize the FigureManagerNbAgg instance.

        Args:
        - canvas: Instance of FigureCanvasWebAggCore or subclass.
        - num: Figure number.
        """
        self._shown = False
        super().__init__(canvas, num)

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a FigureManagerNbAgg instance with given canvas.

        Args:
        - canvas_class: Class of the canvas (e.g., FigureCanvasWebAggCore).
        - figure: Instance of matplotlib figure.
        - num: Figure number.

        Returns:
        - manager: Instance of FigureManagerNbAgg.
        """
        # 创建画布实例
        canvas = canvas_class(figure)
        # 创建管理器实例
        manager = cls(canvas, num)
        # 如果是交互模式，显示图形并绘制
        if is_interactive():
            manager.show()
            canvas.draw_idle()

        def destroy(event):
            """
            Event handler to destroy the manager and disconnect canvas.

            Args:
            - event: Close event.
            """
            canvas.mpl_disconnect(cid)
            Gcf.destroy(manager)

        # 连接关闭事件，调用destroy函数
        cid = canvas.mpl_connect('close_event', destroy)
        return manager

    def display_js(self):
        """
        Display JavaScript for IPython notebook integration.

        Uses IPython.display to show JavaScript.
        """
        # XXX 如何只执行一次？必须处理使用相同内核的多个浏览器实例（require.js - 但文件不是静态的？）。
        display(Javascript(FigureManagerNbAgg.get_javascript()))
    def show(self):
        # 如果图形尚未显示，则显示 JavaScript，并创建通信
        if not self._shown:
            self.display_js()  # 调用显示 JavaScript 方法
            self._create_comm()  # 创建通信
        else:
            self.canvas.draw_idle()  # 否则，仅更新画布
        self._shown = True  # 标记图形已显示
        # plt.figure 添加一个事件，使得显示后的图形成为活动图形。
        # 禁用此行为，避免即使在非交互模式下，图形仍然成为活动图形。
        if hasattr(self, '_cidgcf'):
            self.canvas.mpl_disconnect(self._cidgcf)  # 断开连接指定的事件
        if not is_interactive():
            from matplotlib._pylab_helpers import Gcf
            Gcf.figs.pop(self.num, None)  # 如果非交互模式，则移除图形

    def reshow(self):
        """
        一个特殊方法，重新在笔记本中显示图形。
        """
        self._shown = False  # 标记图形未显示
        self.show()  # 调用显示方法重新显示图形

    @property
    def connected(self):
        return bool(self.web_sockets)  # 返回是否有连接的 WebSocket

    @classmethod
    def get_javascript(cls, stream=None):
        if stream is None:
            output = io.StringIO()  # 如果未提供流对象，则创建一个字符串流
        else:
            output = stream
        super().get_javascript(stream=output)  # 调用父类方法获取 JavaScript
        output.write((pathlib.Path(__file__).parent
                      / "web_backend/js/nbagg_mpl.js")
                     .read_text(encoding="utf-8"))  # 将指定 JavaScript 文件内容写入流
        if stream is None:
            return output.getvalue()  # 如果未提供流对象，则返回流中的内容

    def _create_comm(self):
        comm = CommSocket(self)  # 创建通信对象
        self.add_web_socket(comm)  # 将通信对象添加到 WebSocket 集合中
        return comm  # 返回通信对象

    def destroy(self):
        self._send_event('close')  # 发送关闭事件
        # 需要复制通信对象列表，因为回调函数可能会修改此列表
        for comm in list(self.web_sockets):
            comm.on_close()  # 调用通信对象的关闭方法
        self.clearup_closed()  # 清理已关闭的通信对象

    def clearup_closed(self):
        """清理任何已关闭的通信。"""
        self.web_sockets = {socket for socket in self.web_sockets
                            if socket.is_open()}  # 更新仍然打开的 WebSocket 集合

        if len(self.web_sockets) == 0:
            CloseEvent("close_event", self.canvas)._process()  # 如果没有打开的 WebSocket，则处理关闭事件

    def remove_comm(self, comm_id):
        self.web_sockets = {socket for socket in self.web_sockets
                            if socket.comm.comm_id != comm_id}  # 移除指定通信 ID 的通信对象集合
class FigureCanvasNbAgg(FigureCanvasWebAggCore):
    manager_class = FigureManagerNbAgg



# FigureCanvasNbAgg 类，继承自 FigureCanvasWebAggCore 类，用于支持交互式的绘图
# 设置管理器类为 FigureManagerNbAgg
class FigureCanvasNbAgg(FigureCanvasWebAggCore):
    manager_class = FigureManagerNbAgg


class CommSocket:
    """
    Manages the Comm connection between IPython and the browser (client).

    Comms are 2 way, with the CommSocket being able to publish a message
    via the send_json method, and handle a message with on_message. On the
    JS side figure.send_message and figure.ws.onmessage do the sending and
    receiving respectively.

    """
    def __init__(self, manager):
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid.uuid4())
        # Publish an output area with a unique ID. The javascript can then
        # hook into this area.
        display(HTML("<div id=%r></div>" % self.uuid))
        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError as err:
            raise RuntimeError('Unable to create an IPython notebook Comm '
                               'instance. Are you in the IPython '
                               'notebook?') from err
        self.comm.on_msg(self.on_message)

        manager = self.manager
        self._ext_close = False

        def _on_close(close_message):
            self._ext_close = True
            manager.remove_comm(close_message['content']['comm_id'])
            manager.clearup_closed()

        self.comm.on_close(_on_close)

    def is_open(self):
        return not (self._ext_close or self.comm._closed)

    def on_close(self):
        # When the socket is closed, deregister the websocket with
        # the FigureManager.
        if self.is_open():
            try:
                self.comm.close()
            except KeyError:
                # apparently already cleaned it up?
                pass

    def send_json(self, content):
        # Send a JSON-encoded message through the CommSocket
        self.comm.send({'data': json.dumps(content)})

    def send_binary(self, blob):
        # Send a binary blob (e.g., image data) through the CommSocket
        if self.supports_binary:
            self.comm.send({'blob': 'image/png'}, buffers=[blob])
        else:
            # If binary is not supported, send the image data as a base64-encoded
            # data URL
            data = b64encode(blob).decode('ascii')
            data_uri = f"data:image/png;base64,{data}"
            self.comm.send({'data': data_uri})

    def on_message(self, message):
        # Handle incoming messages from the CommSocket

        # Parse the JSON message content
        message = json.loads(message['content']['data'])
        if message['type'] == 'closing':
            self.on_close()
            self.manager.clearup_closed()
        elif message['type'] == 'supports_binary':
            self.supports_binary = message['value']
        else:
            # Pass other messages along to the FigureManager
            self.manager.handle_json(message)


@_Backend.export
class _BackendNbAgg(_Backend):
    # Export _BackendNbAgg as part of the backend for matplotlib
    FigureCanvas = FigureCanvasNbAgg
    FigureManager = FigureManagerNbAgg



# _BackendNbAgg 类，作为 matplotlib 的后端之一，定义了 FigureCanvas 和 FigureManager
# 分别指定为 FigureCanvasNbAgg 和 FigureManagerNbAgg
@_Backend.export
class _BackendNbAgg(_Backend):
    FigureCanvas = FigureCanvasNbAgg
    FigureManager = FigureManagerNbAgg
```