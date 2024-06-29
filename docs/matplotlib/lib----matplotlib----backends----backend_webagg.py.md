# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_webagg.py`

```
"""Displays Agg images in the browser, with interactivity."""

# The WebAgg backend is divided into two modules:
#
# - `backend_webagg_core.py` contains code necessary to embed a WebAgg
#   plot inside of a web application, and communicate in an abstract
#   way over a web socket.
#
# - `backend_webagg.py` contains a concrete implementation of a basic
#   application, implemented with tornado.

from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading

try:
    import tornado
except ImportError as err:
    raise RuntimeError("The WebAgg backend requires Tornado.") from err

import tornado.web
import tornado.ioloop
import tornado.websocket

import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
    TimerAsyncio, TimerTornado)


# 创建一个线程，用于启动 Tornado 的 IOLoop 实例
webagg_server_thread = threading.Thread(
    target=lambda: tornado.ioloop.IOLoop.instance().start())


class FigureManagerWebAgg(core.FigureManagerWebAgg):
    _toolbar2_class = core.NavigationToolbar2WebAgg

    @classmethod
    def pyplot_show(cls, *, block=None):
        # 初始化 WebAgg 应用程序
        WebAggApplication.initialize()

        # 构建访问 URL
        url = "http://{address}:{port}{prefix}".format(
            address=WebAggApplication.address,
            port=WebAggApplication.port,
            prefix=WebAggApplication.url_prefix)

        # 如果配置允许在浏览器中自动打开，则尝试打开浏览器
        if mpl.rcParams['webagg.open_in_browser']:
            import webbrowser
            if not webbrowser.open(url):
                print(f"To view figure, visit {url}")
        else:
            print(f"To view figure, visit {url}")

        # 启动 WebAgg 应用程序
        WebAggApplication.start()


class FigureCanvasWebAgg(core.FigureCanvasWebAggCore):
    manager_class = FigureManagerWebAgg


class WebAggApplication(tornado.web.Application):
    initialized = False
    started = False

    # 处理获取网站图标请求的处理器
    class FavIcon(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Content-Type', 'image/png')
            self.write(Path(mpl.get_data_path(),
                            'images/matplotlib.png').read_bytes())

    # 处理获取单个图形页面请求的处理器
    class SingleFigurePage(tornado.web.RequestHandler):
        def __init__(self, application, request, *, url_prefix='', **kwargs):
            self.url_prefix = url_prefix
            super().__init__(application, request, **kwargs)

        def get(self, fignum):
            # 解析图形编号
            fignum = int(fignum)
            # 获取对应图形管理器
            manager = Gcf.get_fig_manager(fignum)

            # 构建 WebSocket 的 URI
            ws_uri = f'ws://{self.request.host}{self.url_prefix}/'
            
            # 渲染单个图形的 HTML 页面，传入所需参数
            self.render(
                "single_figure.html",
                prefix=self.url_prefix,
                ws_uri=ws_uri,
                fig_id=fignum,
                toolitems=core.NavigationToolbar2WebAgg.toolitems,
                canvas=manager.canvas)
    class AllFiguresPage(tornado.web.RequestHandler):
        # 定义一个处理所有图形页面的请求处理器类
        def __init__(self, application, request, *, url_prefix='', **kwargs):
            # 初始化方法，设置URL前缀，并调用父类的初始化方法
            self.url_prefix = url_prefix
            super().__init__(application, request, **kwargs)

        def get(self):
            # 处理GET请求，构建WebSocket的URI地址
            ws_uri = f'ws://{self.request.host}{self.url_prefix}/'
            # 渲染HTML页面并传入必要的参数
            self.render(
                "all_figures.html",
                prefix=self.url_prefix,
                ws_uri=ws_uri,
                figures=sorted(Gcf.figs.items()),  # 将图形对象字典按键排序
                toolitems=core.NavigationToolbar2WebAgg.toolitems)  # 获取WebAgg工具条项目

    class MplJs(tornado.web.RequestHandler):
        # 处理返回Matplotlib JavaScript脚本的请求处理器类
        def get(self):
            # 设置响应头的Content-Type为application/javascript
            self.set_header('Content-Type', 'application/javascript')
            # 获取FigureManagerWebAgg生成的JavaScript内容
            js_content = core.FigureManagerWebAgg.get_javascript()
            # 将JavaScript内容写入响应体
            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        # 处理下载图形文件的请求处理器类
        def get(self, fignum, fmt):
            # 将传入的图形编号转换为整数类型
            fignum = int(fignum)
            # 获取对应图形的图形管理器
            manager = Gcf.get_fig_manager(fignum)
            # 设置响应头的Content-Type为指定格式的MIME类型，若未知则为binary
            self.set_header(
                'Content-Type', mimetypes.types_map.get(fmt, 'binary'))
            # 创建一个字节流缓冲区
            buff = BytesIO()
            # 将图形保存为指定格式的数据，并写入字节流缓冲区
            manager.canvas.figure.savefig(buff, format=fmt)
            # 将字节流缓冲区的数据作为响应体内容写入响应
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        # 处理WebSocket连接的请求处理器类
        supports_binary = True  # WebSocket是否支持二进制数据

        def open(self, fignum):
            # 在WebSocket连接建立时调用，初始化WebSocket连接
            self.fignum = int(fignum)  # 将传入的图形编号转换为整数类型
            # 获取对应图形的图形管理器
            self.manager = Gcf.get_fig_manager(self.fignum)
            # 将WebSocket连接添加到图形管理器的WebSocket集合中
            self.manager.add_web_socket(self)
            # 设置TCP_NODELAY选项以提高WebSocket连接性能
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            # 当WebSocket连接关闭时调用，从图形管理器的WebSocket集合中移除连接
            self.manager.remove_web_socket(self)

        def on_message(self, message):
            # 处理接收到的WebSocket消息
            message = json.loads(message)  # 解析收到的JSON格式消息
            # 消息类型为'supports_binary'时，更新supports_binary属性
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = Gcf.get_fig_manager(self.fignum)
                # 处理图形相关的JSON消息
                # 当图形管理器不为None时，处理JSON消息
                if manager is not None:
                    manager.handle_json(message)

        def send_json(self, content):
            # 发送JSON格式的消息到WebSocket连接
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            # 发送二进制数据到WebSocket连接
            if self.supports_binary:
                self.write_message(blob, binary=True)  # 如果支持二进制数据，则直接发送
            else:
                # 否则将二进制数据编码为Base64格式后发送
                data_uri = "data:image/png;base64,{}".format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)
    # 初始化函数，用于设置 URL 前缀
    def __init__(self, url_prefix=''):
        # 如果提供了 URL 前缀，则进行以下断言检查
        if url_prefix:
            # 确保 URL 前缀以 '/' 开头且不以 '/' 结尾
            assert url_prefix[0] == '/' and url_prefix[-1] != '/', \
                'url_prefix must start with a "/" and not end with one.'

        # 调用父类初始化函数
        super().__init__(
            [
                # 处理静态文件的路由，用于 CSS 和 JS
                (url_prefix + r'/_static/(.*)',
                 tornado.web.StaticFileHandler,
                 {'path': core.FigureManagerWebAgg.get_static_file_path()}),

                # 处理工具栏静态图片的路由
                (url_prefix + r'/_images/(.*)',
                 tornado.web.StaticFileHandler,
                 {'path': Path(mpl.get_data_path(), 'images')}),

                # Matplotlib 的网站图标
                (url_prefix + r'/favicon.ico', self.FavIcon),

                # 包含所有部件的页面
                (url_prefix + r'/([0-9]+)', self.SingleFigurePage,
                 {'url_prefix': url_prefix}),

                # 包含所有图形的页面
                (url_prefix + r'/?', self.AllFiguresPage,
                 {'url_prefix': url_prefix}),

                # 提供 mpl.js 脚本的路由
                (url_prefix + r'/js/mpl.js', self.MplJs),

                # 处理与浏览器之间的图像和事件传输，并接收来自浏览器的事件
                (url_prefix + r'/([0-9]+)/ws', self.WebSocket),

                # 处理静态图像下载（保存）的路由
                (url_prefix + r'/([0-9]+)/download.([a-z0-9.]+)',
                 self.Download),
            ],
            # 设置模板路径为静态文件路径
            template_path=core.FigureManagerWebAgg.get_static_file_path())

    @classmethod
    # 初始化方法，用于设置 WebAgg 服务器的参数和启动流程
    def initialize(cls, url_prefix='', port=None, address=None):
        # 如果已经初始化过，则直接返回，避免重复初始化
        if cls.initialized:
            return

        # 创建类的实例
        app = cls(url_prefix=url_prefix)

        # 设置类属性 url_prefix
        cls.url_prefix = url_prefix

        # 使用 IPython 中的随机端口选择算法
        # 生成一个靠近给定端口的随机端口列表
        def random_ports(port, n):
            """
            生成一个靠近给定端口的随机端口列表。

            前 5 个端口是连续的，剩余的 n-5 个端口在 [port-2*n, port+2*n] 范围内随机选择。
            """
            for i in range(min(5, n)):
                yield port + i
            for i in range(n - 5):
                yield port + random.randint(-2 * n, 2 * n)

        # 如果 address 为 None，则使用 matplotlib 的默认地址
        if address is None:
            cls.address = mpl.rcParams['webagg.address']
        else:
            cls.address = address
        
        # 获取 matplotlib 配置中的默认端口
        cls.port = mpl.rcParams['webagg.port']
        
        # 尝试在随机选定的端口上监听应用
        for port in random_ports(cls.port,
                                 mpl.rcParams['webagg.port_retries']):
            try:
                app.listen(port, cls.address)
            except OSError as e:
                # 如果端口已被占用，则捕获异常
                if e.errno != errno.EADDRINUSE:
                    raise
            else:
                # 成功监听后设置类属性 port，并跳出循环
                cls.port = port
                break
        else:
            # 如果所有尝试都失败，则抛出 SystemExit 异常
            raise SystemExit(
                "The webagg server could not be started because an available "
                "port could not be found")

        # 设置初始化标志为 True
        cls.initialized = True

    @classmethod
    def start(cls):
        import asyncio
        
        # 检查是否已经有运行中的事件循环
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            cls.started = True

        # 如果已经启动过，则直接返回
        if cls.started:
            return

        """
        IOLoop.running() was removed as of Tornado 2.4; see for example
        https://groups.google.com/forum/#!topic/python-tornado/QLMzkpQBGOY
        Thus there is no correct way to check if the loop has already been
        launched. We may end up with two concurrently running loops in that
        unlucky case with all the expected consequences.
        """
        # 获取 Tornado 的 IOLoop 实例
        ioloop = tornado.ioloop.IOLoop.instance()

        # 定义关闭服务器的方法
        def shutdown():
            ioloop.stop()
            # 打印服务器停止信息
            print("Server is stopped")
            sys.stdout.flush()
            # 设置 started 标志为 False
            cls.started = False

        # 定义捕获 SIGINT 信号的上下文管理器
        @contextmanager
        def catch_sigint():
            # 备份旧的信号处理程序，并设置新的处理程序
            old_handler = signal.signal(
                signal.SIGINT,
                lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
            try:
                yield
            finally:
                # 恢复旧的信号处理程序
                signal.signal(signal.SIGINT, old_handler)

        # 在阻塞式地运行事件循环前，将 started 标志设为 True
        cls.started = True

        # 打印提示信息，告知用户如何停止 WebAgg 服务器
        print("Press Ctrl+C to stop WebAgg server")
        sys.stdout.flush()

        # 使用捕获 SIGINT 信号的上下文管理器来运行事件循环
        with catch_sigint():
            ioloop.start()
# 定义一个函数用于在 IPython 中内联显示图形
def ipython_inline_display(figure):
    # 导入 Tornado 模板库，用于后续生成 HTML 内容
    import tornado.template

    # 初始化 WebAgg 应用程序
    WebAggApplication.initialize()

    # 异步处理：检查当前是否有正在运行的事件循环，如果没有则启动 WebAgg 服务器线程
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 如果 WebAgg 服务器线程没有在运行，则启动它
        if not webagg_server_thread.is_alive():
            webagg_server_thread.start()

    # 获取图形的编号
    fignum = figure.number

    # 读取静态文件路径中的 IPython 内联图形模板文件内容
    tpl = Path(core.FigureManagerWebAgg.get_static_file_path(),
               "ipython_inline_figure.html").read_text()

    # 使用 Tornado 模板创建模板对象
    t = tornado.template.Template(tpl)

    # 生成并返回 HTML 内容，传入所需参数
    return t.generate(
        prefix=WebAggApplication.url_prefix,
        fig_id=fignum,
        toolitems=core.NavigationToolbar2WebAgg.toolitems,
        canvas=figure.canvas,
        port=WebAggApplication.port).decode('utf-8')


# 导出 _Backend 类作为 WebAgg 后端的一个实现
@_Backend.export
class _BackendWebAgg(_Backend):
    # 设置图形画布为 FigureCanvasWebAgg 类
    FigureCanvas = FigureCanvasWebAgg
    # 设置图形管理器为 FigureManagerWebAgg 类
    FigureManager = FigureManagerWebAgg
```