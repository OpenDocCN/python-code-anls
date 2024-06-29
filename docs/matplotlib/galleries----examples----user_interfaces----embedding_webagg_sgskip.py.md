# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_webagg_sgskip.py`

```
# 导入必要的模块和库
import argparse  # 解析命令行参数
import io  # 用于流处理
import json  # 处理 JSON 数据
import mimetypes  # 获取文件的 MIME 类型
from pathlib import Path  # 处理路径
import signal  # 处理信号
import socket  # 实现网络通信

try:
    import tornado  # 导入 Tornado 框架
except ImportError as err:
    raise RuntimeError("This example requires tornado.") from err  # 抛出导入错误

import tornado.httpserver  # Tornado HTTP 服务器
import tornado.ioloop  # Tornado I/O 循环
import tornado.web  # Tornado Web 框架
import tornado.websocket  # Tornado WebSocket 模块

import numpy as np  # 导入 NumPy 数组处理库

import matplotlib as mpl  # 导入 Matplotlib 库
from matplotlib.backends.backend_webagg import (
    FigureManagerWebAgg, new_figure_manager_given_figure)  # 导入 WebAgg 后端相关类
from matplotlib.figure import Figure  # 导入 Figure 类


def create_figure():
    """
    创建一个简单的示例图形。
    """
    fig = Figure()  # 创建一个 Figure 对象
    ax = fig.add_subplot()  # 在 Figure 对象上添加一个子图
    t = np.arange(0.0, 3.0, 0.01)  # 创建一个时间序列
    s = np.sin(2 * np.pi * t)  # 计算正弦波
    ax.plot(t, s)  # 绘制正弦波
    return fig  # 返回创建的图形对象


# 以下是网页的内容。通常情况下，应使用网页框架中的模板功能来生成此类内容。
html_content = """<!DOCTYPE html>
<html lang="en">
  <head>
    <!-- TODO: There should be a way to include all of the required javascript
               and CSS so matplotlib can add to the set in the future if it
               needs to. -->
    <link rel="stylesheet" href="_static/css/page.css" type="text/css">  # 导入页面样式表
    <link rel="stylesheet" href="_static/css/boilerplate.css" type="text/css">  # 导入样板样式表
    <link rel="stylesheet" href="_static/css/fbm.css" type="text/css">  # 导入 fbm 样式表
    <link rel="stylesheet" href="_static/css/mpl.css" type="text/css">  # 导入 Matplotlib 样式表
    <script src="mpl.js"></script>  # 导入 Matplotlib 的 JavaScript 文件
    <script>
          /* This is a callback that is called when the user saves
             (downloads) a file.  Its purpose is really to map from a
             figure and file format to a url in the application. */
          function ondownload(figure, format) {
            // 打开一个新窗口来下载特定格式的文件
            window.open('download.' + format, '_blank');
          };
    
          function ready(fn) {
            if (document.readyState != "loading") {
              // 如果文档已经加载完毕，直接执行传入的函数
              fn();
            } else {
              // 否则，等待DOMContentLoaded事件后执行传入的函数
              document.addEventListener("DOMContentLoaded", fn);
            }
          }
    
          ready(
            function() {
              /* It is up to the application to provide a websocket that the figure
                 will use to communicate to the server.  This websocket object can
                 also be a "fake" websocket that underneath multiplexes messages
                 from multiple figures, if necessary. */
              // 获取websocket类型（真实或模拟），用于与服务器通信
              var websocket_type = mpl.get_websocket_type();
              // 创建websocket对象，连接到指定的URI
              var websocket = new websocket_type("%(ws_uri)sws");
    
              // mpl.figure在网页上创建一个新的图形
              var fig = new mpl.figure(
                  // 图形的唯一数字标识符
                  %(fig_id)s,
                  // websocket对象（或行为类似的对象）
                  websocket,
                  // 当选择文件类型以下载时调用的函数
                  ondownload,
                  // 放置图形的HTML元素
                  document.getElementById("figure"));
            }
          );
        </script>
    
        <title>matplotlib</title>
      </head>
    
      <body>
        <div id="figure">
        </div>
      </body>
"""
</html>
"""

class MyApplication(tornado.web.Application):
    """
    Tornado web application for serving different types of requests.
    """

    class MainPage(tornado.web.RequestHandler):
        """
        RequestHandler serving the main HTML page.

        Handles GET requests and generates HTML content using `html_content`
        template, replacing placeholders with WebSocket URI and figure ID.
        """

        def get(self):
            manager = self.application.manager
            ws_uri = f"ws://{self.request.host}/"
            content = html_content % {
                "ws_uri": ws_uri, "fig_id": manager.num}
            self.write(content)

    class MplJs(tornado.web.RequestHandler):
        """
        RequestHandler serving dynamically generated matplotlib JavaScript.

        Handles GET requests and sets 'Content-Type' header to 'application/javascript'.
        Generates JavaScript content using `FigureManagerWebAgg.get_javascript()`.
        """

        def get(self):
            self.set_header('Content-Type', 'application/javascript')
            js_content = FigureManagerWebAgg.get_javascript()
            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        """
        RequestHandler for downloading figures in various formats.

        Handles GET requests with 'fmt' parameter specifying the file format.
        Sets 'Content-Type' header based on the requested format.
        Saves the figure to a BytesIO buffer and writes its contents.
        """

        def get(self, fmt):
            manager = self.application.manager
            self.set_header(
                'Content-Type', mimetypes.types_map.get(fmt, 'binary'))
            buff = io.BytesIO()
            manager.canvas.figure.savefig(buff, format=fmt)
            self.write(buff.getvalue())
    class WebSocket(tornado.websocket.WebSocketHandler):
        """
        A websocket for interactive communication between the plot in
        the browser and the server.
    
        In addition to the methods required by tornado, it is required to
        have two callback methods:
    
            - ``send_json(json_content)`` is called by matplotlib when
              it needs to send json to the browser.  `json_content` is
              a JSON tree (Python dictionary), and it is the responsibility
              of this implementation to encode it as a string to send over
              the socket.
    
            - ``send_binary(blob)`` is called to send binary image data
              to the browser.
        """
        supports_binary = True  # WebSocket是否支持二进制数据传输
    
        def open(self):
            # Register the websocket with the FigureManager.
            manager = self.application.manager
            manager.add_web_socket(self)  # 将当前websocket注册到FigureManager中
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)  # 设置TCP连接的TCP_NODELAY选项，禁用Nagle算法
    
        def on_close(self):
            # When the socket is closed, deregister the websocket with
            # the FigureManager.
            manager = self.application.manager
            manager.remove_web_socket(self)  # 从FigureManager中注销当前websocket
    
        def on_message(self, message):
            # The 'supports_binary' message is relevant to the
            # websocket itself.  The other messages get passed along
            # to matplotlib as-is.
    
            # Every message has a "type" and a "figure_id".
            message = json.loads(message)  # 解析收到的消息为JSON格式
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']  # 更新supports_binary属性为收到的值
            else:
                manager = self.application.manager
                manager.handle_json(message)  # 将收到的消息传递给FigureManager处理
    
        def send_json(self, content):
            self.write_message(json.dumps(content))  # 将JSON内容转为字符串并发送给客户端
    
        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)  # 如果支持二进制数据传输，则直接发送二进制数据
            else:
                data_uri = ("data:image/png;base64," +
                            blob.encode('base64').replace('\n', ''))  # 将二进制数据转为Base64编码的数据URI
                self.write_message(data_uri)  # 发送Base64编码的数据URI给客户端
    # 初始化函数，接收一个图形对象作为参数
    def __init__(self, figure):
        # 将传入的图形对象保存到实例变量中
        self.figure = figure
        # 使用给定的图形对象创建新的图形管理器，并保存到实例变量中
        self.manager = new_figure_manager_given_figure(id(figure), figure)

        # 调用父类的初始化函数，传入一个包含多个路由处理器的列表
        super().__init__([
            # 处理静态文件（CSS 和 JS）
            (r'/_static/(.*)',
             tornado.web.StaticFileHandler,
             {'path': FigureManagerWebAgg.get_static_file_path()}),

            # 处理工具栏的静态图像文件
            (r'/_images/(.*)',
             tornado.web.StaticFileHandler,
             {'path': Path(mpl.get_data_path(), 'images')}),

            # 主页面路由处理器
            ('/', self.MainPage),

            # 处理mpl.js文件的路由
            ('/mpl.js', self.MplJs),

            # 处理与浏览器之间图片和事件的传输，以及接收来自浏览器的事件
            ('/ws', self.WebSocket),

            # 处理静态图片下载（保存）的路由
            (r'/download.([a-z0-9.]+)', self.Download),
        ])
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument('-p', '--port', type=int, default=8080,
                        help='Port to listen on (0 for a random port).')
    # 添加一个命令行参数选项：'-p' 或 '--port'，指定监听的端口号，默认为 8080

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    figure = create_figure()
    # 调用 create_figure() 函数创建图形对象 figure

    application = MyApplication(figure)
    # 使用图形对象 figure 创建 MyApplication 类的实例 application

    http_server = tornado.httpserver.HTTPServer(application)
    # 使用 application 创建 Tornado 的 HTTPServer 实例 http_server

    sockets = tornado.netutil.bind_sockets(args.port, '')
    # 绑定指定端口 args.port 的套接字列表，并存储到 sockets 变量中

    http_server.add_sockets(sockets)
    # 将套接字列表添加到 HTTPServer 实例中

    for s in sockets:
        addr, port = s.getsockname()[:2]
        # 获取每个套接字的本地地址和端口号

        if s.family is socket.AF_INET6:
            addr = f'[{addr}]'
            # 如果是 IPv6 地址，格式化成带方括号的形式

        print(f"Listening on http://{addr}:{port}/")
        # 打印监听的地址和端口号信息

    print("Press Ctrl+C to quit")
    # 提示用户按 Ctrl+C 来退出程序

    ioloop = tornado.ioloop.IOLoop.instance()
    # 获取 Tornado 的 IOLoop 实例对象

    def shutdown():
        ioloop.stop()
        # 定义关闭服务的函数，停止 IOLoop 循环
        print("Server stopped")

    old_handler = signal.signal(
        signal.SIGINT,
        lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
    # 设置 SIGINT 信号处理函数，使用 lambda 函数添加回调函数到 IOLoop 循环中

    try:
        ioloop.start()
        # 开始 Tornado 的 IOLoop 循环
    finally:
        signal.signal(signal.SIGINT, old_handler)
        # 恢复 SIGINT 信号的原始处理函数
```