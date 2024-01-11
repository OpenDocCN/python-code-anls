# `ZeroNet\src\Ui\UiServer.py`

```
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入 urllib 模块
import urllib
# 导入 socket 模块
import socket
# 导入 gevent 模块
import gevent
# 从 gevent.pywsgi 模块中导入 WSGIServer 类
from gevent.pywsgi import WSGIServer
# 从 lib.gevent_ws 模块中导入 WebSocketHandler 类
from lib.gevent_ws import WebSocketHandler
# 从当前目录下的 UiRequest 模块中导入 UiRequest 类
from .UiRequest import UiRequest
# 从 Site 模块中导入 SiteManager 类
from Site import SiteManager
# 从 Config 模块中导入 config 变量
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 导入 importlib 模块

# 如果不需要 WebSocket 处理程序，则跳过
class UiWSGIHandler(WebSocketHandler):

    def __init__(self, *args, **kwargs):
        # 保存服务器对象
        self.server = args[2]
        # 调用父类的构造函数
        super(UiWSGIHandler, self).__init__(*args, **kwargs)
        # 保存参数和关键字参数
        self.args = args
        self.kwargs = kwargs

    # 处理错误的方法
    def handleError(self, err):
        # 如果处于调试模式，则允许 WebSocket 错误出现在 /Debug 页面
        if config.debug:
            # 导入 main 模块
            import main
            # 调用 DebugHook.handleError() 方法
            main.DebugHook.handleError()
        else:
            # 创建 UiRequest 对象
            ui_request = UiRequest(self.server, {}, self.environ, self.start_response)
            # 生成错误页面的块
            block_gen = ui_request.error500("UiWSGIHandler error: %s" % Debug.formatExceptionMessage(err))
            # 遍历生成的块并写入
            for block in block_gen:
                self.write(block)

    # 运行应用程序的方法
    def run_application(self):
        # 错误名称
        err_name = "UiWSGIHandler websocket" if "HTTP_UPGRADE" in self.environ else "UiWSGIHandler"
        try:
            # 调用父类的 run_application() 方法
            super(UiWSGIHandler, self).run_application()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as err:
            # 记录连接错误
            logging.warning("%s connection error: %s" % (err_name, err))
        except Exception as err:
            # 记录错误
            logging.warning("%s error: %s" % (err_name, Debug.formatException(err)))
            # 处理错误
            self.handleError(err)

    # 处理请求的方法
    def handle(self):
        # 保存套接字以便在退出时正确关闭它们
        self.server.sockets[self.client_address] = self.socket
        # 调用父类的 handle() 方法
        super(UiWSGIHandler, self).handle()
        # 删除保存的套接字
        del self.server.sockets[self.client_address]

# UiServer 类
class UiServer:
    # 初始化方法，设置 IP 和端口，以及是否正在运行的标志
    def __init__(self):
        self.ip = config.ui_ip
        self.port = config.ui_port
        self.running = False
        # 如果 IP 是通配符，则将其设置为 "0.0.0.0"，表示绑定所有地址
        if self.ip == "*":
            self.ip = "0.0.0.0"  # Bind all
        # 如果配置了 UI 主机，则将其设置为允许的主机集合
        if config.ui_host:
            self.allowed_hosts = set(config.ui_host)
        # 如果 IP 是 "127.0.0.1"，则将允许的主机设置为一些特定的值
        elif config.ui_ip == "127.0.0.1":
            # IP 地址本身是允许的，因为它们不受 DNS 重绑定攻击的影响
            self.allowed_hosts = set(["zero", "localhost:%s" % config.ui_port])
            # 根据 RFC3986 规范，如果端口为空或与默认端口相同，则应省略端口部分
            # 因此，如果端口为 80，则需要支持无端口的主机
            if config.ui_port == 80:
                self.allowed_hosts.update(["localhost"])
        else:
            self.allowed_hosts = set([])
        # 允许的 WebSocket 源集合
        self.allowed_ws_origins = set()
        # 是否允许透明代理
        self.allow_trans_proxy = config.ui_trans_proxy

        # 一些用于安全性的变量
        self.wrapper_nonces = []
        self.add_nonces = []
        self.websockets = []
        # 站点管理器
        self.site_manager = SiteManager.site_manager
        # 站点列表
        self.sites = SiteManager.site_manager.list()
        # 日志记录器
        self.log = logging.getLogger(__name__)
        # 设置错误日志记录器的处理方法
        config.error_logger.onNewRecord = self.handleErrorLogRecord

    # 处理错误日志记录
    def handleErrorLogRecord(self, record):
        # 更新 WebSocket，记录日志事件的级别
        self.updateWebsocket(log_event=record.levelname)

    # WebUI 启动后执行的方法
    def afterStarted(self):
        # 设置系统最大打开文件数
        from util import Platform
        Platform.setMaxfilesopened(config.max_files_opened)

    # 处理 WSGI 请求
    # 处理请求的方法，接收环境变量、开始响应的函数作为参数
    def handleRequest(self, env, start_response):
        # 获取请求路径并解码为 UTF-8 格式
        path = bytes(env["PATH_INFO"], "raw-unicode-escape").decode("utf8")
        # 如果有查询字符串，则解析为字典
        if env.get("QUERY_STRING"):
            get = dict(urllib.parse.parse_qsl(env['QUERY_STRING']))
        else:
            get = {}
        # 创建 UiRequest 对象，传入参数并开始响应
        ui_request = UiRequest(self, get, env, start_response)
        # 如果处于调试模式
        if config.debug:  # Let the exception catched by werkezung
            # 调用 UiRequest 对象的路由方法
            return ui_request.route(path)
        else:  # Catch and display the error
            # 捕获并显示错误
            try:
                # 调用 UiRequest 对象的路由方法
                return ui_request.route(path)
            except Exception as err:
                # 记录错误日志并返回 500 错误页面
                logging.debug("UiRequest error: %s" % Debug.formatException(err))
                return ui_request.error500("Err: %s" % Debug.formatException(err))

    # 重新加载 UiRequest 类以防止在调试模式下重启
    def reload(self):
        # 使用全局变量重新加载 UiRequest 类
        global UiRequest
        import imp
        import sys
        importlib.reload(sys.modules["User.UserManager"])
        importlib.reload(sys.modules["Ui.UiWebsocket"])
        UiRequest = imp.load_source("UiRequest", "src/Ui/UiRequest.py").UiRequest
        # UiRequest.reload()

    # 绑定并运行服务器
    # 停止服务器的方法
    def stop(self):
        # 记录调试信息
        self.log.debug("Stopping...")
        # 关闭 WebSocket 连接
        if "clients" in dir(self.server):
            for client in list(self.server.clients.values()):
                client.ws.close()
        # 关闭 HTTP 连接
        sock_closed = 0
        for sock in list(self.server.sockets.values()):
            try:
                # 发送关闭消息
                sock.send(b"bye")
                # 关闭 socket 的读写
                sock.shutdown(socket.SHUT_RDWR)
                # 关闭 socket
                sock_closed += 1
            except Exception as err:
                # 记录 HTTP 连接关闭错误
                self.log.debug("Http connection close error: %s" % err)
        # 记录关闭的 socket 数量
        self.log.debug("Socket closed: %s" % sock_closed)
        # 等待一段时间
        time.sleep(0.1)
        # 如果处于调试模式
        if config.debug:
            # 导入 DebugReloader 模块
            from Debug import DebugReloader
            # 停止调试重载器
            DebugReloader.watcher.stop()

        # 关闭服务器的 socket
        self.server.socket.close()
        # 停止服务器
        self.server.stop()
        # 标记服务器已停止
        self.running = False
        # 等待一段时间
        time.sleep(1)

    # 更新 WebSocket 连接
    def updateWebsocket(self, **kwargs):
        # 如果有参数
        if kwargs:
            # 将参数转换为字典
            param = {"event": list(kwargs.items())[0]}
        else:
            param = None

        # 遍历所有 WebSocket 连接
        for ws in self.websockets:
            # 发送服务器变化事件
            ws.event("serverChanged", param)
```