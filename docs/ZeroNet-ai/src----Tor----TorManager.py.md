# `ZeroNet\src\Tor\TorManager.py`

```
# 导入日志模块
import logging
# 导入正则表达式模块
import re
# 导入套接字模块
import socket
# 导入二进制转换模块
import binascii
# 导入系统模块
import sys
# 导入操作系统模块
import os
# 导入时间模块
import time
# 导入随机数模块
import random
# 导入子进程模块
import subprocess
# 导入退出处理模块
import atexit

# 导入协程模块
import gevent

# 从Config模块中导入config对象
from Config import config
# 从Crypt模块中导入CryptRsa类
from Crypt import CryptRsa
# 从Site模块中导入SiteManager类
from Site import SiteManager
# 导入socks代理模块
import socks
# 从协程锁模块中导入RLock类
from gevent.lock import RLock
# 从Debug模块中导入Debug类
from Debug import Debug
# 从Plugin模块中导入PluginManager类

# 使用PluginManager.acceptPlugins装饰TorManager类
@PluginManager.acceptPlugins
class TorManager(object):
    # 初始化TorManager类
    def __init__(self, fileserver_ip=None, fileserver_port=None):
        # 初始化私钥字典
        self.privatekeys = {}  # Onion: Privatekey
        # 初始化站点onion地址字典
        self.site_onions = {}  # Site address: Onion
        # 设置tor执行文件路径
        self.tor_exe = "tools/tor/tor.exe"
        # 检查是否有meek桥接
        self.has_meek_bridges = os.path.isfile("tools/tor/PluggableTransports/meek-client.exe")
        # 初始化tor进程为None
        self.tor_process = None
        # 获取TorManager类的日志对象
        self.log = logging.getLogger("TorManager")
        # 初始化启动onion列表为None
        self.start_onions = None
        # 初始化连接为None
        self.conn = None
        # 初始化锁对象
        self.lock = RLock()
        # 设置启动状态为True
        self.starting = True
        # 设置连接状态为True
        self.connecting = True
        # 初始化状态为None
        self.status = None
        # 初始化事件开始为异步结果对象
        self.event_started = gevent.event.AsyncResult()

        # 如果配置中tor为"disable"
        if config.tor == "disable":
            # 设置启用状态为False
            self.enabled = False
            # 设置启动onions为False
            self.start_onions = False
            # 设置状态为"Disabled"
            self.setStatus("Disabled")
        else:
            # 设置启用状态为True
            self.enabled = True
            # 设置状态为"Waiting"
            self.setStatus("Waiting")

        # 如果fileserver_port存在，则使用fileserver_port，否则使用config中的fileserver_port
        if fileserver_port:
            self.fileserver_port = fileserver_port
        else:
            self.fileserver_port = config.fileserver_port

        # 从配置中获取tor控制器的IP和端口
        self.ip, self.port = config.tor_controller.rsplit(":", 1)
        self.port = int(self.port)

        # 从配置中获取tor代理的IP和端口
        self.proxy_ip, self.proxy_port = config.tor_proxy.rsplit(":", 1)
        self.proxy_port = int(self.proxy_port)
    # 启动函数，记录调试信息，设置启动标志
    def start(self):
        self.log.debug("Starting (Tor: %s)" % config.tor)
        self.starting = True
        try:
            # 尝试连接到代理
            if not self.connect():
                raise Exception(self.status)
            # 记录调试信息
            self.log.debug("Tor proxy port %s check ok" % config.tor_proxy)
        except Exception as err:
            # 处理连接异常
            if sys.platform.startswith("win") and os.path.isfile(self.tor_exe):
                # 如果是 Windows 平台且存在 Tor 可执行文件，则启动自包含的 Tor
                self.log.info("Starting self-bundled Tor, due to Tor proxy port %s check error: %s" % (config.tor_proxy, err))
                # 更改为自包含 Tor 的端口
                self.port = 49051
                self.proxy_port = 49050
                # 如果配置为始终使用 Tor，则设置代理
                if config.tor == "always":
                    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", self.proxy_port)
                self.enabled = True
                # 如果连接失败，则启动 Tor
                if not self.connect():
                    self.startTor()
            else:
                # 如果不是 Windows 平台或者没有 Tor 可执行文件，则禁用 Tor
                self.log.info("Disabling Tor, because error while accessing Tor proxy at port %s: %s" % (config.tor_proxy, err))
                self.enabled = False

    # 设置状态函数
    def setStatus(self, status):
        self.status = status
        # 如果主模块存在，则更新 WebSocket
        if "main" in sys.modules: # import main has side-effects, breaks tests
            import main
            if "ui_server" in dir(main):
                main.ui_server.updateWebsocket()
    # 启动 Tor 客户端
    def startTor(self):
        # 如果是 Windows 系统
        if sys.platform.startswith("win"):
            try:
                # 记录日志，启动 Tor 客户端
                self.log.info("Starting Tor client %s..." % self.tor_exe)
                # 获取 Tor 可执行文件所在目录
                tor_dir = os.path.dirname(self.tor_exe)
                # 创建启动信息对象
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                # 构建启动命令
                cmd = r"%s -f torrc --defaults-torrc torrc-defaults --ignore-missing-torrc" % self.tor_exe
                # 如果配置中使用桥接
                if config.tor_use_bridges:
                    cmd += " --UseBridges 1"

                # 启动 Tor 进程
                self.tor_process = subprocess.Popen(cmd, cwd=tor_dir, close_fds=True, startupinfo=startupinfo)
                # 等待启动
                for wait in range(1, 3):  # Wait for startup
                    time.sleep(wait * 0.5)
                    self.enabled = True
                    # 如果连接成功
                    if self.connect():
                        # 如果子进程正在运行
                        if self.isSubprocessRunning():
                            # 当控制连接关闭时关闭 Tor 客户端
                            self.request("TAKEOWNERSHIP")
                        break
                # 在退出时终止
                atexit.register(self.stopTor)
            except Exception as err:
                # 记录错误日志
                self.log.error("Error starting Tor client: %s" % Debug.formatException(str(err)))
                self.enabled = False
        # 标记启动完成
        self.starting = False
        self.event_started.set(False)
        # 返回 False
        return False

    # 检查子进程是否正在运行
    def isSubprocessRunning(self):
        return self.tor_process and self.tor_process.pid and self.tor_process.poll() is None

    # 停止 Tor 客户端
    def stopTor(self):
        # 记录调试日志
        self.log.debug("Stopping...")
        try:
            # 如果子进程正在运行
            if self.isSubprocessRunning():
                # 发送关闭信号
                self.request("SIGNAL SHUTDOWN")
        except Exception as err:
            # 记录错误日志
            self.log.error("Error stopping Tor: %s" % err)

    # 连接到 Tor 控制器
    def connect(self):
        # 如果未启用
        if not self.enabled:
            return False
        # 初始化站点和私钥
        self.site_onions = {}
        self.privatekeys = {}

        # 返回连接控制器的结果
        return self.connectController()
    # 断开连接方法，关闭连接并将连接对象置为None
    def disconnect(self):
        if self.conn:
            self.conn.close()
        self.conn = None

    # 启动Onions方法，如果启用了Tor，则记录调试信息，设置start_onions为True，并获取全局Onion
    def startOnions(self):
        if self.enabled:
            self.log.debug("Start onions")
            self.start_onions = True
            self.getOnion("global")

    # 获取新的出口节点IP地址
    def resetCircuits(self):
        # 发送请求信号SIGNAL NEWNYM，如果返回结果不包含"250 OK"，则记录错误信息
        res = self.request("SIGNAL NEWNYM")
        if "250 OK" not in res:
            self.setStatus("Reset circuits error (%s)" % res)
            self.log.error("Tor reset circuits error: %s" % res)

    # 添加Onion方法，如果私钥数量超过配置的tor_hs_limit，则返回随机选择的私钥，否则生成新的Onion地址和私钥，并将其添加到私钥字典中
    def addOnion(self):
        if len(self.privatekeys) >= config.tor_hs_limit:
            return random.choice([key for key in list(self.privatekeys.keys()) if key != self.site_onions.get("global")])

        result = self.makeOnionAndKey()
        if result:
            onion_address, onion_privatekey = result
            self.privatekeys[onion_address] = onion_privatekey
            self.setStatus("OK (%s onions running)" % len(self.privatekeys))
            SiteManager.peer_blacklist.append((onion_address + ".onion", self.fileserver_port))
            return onion_address
        else:
            return False

    # 生成Onion地址和私钥的方法，发送请求信号ADD_ONION NEW:RSA1024，并从返回结果中匹配出Onion地址和私钥
    def makeOnionAndKey(self):
        res = self.request("ADD_ONION NEW:RSA1024 port=%s" % self.fileserver_port)
        match = re.search("ServiceID=([A-Za-z0-9]+).*PrivateKey=RSA1024:(.*?)[\r\n]", res, re.DOTALL)
        if match:
            onion_address, onion_privatekey = match.groups()
            return (onion_address, onion_privatekey)
        else:
            self.setStatus("AddOnion error (%s)" % res)
            self.log.error("Tor addOnion error: %s" % res)
            return False
    # 从 Tor 服务中删除指定的地址，并返回操作结果
    def delOnion(self, address):
        # 发送删除指定地址的请求，并获取返回结果
        res = self.request("DEL_ONION %s" % address)
        # 如果返回结果包含 "250 OK"，则表示删除成功
        if "250 OK" in res:
            # 从私钥字典中删除对应地址的私钥
            del self.privatekeys[address]
            # 设置状态信息，表示删除成功并显示当前私钥数量
            self.setStatus("OK (%s onion running)" % len(self.privatekeys))
            return True
        else:
            # 设置状态信息，表示删除失败并显示返回结果
            self.setStatus("DelOnion error (%s)" % res)
            # 记录 Tor 删除地址的错误日志
            self.log.error("Tor delOnion error: %s" % res)
            # 断开连接
            self.disconnect()
            return False

    # 发送请求到 Tor 服务，并返回结果
    def request(self, cmd):
        # 使用线程锁保证请求的原子性
        with self.lock:
            # 如果 Tor 服务未启用，则直接返回 False
            if not self.enabled:
                return False
            # 如果连接不存在，则尝试建立连接
            if not self.conn:
                if not self.connect():
                    return ""
            # 发送指定命令到 Tor 服务，并返回结果
            return self.send(cmd)

    # 发送命令到 Tor 服务，并返回结果
    def send(self, cmd, conn=None):
        # 如果连接未指定，则使用默认连接
        if not conn:
            conn = self.conn
        # 记录发送的命令到日志
        self.log.debug("> %s" % cmd)
        # 初始化返回结果为空字符串
        back = ""
        # 最多重试两次
        for retry in range(2):
            try:
                # 发送命令到 Tor 服务
                conn.sendall(b"%s\r\n" % cmd.encode("utf8"))
                # 循环接收返回结果，直到收到 "250 OK" 结尾
                while not back.endswith("250 OK\r\n"):
                    back += conn.recv(1024 * 64).decode("utf8")
                break
            except Exception as err:
                # 记录发送命令错误到日志，并尝试重新连接
                self.log.error("Tor send error: %s, reconnecting..." % err)
                if not self.connecting:
                    self.disconnect()
                    time.sleep(1)
                    self.connect()
                back = None
        # 如果有返回结果，则记录到日志
        if back:
            self.log.debug("< %s" % back.strip())
        return back

    # 获取指定地址对应的私钥
    def getPrivatekey(self, address):
        return self.privatekeys[address]

    # 获取指定地址对应的公钥
    def getPublickey(self, address):
        # 使用私钥生成对应的公钥，并返回
        return CryptRsa.privatekeyToPublickey(self.privatekeys[address])
    # 获取指定站点的 .onion 地址
    def getOnion(self, site_address):
        # 如果代理未启用，则返回空
        if not self.enabled:
            return None

        # 如果配置为始终使用 Tor，则为每个站点返回不同的 .onion 地址
        if config.tor == "always":  # 每个站点有不同的 .onion 地址
            onion = self.site_onions.get(site_address)
        else:  # 每个站点使用相同的 .onion 地址
            onion = self.site_onions.get("global")
            site_address = "global"

        # 如果没有 .onion 地址，则创建一个新的 .onion 地址
        if not onion:
            with self.lock:
                self.site_onions[site_address] = self.addOnion()
                onion = self.site_onions[site_address]
                self.log.debug("Created new hidden service for %s: %s" % (site_address, onion))

        return onion

    # 创建并返回连接到 Tor 网络的套接字
    def createSocket(self, onion, port):
        # 如果代理未启用，则返回 False
        if not self.enabled:
            return False
        # 记录创建新的 Tor 套接字的日志
        self.log.debug("Creating new Tor socket to %s:%s" % (onion, port))
        # 如果正在启动 Tor，则等待启动完成
        if self.starting:
            self.log.debug("Waiting for startup...")
            self.event_started.get()
        # 如果配置为始终使用 Tor，则默认情况下每个套接字都使用代理
        if config.tor == "always":  # 每个套接字默认情况下都使用代理
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            # 使用 Tor 代理创建套接字
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS5, self.proxy_ip, self.proxy_port)
        return sock
```