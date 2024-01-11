# `ZeroNet\src\Connection\ConnectionServer.py`

```
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入系统模块
import sys
# 导入套接字模块
import socket
# 导入默认字典模块
from collections import defaultdict
# 导入协程模块
import gevent
# 导入消息打包模块
import msgpack
# 从协程服务器模块中导入流服务器
from gevent.server import StreamServer
# 从协程池模块中导入池
from gevent.pool import Pool
# 导入自定义工具模块
import util
# 从自定义工具模块中导入助手
from util import helper
# 从调试模块中导入调试
from Debug import Debug
# 从连接模块中导入连接
from .Connection import Connection
# 从配置模块中导入配置
from Config import config
# 从加密模块中导入加密连接
from Crypt import CryptConnection
# 从加密模块中导入加密哈希
from Crypt import CryptHash
# 从Tor模块中导入Tor管理器
from Tor import TorManager
# 从站点模块中导入站点管理器
from Site import SiteManager

# 定义连接服务器类
class ConnectionServer(object):
    # 启动方法
    def start(self, check_connections=True):
        # 如果正在停止，则返回假
        if self.stopping:
            return False
        # 设置运行状态为真
        self.running = True
        # 如果需要检查连接
        if check_connections:
            # 创建一个协程来检查连接
            self.thread_checker = gevent.spawn(self.checkConnections)
        # 加载证书
        CryptConnection.manager.loadCerts()
        # 如果Tor不是禁用状态
        if config.tor != "disable":
            # 启动Tor管理器
            self.tor_manager.start()
        # 如果没有端口
        if not self.port:
            # 记录日志，没有找到端口，不进行绑定
            self.log.info("No port found, not binding")
            return False

        # 记录调试日志，绑定到指定IP和端口，消息打包版本，支持的加密方式
        self.log.debug("Binding to: %s:%s, (msgpack: %s), supported crypt: %s" % (
            self.ip, self.port, ".".join(map(str, msgpack.version)),
            CryptConnection.manager.crypt_supported
        ))
        try:
            # 创建流服务器
            self.stream_server = StreamServer(
                (self.ip, self.port), self.handleIncomingConnection, spawn=self.pool, backlog=100
            )
        except Exception as err:
            # 记录日志，流服务器创建错误
            self.log.info("StreamServer create error: %s" % Debug.formatException(err))

    # 监听方法
    def listen(self):
        # 如果没有运行，则返回空
        if not self.running:
            return None

        # 如果有流服务器代理
        if self.stream_server_proxy:
            # 创建一个协程来监听代理
            gevent.spawn(self.listenProxy)
        try:
            # 开始监听
            self.stream_server.serve_forever()
        except Exception as err:
            # 记录日志，流服务器监听错误
            self.log.info("StreamServer listen error: %s" % err)
            return False
        # 记录调试日志，停止监听
        self.log.debug("Stopped.")
    # 停止服务器，记录日志并设置停止标志
    def stop(self):
        self.log.debug("Stopping %s" % self.stream_server)
        self.stopping = True
        self.running = False
        # 如果存在线程检查器，则终止线程
        if self.thread_checker:
            gevent.kill(self.thread_checker)
        # 如果存在流服务器，则停止流服务器
        if self.stream_server:
            self.stream_server.stop()

    # 关闭所有连接
    def closeConnections(self):
        self.log.debug("Closing all connection: %s" % len(self.connections)
        # 遍历所有连接并关闭
        for connection in self.connections[:]:
            connection.close("Close all connections")

    # 处理传入连接
    def handleIncomingConnection(self, sock, addr):
        # 如果处于离线模式，则关闭套接字并返回
        if config.offline:
            sock.close()
            return False

        # 获取连接的IP地址和端口
        ip, port = addr[0:2]
        ip = ip.lower()
        # 如果是IPv6映射到IPv4，则进行替换
        if ip.startswith("::ffff:"):  # IPv6 to IPv4 mapping
            ip = ip.replace("::ffff:", "", 1)
        self.num_incoming += 1

        # 如果之前没有外部传入连接并且不是私有IP，则设置标志
        if not self.had_external_incoming and not helper.isPrivateIp(ip):
            self.had_external_incoming = True

        # 连接洪水保护
        if ip in self.ip_incoming and ip not in self.whitelist:
            self.ip_incoming[ip] += 1
            # 如果1分钟内来自同一IP的连接超过6个，则记录日志并关闭连接
            if self.ip_incoming[ip] > 6:  # Allow 6 in 1 minute from same ip
                self.log.debug("Connection flood detected from %s" % ip)
                time.sleep(30)
                sock.close()
                return False
        else:
            self.ip_incoming[ip] = 1

        # 创建连接对象并添加到连接列表中
        connection = Connection(self, ip, port, sock)
        self.connections.append(connection)
        # 如果IP不在本地IP列表中，则将其添加到IP字典中
        if ip not in config.ip_local:
            self.ips[ip] = connection
        # 处理传入连接
        connection.handleIncomingConnection(sock)

    # 处理消息
    def handleMessage(self, *args, **kwargs):
        pass
    # 从连接中删除指定的连接对象
    def removeConnection(self, connection):
        # 如果连接的 IP 在注册表中存在，则删除
        if self.ips.get(connection.ip) == connection:
            del self.ips[connection.ip]
        # 如果连接的目标是 .onion 地址，则删除对应的连接
        if connection.target_onion:
            if self.ips.get(connection.ip + connection.target_onion) == connection:
                del self.ips[connection.ip + connection.target_onion]
        # 如果连接的证书 pin 存在，并且在注册表中存在，则删除
        if connection.cert_pin and self.ips.get(connection.ip + "#" + connection.cert_pin) == connection:
            del self.ips[connection.ip + "#" + connection.cert_pin]

        # 如果连接在连接列表中存在，则移除
        if connection in self.connections:
            self.connections.remove(connection)

    # 异步方法，检查最大连接数限制
    @util.Noparallel(blocking=False)
    def checkMaxConnections(self):
        # 如果连接数小于全局连接限制，则返回 0
        if len(self.connections) < config.global_connected_limit:
            return 0

        # 记录开始时间
        s = time.time()
        # 记录当前连接数
        num_connected_before = len(self.connections)
        # 根据连接的站点数排序连接列表
        self.connections.sort(key=lambda connection: connection.sites)
        # 记录关闭的连接数
        num_closed = 0
        # 遍历连接列表
        for connection in self.connections:
            # 计算连接的空闲时间
            idle = time.time() - max(connection.last_recv_time, connection.start_time, connection.last_message_time)
            # 如果空闲时间超过 60 秒，则关闭连接并增加关闭连接数
            if idle > 60:
                connection.close("Connection limit reached")
                num_closed += 1
            # 如果关闭连接数超过全局连接限制的 10%，则跳出循环
            if num_closed > config.global_connected_limit * 0.1:
                break

        # 记录日志，显示关闭的连接数、连接数、全局连接限制和执行时间
        self.log.debug("Closed %s connections of %s after reached limit %s in %.3fs" % (
            num_closed, num_connected_before, config.global_connected_limit, time.time() - s
        ))
        return num_closed

    # 在网络在线时调用的方法
    def onInternetOnline(self):
        self.log.info("Internet online")

    # 在网络离线时调用的方法
    def onInternetOffline(self):
        # 重置外部传入连接标志
        self.had_external_incoming = False
        self.log.info("Internet offline")
    # 获取时间校正值
    def getTimecorrection(self):
        # 对连接列表中的每个连接进行操作，计算时间校正值
        corrections = sorted([
            connection.handshake.get("time") - connection.handshake_time + connection.last_ping_delay
            for connection in self.connections
            if connection.handshake.get("time") and connection.last_ping_delay
        ])
        # 如果校正值列表长度小于9，返回0.0
        if len(corrections) < 9:
            return 0.0
        # 计算校正值列表的中位数
        mid = int(len(corrections) / 2 - 1)
        median = (corrections[mid - 1] + corrections[mid] + corrections[mid + 1]) / 3
        # 返回中位数作为时间校正值
        return median
```