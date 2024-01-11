# `ZeroNet\src\Connection\Connection.py`

```
# 导入 socket 和 time 模块
import socket
import time

# 导入 gevent 模块
import gevent
# 尝试导入 gevent.coros 模块中的 RLock 类，如果失败则导入 gevent.lock 模块中的 RLock 类
try:
    from gevent.coros import RLock
except:
    from gevent.lock import RLock

# 从 Config 模块中导入 config 变量
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 Msgpack 和 helper 函数
from util import Msgpack
# 从 Crypt 模块中导入 CryptConnection 类
from Crypt import CryptConnection
# 从 util 模块中导入 helper 函数
from util import helper

# 定义 Connection 类
class Connection(object):
    # 定义 __slots__ 属性
    __slots__ = (
        "sock", "sock_wrapped", "ip", "port", "cert_pin", "target_onion", "id", "protocol", "type", "server", "unpacker", "unpacker_bytes", "req_id", "ip_type",
        "handshake", "crypt", "connected", "event_connected", "closed", "start_time", "handshake_time", "last_recv_time", "is_private_ip", "is_tracker_connection",
        "last_message_time", "last_send_time", "last_sent_time", "incomplete_buff_recv", "bytes_recv", "bytes_sent", "cpu_time", "send_lock",
        "last_ping_delay", "last_req_time", "last_cmd_sent", "last_cmd_recv", "bad_actions", "sites", "name", "waiting_requests", "waiting_streams"
    )

    # 定义 setIp 方法
    def setIp(self, ip):
        # 设置 ip 属性为传入的 ip 参数
        self.ip = ip
        # 根据 ip 获取 IP 类型，并设置 ip_type 属性
        self.ip_type = helper.getIpType(ip)
        # 更新连接名称
        self.updateName()

    # 定义 createSocket 方法
    def createSocket(self):
        # 如果连接的 IP 类型为 IPv6 并且没有 socket_noproxy 属性
        if helper.getIpType(self.ip) == "ipv6" and not hasattr(socket, "socket_noproxy"):
            # 创建 IPv6 连接作为 IPv4 连接
            return socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            # 创建 IPv4 连接
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 定义 updateName 方法
    def updateName(self):
        # 根据 id、ip 和 protocol 更新连接名称
        self.name = "Conn#%2s %-12s [%s]" % (self.id, self.ip, self.protocol)

    # 定义 __str__ 方法
    def __str__(self):
        # 返回连接名称
        return self.name

    # 定义 __repr__ 方法
    def __repr__(self):
        # 返回连接的字符串表示形式
        return "<%s>" % self.__str__()

    # 定义 log 方法
    def log(self, text):
        # 记录日志信息，包括连接名称和文本内容
        self.server.log.debug("%s > %s" % (self.name, text))

    # 定义 getValidSites 方法
    def getValidSites(self):
        # 返回与目标洋葱地址匹配的有效站点列表
        return [key for key, val in self.server.tor_manager.site_onions.items() if val == self.target_onion]
    # 执行不良操作，可以指定权重，默认为1
    def badAction(self, weight=1):
        # 增加不良操作计数
        self.bad_actions += weight
        # 如果不良操作计数超过40次，关闭连接并提示错误
        if self.bad_actions > 40:
            self.close("Too many bad actions")
        # 如果不良操作计数超过20次，休眠5秒
        elif self.bad_actions > 20:
            time.sleep(5)

    # 执行良好操作，将不良操作计数清零
    def goodAction(self):
        self.bad_actions = 0

    # 打开与对等方的连接并等待握手
    # 处理传入连接
    def handleIncomingConnection(self, sock):
        self.log("Incoming connection...")

        # 如果socket支持TCP_NODELAY选项，设置TCP_NODELAY选项为1
        if "TCP_NODELAY" in dir(socket):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # 设置连接类型为"in"
        self.type = "in"
        # 如果IP不在本地IP列表中，检查是否使用隐式SSL
        if self.ip not in config.ip_local:   
            try:
                # 探测传入连接的第一个字节，如果是SSL握手协议，使用隐式SSL加密连接
                first_byte = sock.recv(1, gevent.socket.MSG_PEEK)
                if first_byte == b"\x16":
                    self.log("Crypt in connection using implicit SSL")
                    self.sock = CryptConnection.manager.wrapSocket(self.sock, "tls-rsa", True)
                    self.sock_wrapped = True
                    self.crypt = "tls-rsa"
            except Exception as err:
                self.log("Socket peek error: %s" % Debug.formatException(err))
        # 进入消息循环
        self.messageLoop()

    # 获取Msgpack解包器
    def getMsgpackUnpacker(self):
        # 如果握手信息中指定使用二进制类型，则返回解包器（不解码）
        if self.handshake and self.handshake.get("use_bin_type"):
            return Msgpack.getUnpacker(fallback=True, decode=False)
        else:  # 向后兼容<0.7.0
            return Msgpack.getUnpacker(fallback=True, decode=True)

    # 获取未处理字节数
    def getUnpackerUnprocessedBytesNum(self):
        # 如果解包器支持tell方法，计算未处理字节数
        if "tell" in dir(self.unpacker):
            bytes_num = self.unpacker_bytes - self.unpacker.tell()
        else:
            bytes_num = self.unpacker._fb_buf_n - self.unpacker._fb_buf_o
        return bytes_num

    # 将流式套接字直接写入文件
    # 我的握手信息
    # 获取握手信息
    def getHandshakeInfo(self):
        # 对于洋葱连接，不使用 TLS
        if self.ip_type == "onion":
            crypt_supported = []
        # 对于已知的 SSL 有问题的 IP 地址，也不使用 TLS
        elif self.ip in self.server.broken_ssl_ips:
            crypt_supported = []
        else:
            # 使用 CryptConnection 管理器中支持的加密方式
            crypt_supported = CryptConnection.manager.crypt_supported
        # 对于洋葱连接或本地 IP 地址，不设置对等方 ID
        if self.ip_type == "onion" or self.ip in config.ip_local:
            peer_id = ""
        else:
            # 使用服务器的对等方 ID
            peer_id = self.server.peer_id
        # 从握手信息中设置目标洋葱地址的对等锁
        if self.handshake and self.handshake.get("target_ip", "").endswith(".onion") and self.server.tor_manager.start_onions:
            # 设置目标洋葱地址
            self.target_onion = self.handshake.get("target_ip").replace(".onion", "")  # 我的洋葱地址
            if not self.server.tor_manager.site_onions.values():
                self.server.log.warning("未知的目标洋葱地址：%s" % self.target_onion)

        # 构建握手信息字典
        handshake = {
            "version": config.version,
            "protocol": "v2",
            "use_bin_type": True,
            "peer_id": peer_id,
            "fileserver_port": self.server.port,
            "port_opened": self.server.port_opened.get(self.ip_type, None),
            "target_ip": self.ip,
            "rev": config.rev,
            "crypt_supported": crypt_supported,
            "crypt": self.crypt,
            "time": int(time.time())
        }
        # 如果存在目标洋葱地址，将其添加到握手信息中
        if self.target_onion:
            handshake["onion"] = self.target_onion
        # 对于洋葱连接，使用全局洋葱地址
        elif self.ip_type == "onion":
            handshake["onion"] = self.server.tor_manager.getOnion("global")

        # 如果是跟踪器连接，设置握手信息中的 tracker_connection 为 True
        if self.is_tracker_connection:
            handshake["tracker_connection"] = True

        # 如果开启了调试模式，记录握手信息
        if config.debug_socket:
            self.log("我的握手信息：%s" % handshake)

        # 返回握手信息字典
        return handshake

    # 处理传入消息
    # 处理传入的消息
    def handleMessage(self, message):
        # 从消息中获取命令
        cmd = message["cmd"]

        # 更新最后一条消息的时间和命令
        self.last_message_time = time.time()
        self.last_cmd_recv = cmd

        # 如果命令是 "response"，则执行以下操作
        if cmd == "response":  # New style response
            # 如果消息中的接收者在等待的请求列表中
            if message["to"] in self.waiting_requests:
                # 如果最后一次发送时间存在且等待请求列表中只有一个请求
                if self.last_send_time and len(self.waiting_requests) == 1:
                    # 计算并更新最后一次的 ping 延迟
                    ping = time.time() - self.last_send_time
                    self.last_ping_delay = ping
                # 将响应设置为事件
                self.waiting_requests[message["to"]]["evt"].set(message)
                # 从等待请求列表中删除该请求
                del self.waiting_requests[message["to"]]
            # 如果消息中的接收者为 0，表示其他对等方的握手
            elif message["to"] == 0:  # Other peers handshake
                # 计算握手的 ping 延迟
                ping = time.time() - self.start_time
                # 如果开启了调试模式，记录握手响应和 ping 延迟
                if config.debug_socket:
                    self.log("Handshake response: %s, ping: %s" % (message, ping))
                # 更新最后一次的 ping 延迟
                self.last_ping_delay = ping
                # 如果消息中包含加密信息，并且套接字未加密，则进行加密
                if message.get("crypt") and not self.sock_wrapped:
                    self.crypt = message["crypt"]
                    server = (self.type == "in")
                    # 记录加密输出连接的信息
                    self.log("Crypt out connection using: %s (server side: %s, ping: %.3fs)..." % (self.crypt, server, ping))
                    # 使用 CryptConnection 管理器对套接字进行加密
                    self.sock = CryptConnection.manager.wrapSocket(self.sock, self.crypt, server, cert_pin=self.cert_pin)
                    self.sock.do_handshake()
                    self.sock_wrapped = True

                # 如果套接字未加密且存在证书 pin，则关闭连接并返回错误信息
                if not self.sock_wrapped and self.cert_pin:
                    self.close("Crypt connection error: Socket not encrypted, but certificate pin present")
                    return

                # 设置握手信息
                self.setHandshake(message)
            # 如果消息中的接收者不在等待请求列表中，也不是 0，则记录未知响应
            else:
                self.log("Unknown response: %s" % message)
        # 如果命令存在
        elif cmd:
            # 增加服务器接收的消息计数
            self.server.num_recv += 1
            # 如果命令是 "handshake"，则处理握手消息
            if cmd == "handshake":
                self.handleHandshake(message)
            # 否则，由服务器处理请求
            else:
                self.server.handleRequest(self, message)
    # 处理握手请求
    def handleHandshake(self, message):
        # 设置握手信息
        self.setHandshake(message["params"])
        # 获取握手信息
        data = self.getHandshakeInfo()
        # 设置命令为响应
        data["cmd"] = "response"
        # 设置接收方为请求的ID
        data["to"] = message["req_id"]
        # 发送响应给握手请求
        self.send(data)
        # 如果需要加密并且套接字未包装
        if self.crypt and not self.sock_wrapped:
            # 判断是服务端还是客户端
            server = (self.type == "in")
            # 记录加密连接信息
            self.log("Crypt in connection using: %s (server side: %s)..." % (self.crypt, server))
            try:
                # 使用加密方式包装套接字
                self.sock = CryptConnection.manager.wrapSocket(self.sock, self.crypt, server, cert_pin=self.cert_pin)
                self.sock_wrapped = True
            except Exception as err:
                # 如果不强制加密，则记录加密连接错误
                if not config.force_encryption:
                    self.log("Crypt connection error, adding %s:%s as broken ssl. %s" % (self.ip, self.port, Debug.formatException(err)))
                    self.server.broken_ssl_ips[self.ip] = True
                # 关闭连接
                self.close("Broken ssl")
        # 如果套接字未包装并且存在证书锁定
        if not self.sock_wrapped and self.cert_pin:
            # 关闭连接并记录错误信息
            self.close("Crypt connection error: Socket not encrypted, but certificate pin present")
    
    # 发送数据到连接
    # 定义一个发送消息的方法，可以选择是否使用流式传输
    def send(self, message, streaming=False):
        # 记录发送消息的时间
        self.last_send_time = time.time()
        # 如果开启了调试模式，记录发送的消息内容
        if config.debug_socket:
            self.log("Send: %s, to: %s, streaming: %s, site: %s, inner_path: %s, req_id: %s" % (
                message.get("cmd"), message.get("to"), streaming,
                message.get("params", {}).get("site"), message.get("params", {}).get("inner_path"),
                message.get("req_id"))
            )

        # 如果没有建立 socket 连接，记录错误并返回 False
        if not self.sock:
            self.log("Send error: missing socket")
            return False

        # 如果未连接并且消息命令不是握手命令，等待握手完成
        if not self.connected and message.get("cmd") != "handshake":
            self.log("Wait for handshake before send request")
            self.event_connected.get()

        try:
            # 获取消息命令，如果是响应命令，则使用上一次接收到的命令作为统计键值
            stat_key = message.get("cmd", "unknown")
            if stat_key == "response":
                stat_key = "response: %s" % self.last_cmd_recv
            else:
                self.server.num_sent += 1

            # 更新发送消息的统计信息
            self.server.stat_sent[stat_key]["num"] += 1
            # 如果使用流式传输
            if streaming:
                # 使用发送锁保证线程安全，将消息通过流式传输发送，并记录发送的字节数
                with self.send_lock:
                    bytes_sent = Msgpack.stream(message, self.sock.sendall)
                self.bytes_sent += bytes_sent
                self.server.bytes_sent += bytes_sent
                self.server.stat_sent[stat_key]["bytes"] += bytes_sent
                message = None
            else:
                # 如果不使用流式传输，对消息进行消息打包，并记录发送的字节数
                data = Msgpack.pack(message)
                self.bytes_sent += len(data)
                self.server.bytes_sent += len(data)
                self.server.stat_sent[stat_key]["bytes"] += len(data)
                message = None
                # 使用发送锁保证线程安全，将消息通过 socket 发送
                with self.send_lock:
                    self.sock.sendall(data)
        except Exception as err:
            # 发生异常时，记录错误并关闭连接，返回 False
            self.close("Send error: %s (cmd: %s)" % (err, stat_key))
            return False
        # 记录发送消息的时间
        self.last_sent_time = time.time()
        # 返回 True 表示发送成功
        return True

    # Stream file to connection without msgpacking
    # 发送原始文件数据给对等方
    def sendRawfile(self, file, read_bytes):
        # 设置缓冲区大小
        buff = 64 * 1024
        # 计算剩余要读取的字节数
        bytes_left = read_bytes
        # 已发送的字节数
        bytes_sent = 0
        # 循环发送数据直到全部发送完毕
        while True:
            # 更新最后一次发送时间
            self.last_send_time = time.time()
            # 读取数据
            data = file.read(min(bytes_left, buff))
            # 更新已发送字节数
            bytes_sent += len(data)
            # 使用发送锁发送数据
            with self.send_lock:
                self.sock.sendall(data)
            # 更新剩余字节数
            bytes_left -= buff
            # 如果剩余字节数小于等于0，跳出循环
            if bytes_left <= 0:
                break
        # 更新已发送字节数
        self.bytes_sent += bytes_sent
        # 更新服务器已发送字节数
        self.server.bytes_sent += bytes_sent
        # 更新服务器发送统计信息
        self.server.stat_sent["raw_file"]["num"] += 1
        self.server.stat_sent["raw_file"]["bytes"] += bytes_sent
        # 返回True
        return True

    # 创建并发送对等方的请求
    def request(self, cmd, params={}, stream_to=None):
        # 如果存在等待的请求，并且协议为v2，并且最后一次请求时间距离现在超过10秒，超时
        if self.waiting_requests and self.protocol == "v2" and time.time() - max(self.last_req_time, self.last_recv_time) > 10:
            # 关闭连接，返回False
            self.close("Request %s timeout: %.3fs" % (self.last_cmd_sent, time.time() - self.last_send_time))
            return False

        # 更新最后一次请求时间
        self.last_req_time = time.time()
        # 更新最后一次发送的命令
        self.last_cmd_sent = cmd
        # 增加请求ID
        self.req_id += 1
        # 构建请求数据
        data = {"cmd": cmd, "req_id": self.req_id, "params": params}
        # 创建新的事件对象用于响应
        event = gevent.event.AsyncResult()
        # 将请求ID和命令存储到等待请求字典中
        self.waiting_requests[self.req_id] = {"evt": event, "cmd": cmd}
        # 如果存在流传输目标，存储到等待流字典中
        if stream_to:
            self.waiting_streams[self.req_id] = stream_to
        # 发送请求数据
        self.send(data)
        # 等待事件解决
        res = event.get()
        # 返回响应结果
        return res
    # 发送 ping 请求，计时开始
    def ping(self):
        s = time.time()
        response = None
        # 设置超时时间为 10 秒，如果超时则返回 False
        with gevent.Timeout(10.0, False):
            try:
                # 发送 ping 请求并获取响应
                response = self.request("ping")
            except Exception as err:
                # 记录 ping 错误信息
                self.log("Ping error: %s" % Debug.formatException(err))
        # 如果收到响应且响应中包含 "body" 并且数据为 b"Pong!"，则计算延迟并返回 True
        if response and "body" in response and response["body"] == b"Pong!":
            self.last_ping_delay = time.time() - s
            return True
        else:
            return False

    # 关闭连接
    def close(self, reason="Unknown"):
        if self.closed:
            return False  # 已经关闭
        self.closed = True
        self.connected = False
        if self.event_connected:
            self.event_connected.set(False)

        # 记录关闭连接的原因以及一些状态信息
        self.log(
            "Closing connection: %s, waiting_requests: %s, sites: %s, buff: %s..." %
            (reason, len(self.waiting_requests), self.sites, self.incomplete_buff_recv)
        )
        # 标记所有待处理的请求为失败
        for request in self.waiting_requests.values():
            request["evt"].set(False)
        self.waiting_requests = {}
        self.waiting_streams = {}
        self.sites = 0
        # 从服务器注册表中移除连接
        self.server.removeConnection(self)
        try:
            if self.sock:
                # 关闭套接字的写入通道并关闭套接字
                self.sock.shutdown(gevent.socket.SHUT_WR)
                self.sock.close()
        except Exception as err:
            if config.debug_socket:
                # 如果配置为调试模式，则记录关闭错误信息
                self.log("Close error: %s" % err)

        # 清理一些变量
        self.sock = None
        self.unpacker = None
        self.event_connected = None
```