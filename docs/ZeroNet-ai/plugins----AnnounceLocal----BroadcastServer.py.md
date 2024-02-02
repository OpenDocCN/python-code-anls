# `ZeroNet\plugins\AnnounceLocal\BroadcastServer.py`

```py
# 导入 socket 模块
import socket
# 导入 logging 模块
import logging
# 导入 time 模块
import time
# 从 contextlib 模块中导入 closing 函数
from contextlib import closing
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 UpnpPunch 和 Msgpack
from util import UpnpPunch
from util import Msgpack

# 定义一个名为 BroadcastServer 的类
class BroadcastServer(object):
    # 初始化方法，接受服务名、监听端口和监听 IP 作为参数
    def __init__(self, service_name, listen_port=1544, listen_ip=''):
        # 获取名为 "BroadcastServer" 的日志记录器
        self.log = logging.getLogger("BroadcastServer")
        # 设置监听端口和监听 IP
        self.listen_port = listen_port
        self.listen_ip = listen_ip

        # 初始化运行状态为 False，套接字为 None，发送者信息为包含服务名的字典
        self.running = False
        self.sock = None
        self.sender_info = {"service": service_name}

    # 创建广播套接字的方法
    def createBroadcastSocket(self):
        # 创建一个 UDP 套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 设置套接字选项，允许广播
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # 设置套接字选项，允许地址重用
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 如果支持 SO_REUSEPORT 选项，则设置该选项
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception as err:
                # 记录警告日志，指出设置 SO_REUSEPORT 选项时出现错误
                self.log.warning("Error setting SO_REUSEPORT: %s" % err)

        # 标记是否成功绑定套接字的标志
        binded = False
        # 最多重试 3 次
        for retry in range(3):
            try:
                # 尝试绑定套接字到指定的 IP 和端口
                sock.bind((self.listen_ip, self.listen_port))
                # 设置绑定成功的标志为 True，并跳出循环
                binded = True
                break
            except Exception as err:
                # 记录错误日志，指出绑定套接字时出现错误，并记录重试次数
                self.log.error(
                    "Socket bind to %s:%s error: %s, retry #%s" %
                    (self.listen_ip, self.listen_port, Debug.formatException(err), retry)
                )
                # 休眠重试次数的秒数
                time.sleep(retry)

        # 如果成功绑定，则返回套接字对象，否则返回 False
        if binded:
            return sock
        else:
            return False
    # 启动方法，监听发现请求
    def start(self):  
        # 创建广播套接字
        self.sock = self.createBroadcastSocket()
        # 如果套接字创建失败，记录错误信息并返回
        if not self.sock:
            self.log.error("Unable to listen on port %s" % self.listen_port)
            return

        # 记录调试信息
        self.log.debug("Started on port %s" % self.listen_port)

        # 标记为正在运行
        self.running = True

        # 循环监听请求
        while self.running:
            try:
                # 接收数据和地址信息
                data, addr = self.sock.recvfrom(8192)
            except Exception as err:
                # 如果出现异常并且仍在运行状态，记录错误信息
                if self.running:
                    self.log.error("Listener receive error: %s" % err)
                continue

            # 如果不在运行状态，跳出循环
            if not self.running:
                break

            try:
                # 解包消息数据
                message = Msgpack.unpack(data)
                # 处理消息并获取响应地址和消息
                response_addr, message = self.handleMessage(addr, message)
                # 如果有消息，发送消息
                if message:
                    self.send(response_addr, message)
            except Exception as err:
                # 记录处理消息时的错误信息
                self.log.error("Handlemessage error: %s" % Debug.formatException(err))
        # 记录停止监听的调试信息
        self.log.debug("Stopped listening on port %s" % self.listen_port)

    # 停止方法
    def stop(self):
        # 记录停止信息
        self.log.debug("Stopping, socket: %s" % self.sock)
        # 标记为不在运行状态
        self.running = False
        # 关闭套接字
        if self.sock:
            self.sock.close()

    # 发送方法
    def send(self, addr, message):
        # 如果消息不是列表类型，转换为列表
        if type(message) is not list:
            message = [message]

        # 遍历消息列表
        for message_part in message:
            # 设置发送者信息
            message_part["sender"] = self.sender_info

            # 记录发送信息
            self.log.debug("Send to %s: %s" % (addr, message_part["cmd"]))
            # 使用 UDP 协议发送消息
            with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(Msgpack.pack(message_part), addr)

    # 获取本地 IP 地址方法
    def getMyIps(self):
        return UpnpPunch._get_local_ips()
    # 发送广播消息到指定端口，如果端口未指定，则使用默认监听端口
    def broadcast(self, message, port=None):
        # 如果端口未指定，则使用默认监听端口
        if not port:
            port = self.listen_port

        # 获取本机 IP 地址
        my_ips = self.getMyIps()
        # 广播地址和端口
        addr = ("255.255.255.255", port)

        # 将发送者信息添加到消息中，并记录广播的详细信息
        message["sender"] = self.sender_info
        self.log.debug("Broadcast using ips %s on port %s: %s" % (my_ips, port, message["cmd"]))

        # 遍历本机 IP 地址，发送消息
        for my_ip in my_ips:
            try:
                # 创建 UDP 套接字，并设置广播选项
                with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    # 绑定本机 IP 和随机端口，发送消息到广播地址
                    sock.bind((my_ip, 0))
                    sock.sendto(Msgpack.pack(message), addr)
            except Exception as err:
                # 记录发送广播消息时的错误信息
                self.log.warning("Error sending broadcast using ip %s: %s" % (my_ip, err))

    # 处理接收到的消息
    def handleMessage(self, addr, message):
        # 记录接收到的消息的详细信息
        self.log.debug("Got from %s: %s" % (addr, message["cmd"]))
        # 获取消息中的命令、参数和发送者信息
        cmd = message["cmd"]
        params = message.get("params", {})
        sender = message["sender"]
        sender["ip"] = addr[0]

        # 根据命令构造对应的方法名，并获取该方法
        func_name = "action" + cmd[0].upper() + cmd[1:]
        func = getattr(self, func_name, None)

        # 如果消息不是发给我们的，或者是我们自己发送的，则忽略该消息
        if sender["service"] != "zeronet" or sender["peer_id"] == self.sender_info["peer_id"]:
            message = None
        # 如果存在对应的方法，则执行该方法，否则忽略该消息
        elif func:
            message = func(sender, params)
        else:
            self.log.debug("Unknown cmd: %s" % cmd)
            message = None

        # 返回发送者的 IP 和广播端口，以及处理后的消息
        return (sender["ip"], sender["broadcast_port"]), message
```