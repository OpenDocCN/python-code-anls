# `arknights-mower\arknights_mower\utils\device\minitouch\session.py`

```py
# 导入必要的模块
from __future__ import annotations
import socket
from ...log import logger

# 默认主机地址
DEFAULT_HOST = '127.0.0.1'

# 定义一个会话类，用于管理 PC 和 Android 之间的 socket 连接
class Session(object):
    """ manage socket connections between PC and Android """

    # 初始化方法，接收端口号和缓冲区大小作为参数
    def __init__(self, port: int, buf_size: int = 0) -> None:
        self.port = port
        self.buf_size = buf_size
        # 创建一个 socket 对象，并连接到默认主机和指定端口
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((DEFAULT_HOST, port))
        # 创建一个文件对象，用于写入 socket 输出
        socket_out = self.sock.makefile()

        # 读取并忽略协议版本信息
        socket_out.readline()

        # 读取并解析最大触摸点数、最大 x 坐标、最大 y 坐标和最大压力值
        _, max_contacts, max_x, max_y, max_pressure, *_ = (
            socket_out.readline().strip().split(' '))
        self.max_contacts = max_contacts
        self.max_x = max_x
        self.max_y = max_y
        self.max_pressure = max_pressure

        # 读取并解析进程 ID
        _, pid = socket_out.readline().strip().split(' ')
        self.pid = pid

        # 记录调试信息
        logger.debug(
            f'minitouch running on port: {self.port}, pid: {self.pid}')
        logger.debug(
            f'max_contact: {max_contacts}; max_x: {max_x}; max_y: {max_y}; max_pressure: {max_pressure}')

    # 进入上下文时返回自身
    def __enter__(self) -> Session:
        return self

    # 退出上下文时不做任何操作
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    # 对象被销毁时关闭连接
    def __del__(self) -> None:
        self.close()

    # 关闭连接的方法
    def close(self) -> None:
        """ cancel connection """
        self.sock and self.sock.close()
        self.sock = None

    # 发送数据并返回接收到的数据
    def send(self, content: str) -> bytes:
        content = content.encode('utf8')
        self.sock.sendall(content)
        return self.sock.recv(self.buf_size)
```