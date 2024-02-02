# `arknights-mower\arknights_mower\utils\device\adb_client\socket.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入 socket 模块
import socket

# 从 log 模块中导入 logger 对象
from ...log import logger

# 定义 Socket 类
class Socket(object):
    """ Connect ADB server with socket """

    # 初始化方法，接受服务器地址和超时时间作为参数
    def __init__(self, server: tuple[str, int], timeout: int) -> None:
        # 记录调试日志
        logger.debug(f'server: {server}, timeout: {timeout}')
        try:
            # 初始化套接字对象
            self.sock = None
            # 创建与服务器的连接
            self.sock = socket.create_connection(server, timeout=timeout)
            # 设置套接字选项，禁用 Nagle 算法
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 处理连接被拒绝的异常
        except ConnectionRefusedError as e:
            # 记录错误日志
            logger.error(f'ConnectionRefusedError: {server}')
            # 抛出异常
            raise e

    # 进入上下文管理器时调用的方法
    def __enter__(self) -> Socket:
        return self

    # 退出上下文管理器时调用的方法
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    # 对象被销毁时调用的方法
    def __del__(self) -> None:
        # 关闭套接字
        self.close()

    # 关闭套接字的方法
    def close(self) -> None:
        """ close socket """
        # 如果套接字对象存在，则关闭套接字
        self.sock and self.sock.close()
        # 将套接字对象置为 None
        self.sock = None

    # 接收所有数据的方法
    def recv_all(self, chunklen: int = 65536) -> bytes:
        data = []
        buf = bytearray(chunklen)
        view = memoryview(buf)
        pos = 0
        while True:
            if pos >= chunklen:
                data.append(buf)
                buf = bytearray(chunklen)
                view = memoryview(buf)
                pos = 0
            rcvlen = self.sock.recv_into(view)
            if rcvlen == 0:
                break
            view = view[rcvlen:]
            pos += rcvlen
        data.append(buf[:pos])
        return b''.join(data)

    # 精确接收指定长度数据的方法
    def recv_exactly(self, len: int) -> bytes:
        buf = bytearray(len)
        view = memoryview(buf)
        pos = 0
        while pos < len:
            rcvlen = self.sock.recv_into(view)
            if rcvlen == 0:
                break
            view = view[rcvlen:]
            pos += rcvlen
        if pos != len:
            raise EOFError('recv_exactly %d bytes failed' % len)
        return bytes(buf)
    # 接收服务器响应的数据，数据长度由前4个十六进制数字表示
    def recv_response(self) -> bytes:
        """ read a chunk of length indicated by 4 hex digits """
        # 读取4个字节的数据，将其解析为16进制数字，表示接下来要接收的数据长度
        len = int(self.recv_exactly(4), 16)
        # 如果长度为0，则返回空字节
        if len == 0:
            return b''
        # 否则，接收指定长度的数据并返回
        return self.recv_exactly(len)

    # 检查前4个字节是否为"OKAY"
    def check_okay(self) -> None:
        """ check if first 4 bytes is "OKAY" """
        # 接收4个字节的数据
        result = self.recv_exactly(4)
        # 如果不是"OKAY"，则抛出连接错误并返回响应数据
        if result != b'OKAY':
            raise ConnectionError(self.recv_response())

    # 接收指定长度的数据
    def recv(self, len: int) -> bytes:
        return self.sock.recv(len)

    # 向服务器发送数据
    def send(self, data: bytes) -> Socket:
        """ send data to server """
        # 发送数据到服务器
        self.sock.send(data)
        return self

    # 向服务器发送所有数据
    def sendall(self, data: bytes) -> Socket:
        """ send data to server """
        # 发送所有数据到服务器
        self.sock.sendall(data)
        return self

    # 将接收的数据存入缓冲区
    def recv_into(self, buffer, nbytes: int) -> None:
        self.sock.recv_into(buffer, nbytes)
```