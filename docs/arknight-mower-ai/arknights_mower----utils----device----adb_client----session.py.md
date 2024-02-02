# `arknights-mower\arknights_mower\utils\device\adb_client\session.py`

```py
# 导入必要的模块
from __future__ import annotations
import socket
import struct
import time
from ... import config
from ...log import logger
from .socket import Socket

# 创建一个会话类，用于 ADB 客户端和 ADB 服务器之间的会话
class Session(object):
    """ Session between ADB client and ADB server """

    # 初始化方法，设置服务器地址和超时时间
    def __init__(self, server: tuple[str, int] = None, timeout: int = None) -> None:
        # 如果未提供服务器地址，则使用默认配置中的地址
        if server is None:
            server = (config.ADB_SERVER_IP, config.ADB_SERVER_PORT)
        # 如果未提供超时时间，则使用默认配置中的超时时间
        if timeout is None:
            timeout = config.ADB_SERVER_TIMEOUT
        # 设置服务器地址和超时时间
        self.server = server
        self.timeout = timeout
        self.device_id = None
        # 创建一个套接字对象
        self.sock = Socket(self.server, self.timeout)

    # 进入上下文时的操作，返回会话对象
    def __enter__(self) -> Session:
        return self

    # 退出上下文时的操作，不做任何处理
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    # 发送请求到 ADB 服务器，可选择是否重新连接
    def request(self, cmd: str, reconnect: bool = False) -> Session:
        """ make a service request to ADB server, consult ADB sources for available services """
        # 将命令编码为字节流
        cmdbytes = cmd.encode()
        # 构造数据包，包含命令的长度和命令的字节流
        data = b'%04X%b' % (len(cmdbytes), cmdbytes)
        # 当超时时间小于等于 60 时，循环发送请求
        while self.timeout <= 60:
            try:
                # 发送数据包并检查是否成功
                self.sock.send(data).check_okay()
                return self
            except socket.timeout:
                # 发生超时时，记录警告信息并增加超时时间
                logger.warning(f'socket.timeout: {self.timeout}s, +5s')
                self.timeout += 5
                # 重新创建套接字对象
                self.sock = Socket(self.server, self.timeout)
                # 如果允许重新连接，则重新连接设备
                if reconnect:
                    self.device(self.device_id)
        # 超时时间超过 60 秒时，抛出超时异常
        raise socket.timeout(f'server: {self.server}')

    # 接收响应数据
    def response(self, recv_all: bool = False) -> bytes:
        """ receive response """
        # 如果需要接收所有数据，则调用套接字对象的接收所有数据的方法
        if recv_all:
            return self.sock.recv_all()
        # 否则调用套接字对象的接收响应数据的方法
        else:
            return self.sock.recv_response()

    # 执行命令并返回结果
    def exec(self, cmd: str) -> bytes:
        """ exec: cmd """
        # 如果命令长度为 0，则抛出数值错误
        if len(cmd) == 0:
            raise ValueError('no command specified for exec')
        # 发送执行命令的请求，并获取响应数据
        return self.request('exec:' + cmd, True).response(True)
    # 执行 shell 命令，并返回结果
    def shell(self, cmd: str) -> bytes:
        """ shell: cmd """
        # 如果命令长度为0，则抛出数值错误
        if len(cmd) == 0:
            raise ValueError('no command specified for shell')
        # 发送 shell 命令请求，并返回响应结果
        return self.request('shell:' + cmd, True).response(True)

    # 执行主机命令，并返回结果
    def host(self, cmd: str) -> bytes:
        """ host: cmd """
        # 如果命令长度为0，则抛出数值错误
        if len(cmd) == 0:
            raise ValueError('no command specified for host')
        # 发送主机命令请求，并返回响应结果
        return self.request('host:' + cmd, True).response()

    # 运行命令，并返回结果
    def run(self, cmd: str, recv_all: bool = False) -> bytes:
        """ run command """
        # 如果命令长度为0，则抛出数值错误
        if len(cmd) == 0:
            raise ValueError('no command specified')
        # 发送命令请求，并返回响应结果
        return self.request(cmd, True).response(recv_all)

    # 切换到指定设备
    def device(self, device_id: str = None) -> Session:
        """ switch to a device """
        # 设置设备ID
        self.device_id = device_id
        # 如果设备ID为空，则发送主机传输任意设备请求
        if device_id is None:
            return self.request('host:transport-any')
        # 否则发送主机传输指定设备请求
        else:
            return self.request('host:transport:' + device_id)

    # 连接设备
    def connect(self, device: str, throw_error: bool = False) -> None:
        """ connect device [ip:port] """
        # 发送连接设备请求，并记录响应结果
        resp = self.request(f'host:connect:{device}').response()
        logger.debug(f'adb connect {device}: {repr(resp)}')
        # 如果 throw_error 为真且响应结果包含 'unable' 或 'cannot'，则抛出运行时错误
        if throw_error and (b'unable' in resp or b'cannot' in resp):
            raise RuntimeError(repr(resp))

    # 断开设备连接
    def disconnect(self, device: str, throw_error: bool = False) -> None:
        """ disconnect device [ip:port] """
        # 发送断开设备连接请求，并记录响应结果
        resp = self.request(f'host:disconnect:{device}').response()
        logger.debug(f'adb disconnect {device}: {repr(resp)}')
        # 如果 throw_error 为真且响应结果包含 'unable' 或 'cannot'，则抛出运行时错误
        if throw_error and (b'unable' in resp or b'cannot' in resp):
            raise RuntimeError(repr(resp))

    # 返回adb服务器知道的设备列表
    def devices_list(self) -> list[tuple[str, str]]:
        """ returns list of devices that the adb server knows """
        # 发送获取设备列表请求，并解码响应结果
        resp = self.request('host:devices').response().decode(errors='ignore')
        # 将响应结果按行分割，然后按制表符分割，组成设备列表
        devices = [tuple(line.split('\t')) for line in resp.splitlines()]
        logger.debug(devices)
        # 返回设备列表
        return devices
    # 将数据推送到设备
    def push(self, target_path: str, target: bytes, mode=0o100755, mtime: int = None):
        """ push data to device """
        # 发送同步请求
        self.request('sync:', True)
        # 构建请求数据
        request = b'%s,%d' % (target_path.encode(), mode)
        # 发送请求数据
        self.sock.send(b'SEND' + struct.pack('<I', len(request)) + request)
        # 创建缓冲区
        buf = bytearray(65536+8)
        # 设置缓冲区前4个字节为'DATA'
        buf[0:4] = b'DATA'
        idx = 0
        # 循环发送数据
        while idx < len(target):
            # 获取要发送的内容
            content = target[idx:idx+65536]
            content_len = len(content)
            idx += content_len
            # 设置缓冲区中4-8字节为内容长度
            buf[4:8] = struct.pack('<I', content_len)
            # 将内容写入缓冲区
            buf[8:8+content_len] = content
            # 发送缓冲区中的数据
            self.sock.sendall(bytes(buf[0:8+content_len]))
        # 如果未指定修改时间，则设置为当前时间
        if mtime is None:
            mtime = int(time.time())
        # 发送修改时间
        self.sock.send(b'DONE' + struct.pack('<I', mtime))
        # 接收响应
        response = self.sock.recv_exactly(8)
        # 如果响应不是'OKAY'，则抛出运行时错误
        if response[:4] != b'OKAY':
            raise RuntimeError('push failed')
```