# `ZeroNet\src\lib\gevent_ws\__init__.py`

```
# 导入需要的模块
from gevent.pywsgi import WSGIHandler, _InvalidClientInput
from gevent.queue import Queue
import gevent
import hashlib
import base64
import struct
import socket
import time
import sys

# 定义常量
SEND_PACKET_SIZE = 1300
OPCODE_TEXT = 1
OPCODE_BINARY = 2
OPCODE_CLOSE = 8
OPCODE_PING = 9
OPCODE_PONG = 10
STATUS_OK = 1000
STATUS_PROTOCOL_ERROR = 1002
STATUS_DATA_ERROR = 1007
STATUS_POLICY_VIOLATION = 1008
STATUS_TOO_LONG = 1009

# 定义 WebSocket 类
class WebSocket:
    def __init__(self, socket):
        # 初始化 WebSocket 对象，设置初始属性
        self.socket = socket
        self.closed = False
        self.status = None
        self._receive_error = None
        self._queue = Queue()
        self.max_length = 10 * 1024 * 1024
        # 创建一个协程来监听 WebSocket 连接
        gevent.spawn(self._listen)

    # 设置最大消息长度
    def set_max_message_length(self, length):
        self.max_length = length
    # 监听 WebSocket 连接，接收并处理消息
    def _listen(self):
        try:
            while True:
                # 初始化结束标志和消息字节流
                fin = False
                message = bytearray()
                is_first_message = True
                start_opcode = None
                while not fin:
                    # 获取帧的有效载荷、操作码和结束标志
                    payload, opcode, fin = self._get_frame(max_length=self.max_length - len(message))
                    # 确保连续帧具有正确的信息
                    if not is_first_message and opcode != 0:
                        self._error(STATUS_PROTOCOL_ERROR)
                    if is_first_message:
                        if opcode not in (OPCODE_TEXT, OPCODE_BINARY):
                            self._error(STATUS_PROTOCOL_ERROR)
                        # 保存操作码
                        start_opcode = opcode
                    # 将有效载荷添加到消息中
                    message += payload
                    is_first_message = False
                message = bytes(message)
                # 如果操作码为文本类型，将消息解码为 UTF-8 文本
                if start_opcode == OPCODE_TEXT:  # UTF-8 text
                    try:
                        message = message.decode()
                    except UnicodeDecodeError:
                        self._error(STATUS_DATA_ERROR)
                # 将消息放入队列中
                self._queue.put(message)
        except Exception as e:
            # 发生异常时，设置连接关闭标志和接收错误信息
            self.closed = True
            self._receive_error = e
            self._queue.put(None)  # 确保错误信息被读取


    # 接收消息，如果队列不为空则立即返回消息
    def receive(self):
        if not self._queue.empty():
            return self.receive_nowait()
        if isinstance(self._receive_error, EOFError):
            return None
        if self._receive_error:
            raise self._receive_error
        self._queue.peek()
        return self.receive_nowait()


    # 立即返回队列中的消息
    def receive_nowait(self):
        ret = self._queue.get_nowait()
        if self._receive_error and not isinstance(self._receive_error, EOFError):
            raise self._receive_error
        return ret
    # 发送数据，如果连接已关闭则抛出 EOFError 异常
    def send(self, data):
        if self.closed:
            raise EOFError()
        # 如果数据是字符串，则将其编码成字节流后发送
        if isinstance(data, str):
            self._send_frame(OPCODE_TEXT, data.encode())
        # 如果数据是字节流，则直接发送
        elif isinstance(data, bytes):
            self._send_frame(OPCODE_BINARY, data)
        # 如果数据类型不是字符串或字节流，则抛出类型错误异常
        else:
            raise TypeError("Expected str or bytes, got " + repr(type(data)))


    # 从套接字中读取一个帧，自动处理 ping、pong 和 close 数据包
    def _get_frame(self, max_length):
        while True:
            # 读取帧的负载、操作码和结束标志
            payload, opcode, fin = self._read_frame(max_length=max_length)
            # 如果操作码是 ping，则发送一个 pong 帧
            if opcode == OPCODE_PING:
                self._send_frame(OPCODE_PONG, payload)
            # 如果操作码是 pong，则忽略
            elif opcode == OPCODE_PONG:
                pass
            # 如果操作码是 close，则处理关闭连接的逻辑
            elif opcode == OPCODE_CLOSE:
                # 如果负载长度大于等于2，则解析出状态码并设置连接状态为关闭
                if len(payload) >= 2:
                    self.status = struct.unpack("!H", payload[:2])[0]
                was_closed = self.closed
                self.closed = True
                # 如果之前连接未关闭，则发送一个关闭帧作为响应
                if not was_closed:
                    self.close(STATUS_OK)
                # 抛出 EOFError 异常
                raise EOFError()
            # 如果操作码不是 ping、pong 或 close，则返回负载、操作码和结束标志
            else:
                return payload, opcode, fin


    # 低级函数，使用 _get_frame 代替
    # 读取帧数据，最大长度为 max_length
    def _read_frame(self, max_length):
        # 读取帧头部，长度为 2 字节
        header = self._recv_exactly(2)

        # 检查是否为最后一个帧
        if not (header[1] & 0x80):
            self._error(STATUS_POLICY_VIOLATION)

        # 获取操作码和结束标志
        opcode = header[0] & 0xf
        fin = bool(header[0] & 0x80)

        # 获取有效载荷长度
        payload_length = header[1] & 0x7f
        if payload_length == 126:
            # 如果长度为 126，则读取接下来的 2 字节作为长度
            payload_length = struct.unpack("!H", self._recv_exactly(2))[0]
        elif payload_length == 127:
            # 如果长度为 127，则读取接下来的 8 字节作为长度
            payload_length = struct.unpack("!Q", self._recv_exactly(8))[0]

        # 控制帧的最大长度为 125
        if opcode in (OPCODE_PING, OPCODE_PONG):
            max_length = 125

        # 如果有效载荷长度超过最大长度，则报错
        if payload_length > max_length:
            self._error(STATUS_TOO_LONG)

        # 读取掩码和有效载荷数据
        mask = self._recv_exactly(4)
        payload = self._recv_exactly(payload_length)
        # 对有效载荷数据进行解码
        payload = self._unmask(payload, mask)

        # 返回解码后的有效载荷数据和操作码、结束标志
        return payload, opcode, fin


    # 从套接字接收指定长度的数据
    def _recv_exactly(self, length):
        buf = bytearray()
        while len(buf) < length:
            block = self.socket.recv(min(4096, length - len(buf)))
            if block == b"":
                raise EOFError()
            buf += block
        return bytes(buf)


    # 对有效载荷数据进行解码
    def _unmask(self, payload, mask):
        # 定义生成器函数，用于生成解码后的数据
        def gen(c):
            return bytes([x ^ c for x in range(256)])

        # 将有效载荷数据转换为字节数组
        payload = bytearray(payload)
        # 对每个字节进行解码
        payload[0::4] = payload[0::4].translate(gen(mask[0]))
        payload[1::4] = payload[1::4].translate(gen(mask[1]))
        payload[2::4] = payload[2::4].translate(gen(mask[2]))
        payload[3::4] = payload[3::4].translate(gen(mask[3]))
        # 返回解码后的有效载荷数据
        return bytes(payload)
    # 发送帧数据的私有方法，参数为操作码和数据
    def _send_frame(self, opcode, data):
        # 循环发送数据，每次发送 SEND_PACKET_SIZE 大小的数据
        for i in range(0, len(data), SEND_PACKET_SIZE):
            # 切片获取部分数据
            part = data[i:i + SEND_PACKET_SIZE]
            # 判断是否为最后一帧
            fin = int(i == (len(data) - 1) // SEND_PACKET_SIZE * SEND_PACKET_SIZE)
            # 构建帧头部
            header = bytes(
                [
                    (opcode if i == 0 else 0) | (fin << 7),  # 设置操作码和结束标志
                    min(len(part), 126)  # 设置数据长度
                ]
            )
            # 如果数据长度超过126，添加额外的长度信息
            if len(part) >= 126:
                header += struct.pack("!H", len(part))
            # 发送帧数据
            self.socket.sendall(header + part)
    
    # 抛出 EOFError 异常的私有方法，参数为状态码
    def _error(self, status):
        # 关闭连接
        self.close(status)
        # 抛出异常
        raise EOFError()
    
    # 关闭连接的公有方法，参数为状态码，默认为 STATUS_OK
    def close(self, status=STATUS_OK):
        # 标记连接已关闭
        self.closed = True
        try:
            # 发送关闭帧
            self._send_frame(OPCODE_CLOSE, struct.pack("!H", status))
        except (BrokenPipeError, ConnectionResetError):
            # 捕获异常，不做处理
            pass
        # 关闭套接字
        self.socket.close()
# WebSocketHandler 类继承自 WSGIHandler 类
class WebSocketHandler(WSGIHandler):
    # 处理结果的方法
    def process_result(self):
        # 如果环境中包含 wsgi.websocket
        if "wsgi.websocket" in self.environ:
            # 如果结果为空，则返回
            if self.result is None:
                return
            # 为了兼容 werkzeug，需要刷新结果
            for elem in self.result:
                pass
        else:
            # 如果环境中不包含 wsgi.websocket，则调用父类的 process_result 方法
            super(WebSocketHandler, self).process_result()

    # 版本属性的 getter 方法
    @property
    def version(self):
        # 如果环境为空，则返回 None
        if not self.environ:
            return None
        # 返回 HTTP_SEC_WEBSOCKET_VERSION 对应的值
        return self.environ.get('HTTP_SEC_WEBSOCKET_VERSION')
```