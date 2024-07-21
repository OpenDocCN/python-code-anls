# `.\pytorch\torch\multiprocessing\queue.py`

```py
# mypy: allow-untyped-defs
# 引入所需的模块和类
import io  # 导入 io 模块
import multiprocessing.queues  # 导入 multiprocessing 中的队列模块
import pickle  # 导入 pickle 模块
from multiprocessing.reduction import ForkingPickler  # 从 multiprocessing.reduction 模块导入 ForkingPickler 类


class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""

    def __init__(self, conn):
        self.conn = conn  # 初始化连接对象

    def send(self, obj):
        buf = io.BytesIO()  # 创建一个字节流缓冲区
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)  # 使用 ForkingPickler 序列化 obj 到 buf
        self.send_bytes(buf.getvalue())  # 发送 buf 中的字节数据到连接

    def recv(self):
        buf = self.recv_bytes()  # 接收连接中的字节数据
        return pickle.loads(buf)  # 反序列化 buf 中的数据并返回

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)  # 如果在自身属性中找不到，则委托给原始连接对象
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)  # 使用 ConnectionWrapper 包装 _reader
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)  # 使用 ConnectionWrapper 包装 _writer
        self._send = self._writer.send  # 设置发送方法为 _writer 的 send 方法
        self._recv = self._reader.recv  # 设置接收方法为 _reader 的 recv 方法


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)  # 如果 _reader 不是 ConnectionWrapper，则使用 ConnectionWrapper 包装
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)  # 如果 _writer 不是 ConnectionWrapper，则使用 ConnectionWrapper 包装
        super()._make_methods()  # 调用父类的 _make_methods 方法，类型忽略相关警告
```