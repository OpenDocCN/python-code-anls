# `.\pytorch\test\distributed\elastic\test_control_plane.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统功能的模块
import pickle  # 导入处理 pickle 数据格式的模块
import socket  # 导入 socket 模块，用于网络通信
import tempfile  # 导入临时文件和目录创建的模块
from contextlib import contextmanager  # 导入上下文管理器模块

from urllib3.connection import HTTPConnection  # 导入 HTTP 连接的实现
from urllib3.connectionpool import HTTPConnectionPool  # 导入 HTTP 连接池的实现

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,  # 导入 Torch 分布式控制平面的工作器服务器套接字变量
    worker_main,  # 导入 Torch 分布式控制平面的工作器主函数
)
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase  # 导入 Torch 测试相关的实用函数和类


class UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path  # 初始化 Unix 套接字路径

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # 创建 Unix 套接字
        self.sock.connect(self.socket_path)  # 连接到指定的 Unix 套接字路径


class UnixHTTPConnectionPool(HTTPConnectionPool):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path  # 初始化 Unix 套接字路径

    def _new_conn(self):
        return UnixHTTPConnection(self.socket_path)  # 返回一个新的 Unix HTTP 连接


@contextmanager
def local_worker_server() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:  # 创建临时目录
        socket_path = os.path.join(tmpdir, "socket.sock")  # 拼接 Unix 套接字路径
        os.environ[TORCH_WORKER_SERVER_SOCKET] = socket_path  # 设置环境变量 TORCH_WORKER_SERVER_SOCKET

        with worker_main():  # 启动工作器主函数
            pool = UnixHTTPConnectionPool(socket_path)  # 创建 Unix HTTP 连接池
            yield pool  # 通过生成器向调用方提供连接池


class WorkerServerTest(TestCase):
    def test_worker_server(self) -> None:
        with local_worker_server() as pool:  # 使用本地工作器服务器上下文管理器
            resp = pool.request("GET", "/")  # 发送 GET 请求到工作器服务器的根路径
            self.assertEqual(resp.status, 200)  # 断言响应状态为 200
            self.assertEqual(
                resp.data,
                b"""<h1>torch.distributed.WorkerServer</h1>
<a href="/handler/">Handler names</a>
""",
            )  # 断言响应数据与预期 HTML 字节串相等

            resp = pool.request("POST", "/handler/ping")  # 发送 POST 请求到工作器服务器的 ping 路径
            self.assertEqual(resp.status, 200)  # 断言响应状态为 200
            self.assertEqual(resp.data, b"pong")  # 断言响应数据为 b"pong"

            resp = pool.request("GET", "/handler/")  # 发送 GET 请求到工作器服务器的 handler 路径
            self.assertEqual(resp.status, 200)  # 断言响应状态为 200
            self.assertIn("ping", json.loads(resp.data))  # 断言响应数据中包含 "ping" 键

            resp = pool.request("POST", "/handler/nonexistant")  # 发送 POST 请求到工作器服务器的不存在路径
            self.assertEqual(resp.status, 404)  # 断言响应状态为 404
            self.assertIn(b"Handler nonexistant not found:", resp.data)  # 断言响应数据中包含指定的错误消息

    @requires_cuda
    def test_dump_nccl_trace_pickle(self) -> None:
        with local_worker_server() as pool:  # 使用本地工作器服务器上下文管理器
            resp = pool.request("POST", "/handler/dump_nccl_trace_pickle")  # 发送 POST 请求到工作器服务器的 dump_nccl_trace_pickle 路径
            self.assertEqual(resp.status, 200)  # 断言响应状态为 200
            out = pickle.loads(resp.data)  # 反序列化响应数据
            self.assertIsInstance(out, dict)  # 断言反序列化后的数据类型为字典
            self.assertIn("version", out)  # 断言反序列化后的字典中包含 "version" 键

    @requires_cuda
    # 定义测试函数，测试在特定参数下将 nccl 跟踪信息转储为 pickle 格式的功能
    def test_dump_nccl_trace_pickle_with_params(self) -> None:
        # 在本地启动一个工作服务器，使用 with 语句确保资源的正确释放
        with local_worker_server() as pool:
            # 发起 POST 请求，请求将 nccl 跟踪信息转储为 pickle 格式，包含集合操作
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includeCollectives=true"
            )
            # 断言响应状态为 400，因为参数 key 包含大写字母
            self.assertEqual(resp.status, 400)
            # 发起 POST 请求，包含未知的查询参数键
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?unknownkey=true"
            )
            # 断言响应状态为 400，因为参数键未知
            self.assertEqual(resp.status, 400)
            # 发起 POST 请求，参数值不是布尔类型
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=notabool"
            )
            # 断言响应状态为 400，因为参数值不是布尔类型
            self.assertEqual(resp.status, 400)
            # 发起 POST 请求，参数值不是小写形式
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=True"
            )
            # 断言响应状态为 400，因为参数值不是小写形式
            self.assertEqual(resp.status, 400)
            # 发起 POST 请求，参数键和值都符合要求
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=true"
            )
            # 断言响应状态为 200，表明请求成功
            self.assertEqual(resp.status, 200)
            # 发起 POST 请求，包含多个参数键和值
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_pickle?includecollectives=true&includestacktraces=false&onlyactive=true",
            )
            # 断言响应状态为 200，表明请求成功
            self.assertEqual(resp.status, 200)

    # 定义测试函数，测试与 TCP 通信相关的功能
    def test_tcp(self) -> None:
        # 导入需要的模块和类
        import requests
        from torch._C._distributed_c10d import _WorkerServer

        # 创建 WorkerServer 实例，绑定到本地端口 1234
        server = _WorkerServer("", 1234)
        # 发起 GET 请求，请求服务器根路径
        out = requests.get("http://localhost:1234/handler/")
        # 断言响应状态码为 200，表明请求成功
        self.assertEqual(out.status_code, 200)

        # 关闭 WorkerServer 实例
        server.shutdown()

    # 定义测试函数，测试将回溯信息转储为文件的功能
    def test_dump_traceback(self) -> None:
        # 在本地启动一个工作服务器，使用 with 语句确保资源的正确释放
        with local_worker_server() as pool:
            # 发起 POST 请求，请求将当前回溯信息转储为文件
            resp = pool.request("POST", "/handler/dump_traceback")
            # 断言响应状态为 200，表明请求成功
            self.assertEqual(resp.status, 200)
            # 断言响应数据中包含特定的回溯信息提示
            self.assertIn(b"in test_dump_traceback\n", resp.data)
# 如果当前模块是主程序（而不是被导入的模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行程序的测试
    run_tests()
```