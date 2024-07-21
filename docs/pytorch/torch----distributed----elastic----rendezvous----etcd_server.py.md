# `.\pytorch\torch\distributed\elastic\rendezvous\etcd_server.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import atexit  # 导入 atexit 模块，用于注册退出处理函数
import logging  # 导入 logging 模块，用于记录日志
import os  # 导入 os 模块，提供对操作系统功能的访问
import shlex  # 导入 shlex 模块，用于解析 shell 命令
import shutil  # 导入 shutil 模块，提供高级文件操作功能
import socket  # 导入 socket 模块，提供网络通信功能
import subprocess  # 导入 subprocess 模块，用于创建和管理子进程
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import time  # 导入 time 模块，提供时间相关的功能
from typing import Optional, TextIO, Union  # 导入类型提示相关的功能

try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    pass  # 如果找不到 etcd 模块，忽略异常继续执行

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def find_free_port():
    """
    Find a free port and binds a temporary socket to it so that the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise a ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket binded to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)  # 创建一个新的套接字
            s.bind(("localhost", 0))  # 将套接字绑定到一个随机可用的端口
            s.listen(0)  # 开始监听连接请求
            return s  # 返回绑定到的套接字对象
        except OSError as e:
            s.close()  # 如果绑定失败，关闭套接字
            print(f"Socket creation attempt failed: {e}")  # 打印错误信息
    raise RuntimeError("Failed to create a socket")  # 如果找不到可用端口，抛出运行时错误


def stop_etcd(subprocess, data_dir: Optional[str] = None):
    if subprocess and subprocess.poll() is None:
        logger.info("stopping etcd server")
        subprocess.terminate()  # 终止 etcd 进程
        subprocess.wait()  # 等待进程结束

    if data_dir:
        logger.info("deleting etcd data dir: %s", data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)  # 删除指定的 etcd 数据目录


class EtcdServer:
    """
    .. note:: tested on etcd server v3.4.3.

    Starts and stops a local standalone etcd server on a random free
    port. Useful for single node, multi-worker launches or testing,
    where a sidecar etcd server is more convenient than having to
    separately setup an etcd server.

    This class registers a termination handler to shutdown the etcd
    subprocess on exit. This termination handler is NOT a substitute for
    calling the ``stop()`` method.

    The following fallback mechanism is used to find the etcd binary:

    1. Uses env var TORCHELASTIC_ETCD_BINARY_PATH
    2. Uses ``<this file root>/bin/etcd`` if one exists
    3. Uses ``etcd`` from ``PATH``

    Usage
    """

    def __init__(self):
        pass  # 初始化方法暂时未实现

    def start(self):
        pass  # 启动 etcd 服务器的方法暂时未实现

    def stop(self):
        pass  # 停止 etcd 服务器的方法暂时未实现

    def wait_until_ready(self, timeout_sec: Optional[float] = None):
        pass  # 等待 etcd 服务器就绪的方法暂时未实现
    # 创建一个 EtcdServer 实例，指定 etcd 服务器的二进制文件路径、端口号和数据目录路径
    server = EtcdServer("/usr/bin/etcd", 2379, "/tmp/default.etcd")
    # 启动 etcd 服务器进程
    server.start()
    # 获取与 etcd 服务器通信的客户端对象
    client = server.get_client()
    # 使用客户端进行操作（这部分代码没有具体展示操作细节）
    # 停止 etcd 服务器进程
    server.stop()

    Args:
        etcd_binary_path: etcd 服务器二进制文件的路径（参考上述的默认路径）
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._port = -1  # 默认端口设置为 -1
        self._host = "localhost"  # 默认主机设置为 localhost

        root = os.path.dirname(__file__)  # 获取当前文件所在目录
        default_etcd_bin = os.path.join(root, "bin/etcd")  # 默认 etcd 二进制文件路径
        # 获取环境变量 TORCHELASTIC_ETCD_BINARY_PATH 的值，作为 etcd 二进制文件的路径
        self._etcd_binary_path = os.environ.get(
            "TORCHELASTIC_ETCD_BINARY_PATH", default_etcd_bin
        )
        if not os.path.isfile(self._etcd_binary_path):
            self._etcd_binary_path = "etcd"  # 如果路径不是文件，则使用默认的 etcd

        # 设置基础数据目录路径，如果未提供则创建一个临时目录
        self._base_data_dir = (
            data_dir if data_dir else tempfile.mkdtemp(prefix="torchelastic_etcd_data")
        )
        self._etcd_cmd = None  # etcd 命令对象初始化为 None
        self._etcd_proc: Optional[subprocess.Popen] = None  # etcd 进程对象初始化为 None

    def _get_etcd_server_process(self) -> subprocess.Popen:
        # 如果 etcd 进程对象未初始化，则抛出运行时错误
        if not self._etcd_proc:
            raise RuntimeError(
                "No etcd server process started. Call etcd_server.start() first"
            )
        else:
            return self._etcd_proc

    def get_port(self) -> int:
        """返回服务器正在运行的端口号。"""
        return self._port

    def get_host(self) -> str:
        """返回服务器正在运行的主机名。"""
        return self._host

    def get_endpoint(self) -> str:
        """返回 etcd 服务器的终端点（主机名:端口号）。"""
        return f"{self._host}:{self._port}"

    def start(
        self,
        timeout: int = 60,
        num_retries: int = 3,
        stderr: Union[int, TextIO, None] = None,
    ) -> None:
        """
        Start the server, and waits for it to be ready. When this function returns the sever is ready to take requests.

        Args:
            timeout: time (in seconds) to wait for the server to be ready
                before giving up.
            num_retries: number of retries to start the server. Each retry
                will wait for max ``timeout`` before considering it as failed.
            stderr: the standard error file handle. Valid values are
                `subprocess.PIPE`, `subprocess.DEVNULL`, an existing file
                descriptor (a positive integer), an existing file object, and
                `None`.

        Raises:
            TimeoutError: if the server is not ready within the specified timeout
        """
        curr_retries = 0
        while True:
            try:
                # 构建当前重试次数的数据目录
                data_dir = os.path.join(self._base_data_dir, str(curr_retries))
                # 创建数据目录，如果目录已存在则忽略
                os.makedirs(data_dir, exist_ok=True)
                # 调用 _start 方法启动服务器，并返回结果
                return self._start(data_dir, timeout, stderr)
            except Exception as e:
                curr_retries += 1
                # 停止 etcd 进程
                stop_etcd(self._etcd_proc)
                # 记录警告日志，表示启动 etcd 服务器失败，输出错误信息并进行重试
                logger.warning(
                    "Failed to start etcd server, got error: %s, retrying", str(e)
                )
                # 如果超过最大重试次数，则删除基础数据目录并抛出异常
                if curr_retries >= num_retries:
                    shutil.rmtree(self._base_data_dir, ignore_errors=True)
                    raise
        # 注册退出时关闭 etcd 进程和基础数据目录清理
        atexit.register(stop_etcd, self._etcd_proc, self._base_data_dir)

    def _start(
        self, data_dir: str, timeout: int = 60, stderr: Union[int, TextIO, None] = None
    ) -> None:
        # 查找空闲端口
        sock = find_free_port()
        sock_peer = find_free_port()
        # 获取服务器监听端口和对等节点端口
        self._port = sock.getsockname()[1]
        peer_port = sock_peer.getsockname()[1]

        # 构建启动 etcd 服务器的命令行参数列表
        etcd_cmd = shlex.split(
            " ".join(
                [
                    self._etcd_binary_path,
                    "--enable-v2",
                    "--data-dir",
                    data_dir,
                    "--listen-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--advertise-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--listen-peer-urls",
                    f"http://{self._host}:{peer_port}",
                ]
            )
        )

        # 记录信息日志，表示正在启动 etcd 服务器，并输出启动命令
        logger.info("Starting etcd server: [%s]", etcd_cmd)

        # 关闭 socket 连接
        sock.close()
        sock_peer.close()
        # 使用 subprocess 模块启动 etcd 进程，并指定 stderr 输出
        self._etcd_proc = subprocess.Popen(etcd_cmd, close_fds=True, stderr=stderr)
        # 等待服务器就绪
        self._wait_for_ready(timeout)

    def get_client(self):
        """Return an etcd client object that can be used to make requests to this server."""
        # 返回一个 etcd 客户端对象，用于向该服务器发出请求
        return etcd.Client(
            host=self._host, port=self._port, version_prefix="/v2", read_timeout=10
        )
    # 等待 etcd 服务器就绪，最长等待时间为 timeout 秒
    def _wait_for_ready(self, timeout: int = 60) -> None:
        # 创建 etcd 客户端对象，连接到指定的主机和端口，使用 /v2 版本前缀，读取超时为 5 秒
        client = etcd.Client(
            host=f"{self._host}", port=self._port, version_prefix="/v2", read_timeout=5
        )
        # 计算超时的终止时间点
        max_time = time.time() + timeout

        # 在超时时间内循环检查服务器是否就绪
        while time.time() < max_time:
            # 如果 etcd 服务器进程已经结束，则抛出运行时错误并显示退出码
            if self._get_etcd_server_process().poll() is not None:
                exitcode = self._get_etcd_server_process().returncode
                raise RuntimeError(
                    f"Etcd server process exited with the code: {exitcode}"
                )
            try:
                # 尝试获取 etcd 服务器的版本信息，若成功则表示服务器已经就绪，记录日志信息
                logger.info("etcd server ready. version: %s", client.version)
                return
            except Exception:
                # 若获取版本信息时发生异常，则等待 1 秒后重试
                time.sleep(1)
        # 若超时仍未检测到服务器就绪，则抛出超时异常
        raise TimeoutError("Timed out waiting for etcd server to be ready!")

    # 停止 etcd 服务器并清理自动生成的资源（如数据目录）
    def stop(self) -> None:
        """Stop the server and cleans up auto generated resources (e.g. data dir)."""
        # 记录停止方法被调用的日志信息
        logger.info("EtcdServer stop method called")
        # 调用 stop_etcd 函数停止 etcd 服务器，并清理指定的基础数据目录
        stop_etcd(self._etcd_proc, self._base_data_dir)
```