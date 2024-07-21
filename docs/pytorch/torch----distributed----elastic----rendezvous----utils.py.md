# `.\pytorch\torch\distributed\elastic\rendezvous\utils.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 引入必要的模块和库
import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union

# 导出的模块列表
__all__ = ["parse_rendezvous_endpoint"]

# 解析会面配置字符串，提取键值对
def _parse_rendezvous_config(config_str: str) -> Dict[str, str]:
    """Extract key-value pairs from a rendezvous configuration string.

    Args:
        config_str:
            A string in format <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: Dict[str, str] = {}

    config_str = config_str.strip()
    if not config_str:
        return config

    key_values = config_str.split(",")
    for kv in key_values:
        key, *values = kv.split("=", 1)

        key = key.strip()
        if not key:
            raise ValueError(
                "The rendezvous configuration string must be in format "
                "<key1>=<value1>,...,<keyN>=<valueN>."
            )

        value: Optional[str]
        if values:
            value = values[0].strip()
        else:
            value = None
        if not value:
            raise ValueError(
                f"The rendezvous configuration option '{key}' must have a value specified."
            )

        config[key] = value
    return config

# 尝试解析端口号字符串，返回整数端口号或者 None
def _try_parse_port(port_str: str) -> Optional[int]:
    """Try to extract the port number from ``port_str``."""
    if port_str and re.match(r"^[0-9]{1,5}$", port_str):
        return int(port_str)
    return None

# 解析会面终端点，提取主机名和端口号
def parse_rendezvous_endpoint(
    endpoint: Optional[str], default_port: int
) -> Tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """
    if endpoint is not None:
        endpoint = endpoint.strip()

    if not endpoint:
        return ("localhost", default_port)

    # An endpoint that starts and ends with brackets represents an IPv6 address.
    if endpoint[0] == "[" and endpoint[-1] == "]":
        host, *rest = endpoint, *[]
    else:
        host, *rest = endpoint.rsplit(":", 1)

    # Sanitize the IPv6 address.
    if len(host) > 1 and host[0] == "[" and host[-1] == "]":
        host = host[1:-1]

    if len(rest) == 1:
        port = _try_parse_port(rest[0])
        if port is None or port >= 2**16:
            raise ValueError(
                f"The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                "between 0 and 65536."
            )
    else:
        port = default_port
    # 如果主机名不符合指定的格式要求，抛出数值错误异常
    if not re.match(r"^[\w\.:-]+$", host):
        raise ValueError(
            f"The hostname of the rendezvous endpoint '{endpoint}' must be a dot-separated list of "
            "labels, an IPv4 address, or an IPv6 address."
        )
    
    # 返回主机名和端口号
    return host, port
# 检查传入的主机名是否匹配本机的主机名或IP地址
def _matches_machine_hostname(host: str) -> bool:
    """Indicate whether ``host`` matches the hostname of this machine.

    This function compares ``host`` to the hostname as well as to the IP
    addresses of this machine. Note that it may return a false negative if this
    machine has CNAME records beyond its FQDN or IP addresses assigned to
    secondary NICs.
    """
    # 如果主机名是 "localhost"，则返回 True
    if host == "localhost":
        return True

    try:
        addr = ipaddress.ip_address(host)  # 尝试解析主机名为 IP 地址对象
    except ValueError:
        addr = None

    # 如果解析成功且是环回地址，则返回 True
    if addr and addr.is_loopback:
        return True

    try:
        # 获取主机名对应的地址信息列表
        host_addr_list = socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
        )
    except (ValueError, socket.gaierror) as _:
        host_addr_list = []

    # 提取出主机名对应的 IP 地址列表
    host_ip_list = [host_addr_info[4][0] for host_addr_info in host_addr_list]

    # 获取本机的主机名
    this_host = socket.gethostname()
    # 如果主机名匹配本机的主机名，则返回 True
    if host == this_host:
        return True

    # 获取本机的地址信息列表
    addr_list = socket.getaddrinfo(
        this_host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
    )
    for addr_info in addr_list:
        # 如果地址信息包含 FQDN，并且与 `host` 相同，则返回 True
        if addr_info[3] and addr_info[3] == host:
            return True

        # 否则，如果 `host` 表示一个 IP 地址，并且与本机的 IP 地址匹配，则返回 True
        if addr and addr_info[4][0] == str(addr):
            return True

        # 如果主机名对应的 IP 地址在主机 IP 地址列表中，则返回 True
        if addr_info[4][0] in host_ip_list:
            return True

    # 如果以上条件都不满足，则返回 False
    return False


def _delay(seconds: Union[float, Tuple[float, float]]) -> None:
    """Suspend the current thread for ``seconds``.

    Args:
        seconds:
            Either the delay, in seconds, or a tuple of a lower and an upper
            bound within which a random delay will be picked.
    """
    if isinstance(seconds, tuple):
        seconds = random.uniform(*seconds)  # 如果 `seconds` 是一个范围，则选择其内的随机延迟
    # 忽略小于 10 毫秒的延迟请求
    if seconds >= 0.01:
        time.sleep(seconds)  # 暂停当前线程指定的秒数


class _PeriodicTimer:
    """Represent a timer that periodically runs a specified function.

    Args:
        interval:
            The interval, in seconds, between each run.
        function:
            The function to run.
    """

    # 定时器的状态保存在一个单独的上下文对象中，以避免定时器和后台线程之间的引用循环。
    class _Context:
        interval: float
        function: Callable[..., None]
        args: Tuple[Any, ...]
        kwargs: Dict[str, Any]
        stop_event: Event

    _name: Optional[str]
    _thread: Optional[Thread]
    _finalizer: Optional[weakref.finalize]

    # 定时器和后台线程之间共享的上下文对象。
    _ctx: _Context

    def __init__(
        self,
        interval: timedelta,
        function: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._name = None  # 初始化定时器的名称为 None

        self._ctx = self._Context()  # 创建上下文对象用于存储定时器的状态和参数
        self._ctx.interval = interval.total_seconds()  # 设置定时器的时间间隔，单位为秒
        self._ctx.function = function  # type: ignore[assignment] 设置定时器要执行的函数
        self._ctx.args = args or ()  # 设置定时器函数的位置参数，默认为空元组
        self._ctx.kwargs = kwargs or {}  # 设置定时器函数的关键字参数，默认为空字典
        self._ctx.stop_event = Event()  # 创建事件对象，用于控制定时器停止的信号

        self._thread = None  # 初始化线程对象为 None
        self._finalizer = None  # 初始化弱引用终结器对象为 None

    @property
    def name(self) -> Optional[str]:
        """Get the name of the timer."""
        return self._name  # 返回定时器的名称

    def set_name(self, name: str) -> None:
        """Set the name of the timer.

        The specified name will be assigned to the background thread and serves
        for debugging and troubleshooting purposes.
        """
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._name = name  # 设置定时器的名称为指定的 name

    def start(self) -> None:
        """Start the timer."""
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._thread = Thread(
            target=self._run,
            name=self._name or "PeriodicTimer",  # 设置线程的名称为定时器的名称，如果未设置则为默认值
            args=(self._ctx,),  # 将定时器的上下文作为参数传递给线程的运行函数
            daemon=True,  # 设置线程为守护线程
        )

        # We avoid using a regular finalizer (a.k.a. __del__) for stopping the
        # timer as joining a daemon thread during the interpreter shutdown can
        # cause deadlocks. The weakref.finalize is a superior alternative that
        # provides a consistent behavior regardless of the GC implementation.
        self._finalizer = weakref.finalize(
            self, self._stop_thread, self._thread, self._ctx.stop_event
        )

        # We do not attempt to stop our background thread during the interpreter
        # shutdown. At that point we do not even know whether it still exists.
        self._finalizer.atexit = False  # 设置终结器不在解释器退出时执行

        self._thread.start()  # 启动线程运行定时器

    def cancel(self) -> None:
        """Stop the timer at the next opportunity."""
        if self._finalizer:
            self._finalizer()  # 调用终结器来停止定时器

    @staticmethod
    def _run(ctx) -> None:
        while not ctx.stop_event.wait(ctx.interval):  # 循环执行直到收到停止信号
            ctx.function(*ctx.args, **ctx.kwargs)  # 调用定时器设定的函数执行

    @staticmethod
    def _stop_thread(thread, stop_event):
        stop_event.set()  # 设置停止事件，通知定时器线程停止

        thread.join()  # 等待线程结束
```