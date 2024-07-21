# `.\pytorch\torch\distributed\rendezvous.py`

```
# mypy: allow-untyped-defs
try:
    # 尝试导入必要的模块和函数
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    # 如果导入失败，抛出自定义的 ImportError 异常
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import numbers  # 导入 numbers 模块
import os  # 导入 os 模块
import sys  # 导入 sys 模块
from datetime import timedelta  # 从 datetime 模块导入 timedelta 类型
from typing import Callable, Dict, Iterator, Optional, Tuple  # 导入类型提示相关的内容

from torch.distributed import FileStore, PrefixStore, Store, TCPStore  # 从 torch.distributed 模块导入特定类

from .constants import default_pg_timeout  # 从当前包中的 constants 模块导入 default_pg_timeout 常量


# 定义一个全局变量 _rendezvous_handlers，用于存储不同的会合处理函数
_rendezvous_handlers: Dict[str, Callable[..., Iterator[Tuple[Store, int, int]]]] = {}

# 定义 __all__ 列表，用于指定在使用 from package import * 语句时导入的名称
__all__ = ["register_rendezvous_handler", "rendezvous"]


def register_rendezvous_handler(scheme, handler):
    """
    Register a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Args:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        # 如果已经注册过具有相同 scheme 的会合处理函数，则抛出 RuntimeError
        raise RuntimeError(f"Rendezvous handler for {scheme}:// already registered")
    _rendezvous_handlers[scheme] = handler


# Query will have format "rank=0&world_size=1" and is
# converted into {"rank": 0, "world_size": 1}
def _query_to_dict(query: str) -> Dict[str, str]:
    # 将查询字符串 query 解析为字典形式
    return {
        pair[0]: pair[1]
        for pair in (pair.split("=") for pair in filter(None, query.split("&")))
    }


def _get_use_libuv_from_query_dict(query_dict: Dict[str, str]) -> bool:
    # 从查询字典中获取 use_libuv 参数的值，若未指定则默认从环境变量 USE_LIBUV 中获取
    # 若值为 "1"，则返回 True，表示使用 libuv 后端；否则返回 False
    return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "1")) == "1"


def _rendezvous_helper(url: str, rank: int, world_size_opt: Optional[int], **kwargs):
    # 解析 URL 字符串
    result = urlparse(url)
    if world_size_opt is None:
        # 若未指定 world_size_opt，则设定默认值 -1
        world_size = -1
        if result.scheme == "env":
            # 如果 URL 方案为 "env"，则从环境变量中获取 rank 和 world_size 的值
            rank = int(os.environ.get("RANK", rank))
            # 如果环境变量中不存在 WORLD_SIZE，则表示是动态组，world_size 设为默认值 -1
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        # 若指定了 world_size_opt 参数，则直接使用该值
        world_size = world_size_opt
    # 检查排名、世界大小和世界大小选项是否有任何一个不为默认值-1
    if rank != -1 or world_size != -1 or world_size_opt is None:
        # 将查询字符串转换为字典形式
        query_dict = _query_to_dict(result.query)
        # 断言排名和世界大小不在查询字典中，否则抛出异常
        assert (
            "rank" not in query_dict and "world_size" not in query_dict
        ), f"The url: {url} has node-specific arguments(rank, world_size) already."
        # 如果排名不为默认值-1，则将排名添加到查询字典中
        if rank != -1:
            query_dict["rank"] = str(rank)
        # 如果世界大小不为默认值-1，或者世界大小选项为None，则将世界大小添加到查询字典中
        if world_size != -1 or world_size_opt is None:
            query_dict["world_size"] = str(world_size)
        # 更新结果对象的查询部分，将字典形式的查询字符串重新组合成字符串形式
        result = result._replace(
            query=f"{'&'.join([f'{k}={v}' for k, v in query_dict.items()])}"
        )
        # 重新构建URL
        url = urlunparse(result)

    # 如果结果对象的方案不在预定义的会合处理程序列表中，则抛出运行时异常
    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f"No rendezvous handler for {result.scheme}://")
    # 返回根据结果对象方案调用对应会合处理程序的结果
    return _rendezvous_handlers[result.scheme](url, **kwargs)
# 定义 rendezvous 函数，用于协调分布式任务的启动
def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    # 检查 url 是否为字符串类型，否则抛出运行时错误
    if not isinstance(url, (str, bytes)):
        raise RuntimeError(f"`url` must be a string. {type(url)}: {url}")

    # 检查 rank 是否为整数类型，否则抛出运行时错误
    if not isinstance(rank, numbers.Integral):
        raise RuntimeError(f"`rank` must be an integer. {rank}")

    # 检查 world_size 是否为整数类型，否则抛出运行时错误
    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError(f"`world_size` must be an integer. {world_size}")

    # 调用辅助函数 _rendezvous_helper 进行实际的协调操作
    return _rendezvous_helper(url, rank, world_size, **kwargs)


# 从给定的 backend_options 创建存储对象，使用 rank 作为参数
def _create_store_from_options(backend_options, rank):
    # 调用 _rendezvous_helper 获取存储对象和其他信息，忽略其余返回值
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store


# 返回一个描述 rendezvous 错误的 ValueError
def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)


# 处理 file:// 协议的 rendezvous 情况
def _file_rendezvous_handler(url: str, **kwargs):
    # 定义内部函数 _error，用于生成特定的 rendezvous 错误
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    # 解析给定的 URL
    result = urlparse(url)
    path = result.path

    # 在 Windows 平台上处理路径
    if sys.platform == "win32":
        import urllib.request

        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # 规范化路径，避免空字符串变成 "."
            path = os.path.normpath(path)

    # 如果路径为空，抛出路径缺失的错误
    if not path:
        raise _error("path missing")

    # 将查询字符串转换为字典形式
    query_dict = _query_to_dict(result.query)

    # 检查查询参数中是否包含 rank 和 world_size
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    # 从查询参数中获取 rank 和 world_size 的整数值
    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])

    # 创建一个 FileStore 对象，用于协调文件系统上的分布式存储
    store = FileStore(path, world_size)

    # 返回生成器，生成存储对象、rank 和 world_size 的元组
    yield (store, rank, world_size)

    # 如果配置无效，抛出运行时错误
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


# 返回当前是否应使用代理存储的布尔值
def _torchelastic_use_agent_store() -> bool:
    return os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None) == str(True)


# 创建一个 c10d Store 对象，基于给定的参数
def _create_c10d_store(
    hostname, port, rank, world_size, timeout, use_libuv=True
) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether we need to re-use agent store.

    The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    By default, the TCPStore server uses the asynchronous implementation
    ``LibUVStoreDaemon`` which utilizes libuv.

    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that
    the agent leader (node rank 0) hosts the TCPStore server (for which the
    endpoint is specified by the given ``hostname:port``). Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host
    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname
    and port are correctly passed via ``hostname`` and ``port``. All
    non-zero ranks will create and return a TCPStore client.
    """
    # 检查端口是否为 uint16_t 类型（16 位无符号整数）
    if not 0 <= port < 2**16:
        # 如果端口不在有效范围内，抛出数值错误异常并显示当前端口值
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    # 检查是否使用 TorchElastic 的代理存储
    if _torchelastic_use_agent_store():
        # 获取当前的重启尝试次数
        attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
        # 创建一个 TCPStore 对象，用于存储通信
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        # 返回一个 PrefixStore 对象，将 TCPStore 作为其基础存储，路径包含当前尝试的编号
        return PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
    else:
        # 确定是否由 rank 为 0 的进程启动守护进程
        start_daemon = rank == 0
        # 返回一个 TCPStore 对象，用于存储通信
        return TCPStore(
            hostname,
            port,
            world_size,
            start_daemon,
            timeout,
            multi_tenant=True,
            use_libuv=use_libuv,
        )
# 定义处理 TCP 协议的 rendezvous handler 函数，接收 URL、超时时间和其他参数
def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    # 定义内部函数用于返回错误消息
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    # 解析给定的 URL
    result = urlparse(url)
    # 如果 URL 中没有指定端口号，则抛出错误
    if not result.port:
        raise _error("port number missing")
    # 将查询字符串转换为字典形式
    query_dict = _query_to_dict(result.query)
    # 如果查询字符串中缺少 "rank" 参数，则抛出错误
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    # 如果查询字符串中缺少 "world_size" 参数，则抛出错误
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    # 将 "rank" 和 "world_size" 参数从查询字典中解析为整数
    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    # 从查询字典中获取是否使用 libuv，若未指定则默认为 False
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    # 确保解析后的主机名不为空
    assert result.hostname is not None

    # 创建 c10d 存储对象，用于 TCP 协议的通信
    store = _create_c10d_store(
        result.hostname, result.port, rank, world_size, timeout, use_libuv
    )

    # 返回生成器对象，包含 store 对象、rank 和 world_size
    yield (store, rank, world_size)

    # 如果配置无效，则抛出运行时错误
    raise RuntimeError("Unable to perform re-rendezvous using tcp:// method")


# 定义处理环境变量的 rendezvous handler 函数，接收 URL、超时时间和其他参数
def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    # 定义内部函数用于返回错误消息
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    # 定义内部函数用于返回环境变量相关的错误消息
    def _env_error(var):
        return _error(f"environment variable {var} expected, but not set")

    # 定义内部函数用于获取环境变量值或者抛出错误
    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    # 解析给定的 URL
    result = urlparse(url)
    # 将查询字符串转换为字典形式
    query_dict = _query_to_dict(result.query)

    # 定义变量用于存储 rank、world_size、master_port 和 master_addr
    rank: int
    world_size: int
    master_port: int
    master_addr: str

    # 如果查询字符串中包含 "rank" 参数，则将其解析为整数；否则从环境变量 RANK 中获取
    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    # 如果查询字符串中包含 "world_size" 参数，则将其解析为整数；否则从环境变量 WORLD_SIZE 中获取
    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    # 从环境变量 MASTER_ADDR 中获取 master 地址
    master_addr = _get_env_or_raise("MASTER_ADDR")
    # 从环境变量 MASTER_PORT 中获取 master 端口号，并将其解析为整数
    master_port = int(_get_env_or_raise("MASTER_PORT"))
    # 从查询字典中获取是否使用 libuv，若未指定则默认为 False
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    # 创建 c10d 存储对象，用于环境变量协议的通信
    store = _create_c10d_store(
        master_addr, master_port, rank, world_size, timeout, use_libuv
    )

    # 返回生成器对象，包含 store 对象、rank 和 world_size
    yield (store, rank, world_size)

    # 如果配置无效，则抛出运行时错误
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")


# 注册 tcp、env 和 file 协议的 rendezvous handler 函数
register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)
```