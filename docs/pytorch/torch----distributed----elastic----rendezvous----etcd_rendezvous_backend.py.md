# `.\pytorch\torch\distributed\elastic\rendezvous\etcd_rendezvous_backend.py`

```py
# mypy: allow-untyped-defs
# 以上为 mypy 工具的配置，允许未标注类型的函数定义
# Copyright (c) Facebook, Inc. and its affiliates.
# 版权声明，此代码受 BSD 风格许可证保护，详见根目录下的 LICENSE 文件

import binascii
# 导入 binascii 模块，用于二进制数据和 ASCII 数据之间的转换
from base64 import b64decode, b64encode
# 导入 base64 模块的 b64decode 和 b64encode 函数，用于 Base64 编解码
from typing import cast, Optional, Tuple
# 导入类型提示相关的模块，用于类型检查和提示

import urllib3.exceptions  # type: ignore[import]
# 导入 urllib3.exceptions 模块，用于处理 HTTP 请求异常（在类型检查时忽略）

from etcd import (  # type: ignore[import]
    Client as EtcdClient,
    EtcdAlreadyExist,
    EtcdCompareFailed,
    EtcdException,
    EtcdKeyNotFound,
    EtcdResult,
)
# 从 etcd 模块中导入 Client 和异常类，用于与 etcd 通信和处理 etcd 相关的异常

from torch.distributed import Store
# 导入 torch 分布式模块的 Store 类

from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
# 从当前包中导入 api 模块中定义的异常类和参数类
from .dynamic_rendezvous import RendezvousBackend, Token
# 从当前包中导入 dynamic_rendezvous 模块中的 RendezvousBackend 和 Token 类
from .etcd_store import EtcdStore
# 从当前包中导入 etcd_store 模块中的 EtcdStore 类
from .utils import parse_rendezvous_endpoint
# 从当前包中导入 utils 模块中的 parse_rendezvous_endpoint 函数，用于解析 rendezvous 端点

class EtcdRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
            用于与 etcd 通信的 ``etcd.Client`` 实例。
        run_id:
            The run id of the rendezvous.
            rendezvous 的运行 id。
        key_prefix:
            The path under which to store the rendezvous state in etcd.
            在 etcd 中存储 rendezvous 状态的路径前缀。
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
            rendezvous 状态的 TTL（生存时间）。如果未指定，默认为两小时。
    """

    _DEFAULT_TTL = 7200  # 2 hours
    # 默认的 rendezvous 状态 TTL，单位为秒，即 2 小时

    _client: EtcdClient
    _key: str
    _ttl: int
    # 类的私有属性，包括 etcd 客户端实例、存储 key、生存时间 TTL

    def __init__(
        self,
        client: EtcdClient,
        run_id: str,
        key_prefix: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")
        # 如果 run_id 为空，则抛出 ValueError 异常

        self._client = client
        # 初始化类的 etcd 客户端实例

        if key_prefix:
            self._key = key_prefix + "/" + run_id
        else:
            self._key = run_id
        # 根据 key_prefix 是否存在，初始化存储在 etcd 中的 key

        if ttl and ttl > 0:
            self._ttl = ttl
        else:
            self._ttl = self._DEFAULT_TTL
        # 初始化生存时间 TTL，若未指定则使用默认值

    @property
    def name(self) -> str:
        """See base class."""
        return "etcd-v2"
    # 返回当前后端的名称，实现基类中的属性

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        try:
            result = self._client.read(self._key)
        except EtcdKeyNotFound:
            return None
        except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc
        # 尝试从 etcd 中读取存储的状态信息，处理可能的异常情况

        return self._decode_state(result)
    # 返回当前存储在 etcd 中的状态信息，若不存在则返回 None

    def set_state(
        self, state: bytes, token: Optional[Token] = None
        # 将指定的状态信息写入到 etcd 中
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""
        # 将状态对象进行 Base64 编码并转换成字符串形式
        base64_state = b64encode(state).decode()

        kwargs = {}

        def get_state():
            # 获取当前对象的状态，如果存在则返回状态元组，标记为未修改
            result = self.get_state()
            if result is not None:
                tmp = *result, False
                # Python 3.6 不支持在 return 语句中进行元组解构
                return tmp
            return None

        if token:
            try:
                # 尝试将 token 转换为整数类型
                token = int(token)
            except ValueError:
                # 如果转换失败，则调用 get_state() 返回当前状态
                return get_state()

        if token:
            # 如果 token 存在，则将其作为参数传递给 kwargs 的 "prevIndex" 键
            kwargs["prevIndex"] = token
        else:
            # 如果 token 不存在，则将 "prevExist" 设置为 False
            kwargs["prevExist"] = False

        try:
            # 尝试将状态写入 etcd 数据库，并获取写入结果
            result = self._client.write(self._key, base64_state, self._ttl, **kwargs)
        except (EtcdAlreadyExist, EtcdCompareFailed):
            # 处理写入过程中已存在或比较失败的异常情况，结果设为 None
            result = None
        except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
            # 处理 etcd 异常和连接超时异常，抛出 RendezvousConnectionError 异常
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        if result is None:
            # 如果写入结果为 None，则调用 get_state() 返回当前状态
            return get_state()

        # 解码从 etcd 返回的结果，并标记为已修改
        tmp = *self._decode_state(result), True
        return tmp

    def _decode_state(self, result: EtcdResult) -> Tuple[bytes, Token]:
        # 从 etcd 返回的结果中获取 Base64 编码的状态字符串
        base64_state = result.value.encode()

        try:
            # 尝试解码 Base64 编码的状态字符串
            state = b64decode(base64_state)
        except binascii.Error as exc:
            # 处理解码错误，抛出 RendezvousStateError 异常
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        # 返回解码后的状态对象及修改索引
        return state, result.modifiedIndex
def _create_etcd_client(params: RendezvousParameters) -> EtcdClient:
    # 解析并获取 RendezvousParameters 中的 etcd 服务端地址和端口号
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)

    # 读取超时时间设置，如果小于等于 0 则抛出数值错误异常
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # 获取通信协议，若不是 http 或 https 则抛出数值错误异常
    protocol = params.get("protocol", "http").strip().lower()
    if protocol != "http" and protocol != "https":
        raise ValueError("The protocol must be HTTP or HTTPS.")

    # 获取 SSL 客户端证书路径，如果存在则获取对应的私钥路径，并组成元组 (cert, cert_key)
    ssl_cert = params.get("ssl_cert")
    if ssl_cert:
        ssl_cert_key = params.get("ssl_cert_key")
        if ssl_cert_key:
            # Etcd 客户端要求证书私钥作为 `cert` 元组的第二个元素
            ssl_cert = (ssl_cert, ssl_cert_key)

    # 获取根 SSL 授权证书路径
    ca_cert = params.get("ca_cert")

    try:
        # 创建 EtcdClient 实例，传入相关参数
        return EtcdClient(
            host,
            port,
            read_timeout=read_timeout,
            protocol=protocol,
            cert=ssl_cert,
            ca_cert=ca_cert,
            allow_reconnect=True,
        )
    except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
        # 捕获 EtcdException 和 TimeoutError 异常，抛出自定义的连接错误异常
        raise RendezvousConnectionError(
            "The connection to etcd has failed. See inner exception for details."
        ) from exc
    # 使用给定参数创建一个 Etcd 客户端
    client = _create_etcd_client(params)
    
    # 使用创建的 Etcd 客户端、运行 ID 和键前缀创建一个 EtcdRendezvousBackend 对象
    backend = EtcdRendezvousBackend(
        client, params.run_id, key_prefix="/torch/elastic/rendezvous"
    )
    
    # 使用相同的 Etcd 客户端和指定的存储路径创建一个 EtcdStore 对象
    store = EtcdStore(client, "/torch/elastic/store")
    
    # 返回创建的 EtcdRendezvousBackend 和 EtcdStore 对象作为结果
    return backend, store
```