# `.\pytorch\torch\distributed\elastic\rendezvous\api.py`

```
# 设置 mypy 选项允许未类型化的定义
# 版权声明，禁止复制和修改
# 此源代码受 BSD 风格许可证保护，详见根目录下的 LICENSE 文件
# 导入必要的模块
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional
# 从 torch.distributed 中导入 Store 类
from torch.distributed import Store
# 从 torch.distributed.elastic.utils.distributed 中导入 get_free_port 函数
from torch.distributed.elastic.utils.distributed import get_free_port as _get_free_port

# 定义导出的类和变量
__all__ = [
    "RendezvousClosedError",
    "RendezvousConnectionError",
    "RendezvousError",
    "RendezvousGracefulExitError",
    "RendezvousHandler",
    "RendezvousHandlerCreator",
    "RendezvousHandlerRegistry",
    "RendezvousInfo",
    "RendezvousParameters",
    "RendezvousStateError",
    "RendezvousStoreInfo",
    "RendezvousTimeoutError",
    "rendezvous_handler_registry",
]

# 定义异常类
class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""

# 派生自 RendezvousError 的异常类
class RendezvousClosedError(RendezvousError):
    """Raised when a rendezvous is closed."""

# 派生自 RendezvousError 的异常类
class RendezvousTimeoutError(RendezvousError):
    """Raised when a rendezvous did not complete on time."""

# 派生自 RendezvousError 的异常类
class RendezvousConnectionError(RendezvousError):
    """Raised when the connection to a rendezvous backend has failed."""

# 派生自 RendezvousError 的异常类
class RendezvousStateError(RendezvousError):
    """Raised when the state of a rendezvous is corrupt."""

# 派生自 RendezvousError 的异常类
class RendezvousGracefulExitError(RendezvousError):
    """Raised when node wasn't not included in rendezvous and gracefully exits.

    Exception is a mechanism to exit the stack, however does not mean a failure.
    """

# 数据类，保存存储地址和端口，用于引导训练者分布式通信
@dataclass
class RendezvousStoreInfo:
    """Store address and port that can be used to bootstrap trainer distributed comms"""

    # 类变量，指定主地址和端口的键名
    MASTER_ADDR_KEY: ClassVar[str] = "MASTER_ADDR"
    MASTER_PORT_KEY: ClassVar[str] = "MASTER_PORT"
    # 主地址和主端口
    master_addr: str
    master_port: int

    # 工厂方法，找到 rank0 主机上未使用的新端口，并获取所有排名的地址和端口信息
    @staticmethod
    def build(rank: int, store: Store) -> "RendezvousStoreInfo":
        """Factory method, finds unused new port on rank0 host and addr/port info with all ranks.

        If master_addr/master_port is knowns (useful when sharing existing tcp store server) use the constructor.
        """
        # 如果当前进程为排名为0的进程
        if rank == 0:
            # 获取主机名和一个空闲端口
            addr = socket.getfqdn()
            port = _get_free_port()
            # 设置主地址和端口信息到存储中
            store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, addr.encode(encoding="UTF-8"))  # type: ignore[arg-type]
            store.set(RendezvousStoreInfo.MASTER_PORT_KEY, str(port).encode(encoding="UTF-8"))  # type: ignore[arg-type]

        # 从存储中获取主地址和端口信息，并解码成字符串和整数格式
        addr = store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode(encoding="UTF-8")
        port = int(store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode(encoding="UTF-8"))
        # 返回 RendezvousStoreInfo 对象
        return RendezvousStoreInfo(master_addr=addr, master_port=port)

# 用于存储关于 rendezvous 的信息的类
class RendezvousInfo:
    """Holds the information about the rendezvous."""
    def __init__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        bootstrap_store_info: RendezvousStoreInfo,
    ):
        # 初始化方法，用于创建一个新的实例
        self._store = store
        # 将传入的 store 参数赋值给实例的 _store 属性
        self._rank = rank
        # 将传入的 rank 参数赋值给实例的 _rank 属性
        self._world_size = world_size
        # 将传入的 world_size 参数赋值给实例的 _world_size 属性
        self._bootstrap_store_info = bootstrap_store_info
        # 将传入的 bootstrap_store_info 参数赋值给实例的 _bootstrap_store_info 属性

    @property
    def store(self) -> Store:
        """Store used by torchelastic control plane"""
        # 返回当前实例的 _store 属性，用于 torchelastic 控制平面
        return self._store

    @property
    def rank(self) -> int:
        """Rank within a group"""
        # 返回当前实例的 _rank 属性，表示在一个组内的排名
        return self._rank

    @property
    def world_size(self) -> int:
        """Global group size"""
        # 返回当前实例的 _world_size 属性，表示全局组大小
        return self._world_size

    @property
    def bootstrap_store_info(self) -> Optional[RendezvousStoreInfo]:
        """Store information that can used by trainer code to bootstrap distributed comms."""
        # 返回当前实例的 _bootstrap_store_info 属性，用于训练代码引导分布式通信时的存储信息
        return self._bootstrap_store_info
# 定义一个抽象基类 RendezvousHandler，用于处理进程间的会合操作

"""Main rendezvous interface.

Note:
    Distributed Torch users normally **do not** need to implement their own
    ``RendezvousHandler``. An implementation based on C10d Store is already
    provided, and is recommended for most users.
"""
class RendezvousHandler(ABC):

    @abstractmethod
    def get_backend(self) -> str:
        """Return the name of the rendezvous backend."""
    
    @property
    def use_agent_store(self) -> bool:
        """Indicates that store reference returned by :py:meth:`next_rendezvous` can be shared with user
        applications and will be available during application lifecyle.

        Rendezous handler impl will share store details as instance of :py:class:`RendezvousStoreInfo`.
        Applications as a convention use `MASTER_ADDR`/`MASTER_PORT` env variables to lookup the store.
        """
        return False

    @abstractmethod
    def next_rendezvous(self) -> RendezvousInfo:
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            Instance of :py:class:`RendezvousInfo`.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    @abstractmethod
    def is_closed(self) -> bool:
        """Check whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        """Mark the rendezvous as closed."""

    @abstractmethod
    def num_nodes_waiting(self) -> int:
        """Return the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        """Return the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """
        # 返回当前会话的运行标识符（run id），用于唯一标识分布式应用的实例
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Close all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """
        # 抽象方法：关闭与 rendezvous 相关的所有资源

        # 注意：具体的关闭逻辑将在派生类中实现
        pass
class RendezvousParameters:
    """Hold the parameters to construct a :py:class:`RendezvousHandler`.
    
    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        **kwargs,
    ):
        # 检查并确保传入的后端名称非空字符串
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        # 确保最小节点数大于等于1
        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        # 确保最大节点数不小于最小节点数
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        # 将参数保存到实例变量中
        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs  # 保存额外的配置参数
        self.local_addr = local_addr  # 保存本地地址信息

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` if ``key`` exists, else ``default``."""
        # 返回指定键的配置值，如果不存在返回默认值
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Return the value for ``key`` as a ``bool``."""
        # 返回指定键的配置值，将其作为布尔值返回
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        # 如果值不是布尔值、整数或字符串形式的布尔值，则抛出错误
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )
    # 返回指定键的值作为整数，如果键不存在则返回默认值（可选）
    def get_as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Return the value for ``key`` as an ``int``."""
        # 获取键 `key` 对应的值，如果不存在则使用默认值 `default`
        value = self.get(key, default)
        # 如果值为 None，则直接返回 None
        if value is None:
            return value
        # 尝试将值转换为整数
        try:
            return int(value)
        # 如果值无法转换为整数，则抛出 ValueError 异常
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e
# 定义了一个类型别名 RendezvousHandlerCreator，表示接受 RendezvousParameters 参数并返回 RendezvousHandler 对象的回调函数
RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]

# 定义了一个类 RendezvousHandlerRegistry，表示 RendezvousHandler 的注册表
class RendezvousHandlerRegistry:
    """Represent a registry of :py:class:`RendezvousHandler` backends."""

    # 类属性 _registry，用于存储注册的后端名称及其对应的创建函数
    _registry: Dict[str, RendezvousHandlerCreator]

    # 初始化方法，创建一个空的注册表字典 _registry
    def __init__(self) -> None:
        self._registry = {}

    # 注册方法，用于向注册表中添加新的后端名称及其对应的创建函数
    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        """Register a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        # 检查后端名称是否为空，若为空则抛出 ValueError 异常
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        # 尝试获取当前注册的后端名称对应的创建函数，如果未注册过则设置为 None
        current_creator: Optional[RendezvousHandlerCreator]
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None

        # 如果当前创建函数不为空且与新的创建函数不同，则抛出 ValueError 异常
        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        # 向注册表中添加或更新后端名称及其对应的创建函数
        self._registry[backend] = creator

    # 创建处理器方法，根据给定的参数创建一个新的 RendezvousHandler 对象
    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Create a new :py:class:`RendezvousHandler`."""
        # 尝试获取给定参数中指定后端名称对应的创建函数，若未注册则抛出 ValueError 异常
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call `{self.register.__name__}`?"
            ) from e

        # 使用获取的创建函数创建一个新的 RendezvousHandler 对象
        handler = creator(params)

        # 进行一些健全性检查，确保创建的处理器后端名称与请求的后端名称一致，否则抛出 RuntimeError 异常
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )

        # 返回创建的处理器对象
        return handler


# 默认的全局注册表实例，用于在启动脚本中实例化会合处理器
rendezvous_handler_registry = RendezvousHandlerRegistry()
```