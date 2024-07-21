# `.\pytorch\torch\distributed\elastic\rendezvous\dynamic_rendezvous.py`

```
# 设置类型提示，允许未类型化的定义
# 版权声明和许可信息，源代码使用 BSD 风格许可证，详见根目录下的 LICENSE 文件
import inspect  # 导入用于检查对象的函数和类的模块
import logging  # 导入日志记录模块
import os  # 导入与操作系统交互的功能模块
import pickle  # 导入用于序列化和反序列化 Python 对象的模块
import socket  # 导入网络通信的套接字模块
import threading  # 导入多线程编程的模块
import time  # 导入处理时间的功能模块
import weakref  # 导入弱引用的模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法的模块
from dataclasses import dataclass  # 导入数据类的装饰器，简化数据结构的定义
from datetime import datetime, timedelta  # 导入处理日期和时间的模块
from enum import Enum  # 导入定义枚举类型的模块
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple  # 导入类型提示相关的模块

import torch.distributed as dist  # 导入 PyTorch 分布式通信模块
from torch.distributed import Store  # 导入 PyTorch 分布式存储模块
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState  # 导入 PyTorch 分布式弹性训练相关的事件记录模块和节点状态

from .api import (  # 导入当前包内的 API 相关模块
    RendezvousClosedError,  # 导入自定义的异常类
    RendezvousError,  # 导入自定义的异常类
    RendezvousGracefulExitError,  # 导入自定义的异常类
    RendezvousHandler,  # 导入自定义的类
    RendezvousInfo,  # 导入自定义的类
    RendezvousParameters,  # 导入自定义的类
    RendezvousStateError,  # 导入自定义的异常类
    RendezvousStoreInfo,  # 导入自定义的类
    RendezvousTimeoutError,  # 导入自定义的异常类
)
from .utils import _delay, _PeriodicTimer  # 从当前包的 utils 模块中导入 _delay 函数和 _PeriodicTimer 类

__all__ = [  # 控制 import * 时导入的符号列表
    "RendezvousBackend",  # 暴露 RendezvousBackend 类
    "RendezvousTimeout",  # 暴露 RendezvousTimeout 类
    "RendezvousSettings",  # 暴露 RendezvousSettings 类
    "DynamicRendezvousHandler",  # 暴露 DynamicRendezvousHandler 类
    "create_handler",  # 暴露 create_handler 函数
]

logger = logging.getLogger(__name__)  # 创建一个日志记录器对象，用于记录当前模块的日志信息


def get_method_name(depth=2):
    """返回调用栈中指定深度处的函数名或默认字符串 'no_method_name'。"""
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return "no_method_name"


Token = Any
"""表示由 rendezvous 后端使用的不透明的围栏令牌。"""


class RendezvousBackend(ABC):
    """表示保存 rendezvous 状态的后端。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """获取后端的名称。"""

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """获取 rendezvous 状态。

        Returns:
            以编码形式返回 rendezvous 状态及其围栏令牌的元组，如果后端中没有状态则返回 None。

        Raises:
            RendezvousConnectionError:
                连接到后端失败。
            RendezvousStateError:
                rendezvous 状态损坏。
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """
        设置会合状态。

        新的会合状态会根据以下条件进行设置：

          - 如果指定的 ``token`` 与存储在后端的围栏令牌匹配，状态将被更新。新的状态
            将与其围栏令牌一起返回给调用者。
          - 如果指定的 ``token`` 与存储在后端的围栏令牌不匹配，状态不会被更新；相反，
            将会返回现有状态及其围栏令牌给调用者。
          - 如果指定的 ``token`` 是 ``None``，只有在后端不存在现有状态时才会设置新的
            状态。将返回新的状态或现有状态及其围栏令牌给调用者。

        Args:
            state:
                编码后的会合状态。
            token:
                可选的围栏令牌，是之前调用 :py:meth:`get_state` 或 ``set_state()`` 获取的。

        Returns:
            包含序列化后的会合状态、其围栏令牌和一个布尔值（指示设置尝试是否成功）的元组。

        Raises:
            RendezvousConnectionError:
                与后端的连接失败。
            RendezvousStateError:
                会合状态损坏。
        """
class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        keep_alive:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """

    _ZERO = timedelta(0)  # Zero timedelta constant for comparison

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),      # Default timeout for join operation
        "last_call": timedelta(seconds=30),  # Default timeout for last call operation
        "close": timedelta(seconds=30),      # Default timeout for close operation
        "heartbeat": timedelta(seconds=5),   # Default timeout for heartbeat operation
    }

    _join: timedelta      # Timeout for join operation
    _last_call: timedelta  # Timeout for last call operation
    _close: timedelta     # Timeout for close operation
    _heartbeat: timedelta # Timeout for heartbeat operation

    def __init__(
        self,
        join: Optional[timedelta] = None,
        last_call: Optional[timedelta] = None,
        close: Optional[timedelta] = None,
        heartbeat: Optional[timedelta] = None,
    ) -> None:
        """Initialize RendezvousTimeout instance with specified timeouts."""
        self._set_timeouts(
            join=join, last_call=last_call, close=close, heartbeat=heartbeat
        )

    @property
    def join(self) -> timedelta:
        """Get the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Get the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Get the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        """Get the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        """Set the timeouts based on provided values or defaults."""
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)


@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """

    run_id: str          # Identifier for the rendezvous run
    min_nodes: int       # Minimum number of nodes required for the rendezvous
    max_nodes: int       # Maximum number of nodes allowed for the rendezvous
    timeout: RendezvousTimeout  # Timeout configuration for the rendezvous
    keep_alive_interval: Optional[timedelta] = None  # Interval for node keep-alive heartbeats
    keep_alive_max_attempt: Optional[int] = None    # Maximum failed attempts before node is considered dead
    # 定义一个名为 timeout 的变量，类型为 RendezvousTimeout，用于设置超时参数
    timeout: RendezvousTimeout
    # 定义一个名为 keep_alive_interval 的变量，类型为 timedelta，用于设置保持连接的时间间隔
    keep_alive_interval: timedelta
    # 定义一个名为 keep_alive_max_attempt 的变量，类型为 int，用于设置保持连接的最大尝试次数
    keep_alive_max_attempt: int
# 定义一个数据类 `_NodeDesc`，描述在会合中的一个节点

@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str  # 节点的完全限定域名或用户指定的本地节点地址
    pid: int  # 会合处理程序所在进程的ID
    local_id: int  # 进程范围内唯一的ID

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0  # 每次调用 generate() 时自增的整数

    def generate(self, local_addr: Optional[str] = None) -> _NodeDesc:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id  # 获取当前本地ID
            self._local_id += 1  # 原子性地递增本地ID

        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)


class _RendezvousState:
    """Hold the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        redundancy_list:
            A set of nodes that are redundant in the current round and can join
            the next rendezvous without triggering re-rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """

    round: int  # 当前会合的轮次
    complete: bool  # 表示当前会合轮次是否完成的布尔值
    deadline: Optional[datetime]  # 如果仍在等待节点加入，则当前会合轮次的完成截止时间
    closed: bool  # 表示会合是否已关闭的布尔值
    participants: Dict[_NodeDesc, int]  # 参与者及其对应排名的字典
    wait_list: Set[_NodeDesc]  # 等待参与下一轮会合的节点集合
    redundancy_list: Set[_NodeDesc]  # 当前轮次中冗余节点的集合，可以在下一次会合中加入而不会触发重新会合
    last_heartbeats: Dict[_NodeDesc, datetime]  # 包含每个节点最后心跳时间的字典


def _remove_participant_epilogue(
    state: _RendezvousState, settings: RendezvousSettings
) -> None:
    # 如果状态为 complete（完成）：
    if state.complete:
        # 如果没有剩余参与者，进入下一轮。
        if not state.participants:
            # 提示消息，指示没有剩余参与者，将 rendezvous 标记为不完整。
            msg = "No participants left in the rendezvous, marking rendezvous as incomplete"
            # 记录调试级别的消息到日志中。
            logger.debug(msg)
            # 将状态标记为不完整。
            state.complete = False

            # 增加轮次计数器。
            state.round += 1
    else:
        # 如果状态不为 complete（未完成）：
        # 如果参与者数量少于设定的最小节点数。
        if len(state.participants) < settings.min_nodes:
            # 构造消息，指示参与者数量少于最小节点数，清空状态中的截止时间。
            msg = (
                f"Number of participants {len(state.participants)}) less than"
                f"min_nodes {settings.min_nodes}, clearning deadline in state"
            )
            # 记录调试级别的消息到日志中。
            logger.debug(msg)
            # 将状态中的截止时间设为 None。
            state.deadline = None
    class _RendezvousStateHolder(ABC):
        """Hold the shared rendezvous state synced with other nodes."""
    
        @property
        @abstractmethod
        def state(self) -> _RendezvousState:
            """Get the local state."""
    
        @abstractmethod
        def sync(self) -> Optional[bool]:
            """Read or writes the latest state.
    
            Returns:
                A boolean value indicating whether the local state, in case marked
                as dirty, was successfully synced with other nodes.
            """
    
        @abstractmethod
        def mark_dirty(self) -> None:
            """Mark the local state as dirty."""
    
    
    class _BackendRendezvousStateHolder(_RendezvousStateHolder):
        """Hold the rendezvous state synced with other nodes via a backend.
    
        Args:
            backend:
                The rendezvous backend to use.
            settings:
                The rendezvous settings.
            cache_duration:
                The amount of time, in seconds, to cache the last rendezvous state
                before requesting it from the backend again.
        """
    
        _backend: RendezvousBackend  # 用于存储后端实例的属性
        _state: _RendezvousState  # 用于存储当前状态的属性
        _settings: RendezvousSettings  # 用于存储设置的属性
        _cache_duration: int  # 缓存持续时间的属性
        _token: Token  # 用于存储令牌的属性
        _dirty: bool  # 表示状态是否脏的布尔属性
        _last_sync_time: float  # 上次同步时间的属性
        _dead_nodes: List[_NodeDesc]  # 存储已离线节点的列表属性
    
        def __init__(
            self,
            backend: RendezvousBackend,
            settings: RendezvousSettings,
            cache_duration: int = 1,
        ) -> None:
            """Initialize the _BackendRendezvousStateHolder instance.
    
            Args:
                backend: The backend instance to use for rendezvous.
                settings: The settings object containing rendezvous configurations.
                cache_duration: Duration to cache the state before refreshing from backend.
            """
            self._backend = backend
            self._state = _RendezvousState()  # 初始化空的状态对象
            self._settings = settings  # 初始化设置对象
            self._cache_duration = cache_duration  # 初始化缓存持续时间
            self._token = None  # 初始化令牌为 None
            self._dirty = False  # 初始化状态为非脏
            self._last_sync_time = -1  # 初始化上次同步时间为 -1
            self._dead_nodes = []  # 初始化离线节点列表为空
    
        def _record(self, message: str, node_state: NodeState = NodeState.RUNNING):
            """Internal method to record rendezvous events.
    
            Args:
                message: The message to record.
                node_state: The state of the node when recording the event.
            """
            construct_and_record_rdzv_event(
                name=f"{self.__class__.__name__}.{get_method_name()}",
                run_id=self._settings.run_id,
                message=message,
                node_state=node_state,
            )
    
        @property
        def state(self) -> _RendezvousState:
            """See base class."""
            return self._state
    # 同步方法，返回一个布尔值或者空值
    def sync(self) -> Optional[bool]:
        """See base class."""
        # 初始化状态位为 None
        state_bits: Optional[bytes] = None

        # 初始化 token 为 None
        token = None

        # 初始化 has_set 为 None
        has_set: Optional[bool]

        # 如果状态为脏
        if self._dirty:
            # 设置 has_set 为 False
            has_set = False

            # 序列化状态数据
            state_bits = pickle.dumps(self._state)

            # 调用后端方法设置状态
            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                # 解析设置响应
                state_bits, token, has_set = set_response
        else:
            # has_set 为 None
            has_set = None

            # 如果缓存时间大于 0
            if self._cache_duration > 0:
                # 避免过度加载后端，尝试提供缓存状态
                if self._last_sync_time >= max(
                    time.monotonic() - self._cache_duration, 0
                ):
                    return None

            # 调用后端方法获取状态
            get_response = self._backend.get_state()
            if get_response is not None:
                # 解析获取响应
                state_bits, token = get_response

        # 如果状态数据不为 None
        if state_bits is not None:
            try:
                # 反序列化状态数据
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                # 抛出异常
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            # 初始化状态数据
            self._state = _RendezvousState()

        # 如果 has_set 为 True 且存在死节点且日志级别为 DEBUG
        if has_set and self._dead_nodes and logger.isEnabledFor(logging.DEBUG):
            # 构建节点列表字符串
            node_list = ", ".join(f"'{dead_node}'" for dead_node in self._dead_nodes)

            # 构建消息
            msg = (
                f"As part of the sync operation the node(s) {node_list} have been removed from the "
                f"rendezvous '{self._settings.run_id}' since they had no heartbeat."
            )
            # 记录消息
            self._record(message=msg)
            logger.debug(msg)

        # 更新 token
        self._token = token

        # 标记状态为非脏
        self._dirty = False

        # 更新最后同步时间
        self._last_sync_time = time.monotonic()

        # 清理数据
        self._sanitize()

        # 返回 has_set
        return has_set
    # 定义一个方法，用于清理和维护对象的内部状态
    def _sanitize(self) -> None:
        # 获取对象的当前状态
        state = self._state

        # 计算超时时间，根据保持活动间隔和最大尝试次数来确定
        expire_time = datetime.utcnow() - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # 过滤出超时的节点并存储在 _dead_nodes 列表中
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]

        # 标记是否有参与者被移除
        participant_removed = False

        # 遍历已确定的超时节点列表
        for dead_node in self._dead_nodes:
            # 记录日志，指出检测到的死亡节点并从会合中移除
            msg = f"Detected dead node '{dead_node}', removing it from the rendezvous"
            logger.debug(msg)

            # 从状态中删除死亡节点的最后心跳记录
            del state.last_heartbeats[dead_node]

            try:
                # 尝试从状态中删除死亡节点的参与者信息
                del state.participants[dead_node]

                # 设置标志，表示至少有一个参与者被移除
                participant_removed = True
            except KeyError:
                pass

            try:
                # 尝试从等待列表中移除死亡节点
                state.wait_list.remove(dead_node)
            except KeyError:
                pass

            try:
                # 尝试从冗余列表中移除死亡节点
                state.redundancy_list.remove(dead_node)
            except KeyError:
                pass

        # 如果至少有一个参与者被移除，则执行 _remove_participant_epilogue() 共同的结尾步骤
        if participant_removed:
            _remove_participant_epilogue(state, self._settings)

    # 标记当前对象的状态为脏，需要在下次同步调用中将变更写回后端
    def mark_dirty(self) -> None:
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """
        self._dirty = True
class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""

    KEEP_ALIVE = 1  # 表示保持活动状态的动作
    ADD_TO_PARTICIPANTS = 2  # 表示将节点添加到参与者列表的动作
    ADD_TO_WAIT_LIST = 3  # 表示将节点添加到等待列表的动作
    ADD_TO_REDUNDANCY_LIST = 4  # 表示将节点添加到冗余列表的动作
    REMOVE_FROM_PARTICIPANTS = 5  # 表示从参与者列表中移除节点的动作
    REMOVE_FROM_WAIT_LIST = 6  # 表示从等待列表中移除节点的动作
    REMOVE_FROM_REDUNDANCY_LIST = 7  # 表示从冗余列表中移除节点的动作
    MARK_RENDEZVOUS_COMPLETE = 8  # 表示标记会合完成的动作
    MARK_RENDEZVOUS_CLOSED = 9  # 表示标记会合关闭的动作
    SYNC = 10  # 表示同步的动作
    ERROR_CLOSED = 11  # 表示标记错误：关闭的动作
    ERROR_TIMEOUT = 12  # 表示标记错误：超时的动作
    FINISH = 13  # 表示完成的动作


class _RendezvousContext:
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """

    node: _NodeDesc  # 与当前会合处理程序实例相关联的节点描述符
    state: _RendezvousState  # 会合的当前状态
    settings: RendezvousSettings  # 会合的设置

    def __init__(
        self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings
    ) -> None:
        self.node = node
        self.state = state
        self.settings = settings


class _RendezvousOpExecutor(ABC):
    """Execute rendezvous operations."""

    @abstractmethod
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Optional[Callable[[timedelta], float]] = None,
    ) -> None:
        """Execute a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
            update_deadline:
                Function to generate a new operation deadline if the current
                node may participate in the next rendezvous.
        """


class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """

    _node: _NodeDesc  # 当前会合处理程序实例相关联的节点描述符
    _state: _RendezvousState  # 会合的当前状态
    _state_holder: _RendezvousStateHolder  # 用于与其他节点同步会合状态的状态持有者
    _settings: RendezvousSettings  # 会合的设置

    def __init__(
        self,
        node: _NodeDesc,
        state_holder: _RendezvousStateHolder,
        settings: RendezvousSettings,
    ) -> None:
        self._node = node
        self._state_holder = state_holder
        self._settings = settings
    # 记录节点状态，生成并记录相关的 rendezvous 事件
    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._node.addr,
            pid=self._node.pid,
            local_id=self._node.local_id,
        )

    # 运行 rendezvous 算法，处理节点状态和超时逻辑
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Optional[Callable[[timedelta], float]] = None,
    ) -> None:
        # 略，未提供完整代码，无法完整注释

    # 更新节点的心跳时间，并记录相应事件
    def _keep_alive(self) -> None:
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.utcnow()

    # 将节点添加到当前 round 的参与者列表中
    def _add_to_participants(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        state = self._state

        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass

        # 参与者的排名将在 rendezvous 完成后设置
        state.participants[self._node] = 0

        self._keep_alive()

        if len(state.participants) == self._settings.min_nodes:
            state.deadline = datetime.utcnow() + self._settings.timeout.last_call

        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    # 将节点添加到等待列表中
    def _add_to_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        if self._node in self._state.redundancy_list:
            self._state.redundancy_list.remove(self._node)
        self._state.wait_list.add(self._node)

        self._keep_alive()

    # 将节点添加到冗余列表中
    def _add_to_redundancy_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the redundancy list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.redundancy_list.add(self._node)

        self._keep_alive()
    # 从参与者列表中移除当前节点的操作方法
    def _remove_from_participants(self) -> None:
        # 构造记录消息，指明节点从当前会合轮次的参与者中移除，并进行同步操作
        msg = (
            f"The node '{self._node}' removed itself from the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)  # 记录消息
        logger.debug(msg)  # 使用调试级别记录日志消息

        state = self._state  # 获取当前状态对象的引用

        del state.participants[self._node]  # 从参与者字典中移除当前节点

        del state.last_heartbeats[self._node]  # 从上次心跳时间字典中移除当前节点

        # 调用 _remove_participant_epilogue 方法，处理参与者列表操作的常见尾声
        _remove_participant_epilogue(state, self._settings)

    # 从等待列表中移除当前节点的操作方法
    def _remove_from_wait_list(self) -> None:
        # 构造记录消息，指明节点从下一个会合轮次的等待列表中移除，并进行同步操作
        msg = (
            f"The node '{self._node}' removed itself from the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)  # 记录消息
        logger.debug(msg)  # 使用调试级别记录日志消息

        self._state.wait_list.remove(self._node)  # 从等待列表中移除当前节点

        del self._state.last_heartbeats[self._node]  # 从上次心跳时间字典中移除当前节点

    # 从冗余列表中移除当前节点的操作方法
    def _remove_from_redundancy_list(self) -> None:
        # 构造记录消息，指明节点从下一个会合轮次的冗余列表中移除，并进行同步操作
        msg = (
            f"The node '{self._node}' removed itself from the redunant list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)  # 记录消息
        logger.debug(msg)  # 使用调试级别记录日志消息

        self._state.redundancy_list.remove(self._node)  # 从冗余列表中移除当前节点

        del self._state.last_heartbeats[self._node]  # 从上次心跳时间字典中移除当前节点

    # 标记会合完成的操作方法
    def _mark_rendezvous_complete(self) -> None:
        # 构造记录消息，指明节点标记当前会合轮次完成，并进行同步操作
        msg = (
            f"The node '{self._node}' marked round {self._state.round} of the rendezvous "
            f"'{self._settings.run_id}' as complete. Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)  # 记录消息，标记节点状态为成功
        logger.debug(msg)  # 使用调试级别记录日志消息

        state = self._state  # 获取当前状态对象的引用

        state.complete = True  # 标记会合完成状态为 True
        state.deadline = None  # 清空会合的截止时间

        # 为参与者分配排名
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank  # 使用排序后的节点列表进行排名赋值

    # 标记会合关闭的操作方法
    def _mark_rendezvous_closed(self) -> None:
        # 构造记录消息，指明节点标记当前会合关闭，并进行同步操作
        msg = (
            f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. "
            "Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)  # 记录消息，标记节点状态为成功
        logger.debug(msg)  # 使用调试级别记录日志消息

        self._state.closed = True  # 标记会合关闭状态为 True
def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    """Determines whether a keep-alive heartbeat should be sent."""
    try:
        # 获取上次心跳时间
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        # 如果找不到对应节点的心跳时间，返回 False
        return False

    # 判断上次心跳时间是否超过了保持活动状态的时间间隔
    return last_heartbeat <= datetime.utcnow() - ctx.settings.keep_alive_interval


class _RendezvousExitOp:
    """Represents a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # 如果节点在参与者列表中
        if ctx.node in ctx.state.participants:
            # 检查当前时间是否超过了截止时间
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            # 返回从参与者列表中移除该节点的操作
            return _Action.REMOVE_FROM_PARTICIPANTS
        # 如果节点不在参与者列表中，则返回完成操作
        return _Action.FINISH


class _RendezvousJoinOp:
    """Represents a rendezvous join operation."""


class _RendezvousCloseOp:
    """Represents a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # 如果会话已关闭，则返回完成操作
        if ctx.state.closed:
            return _Action.FINISH
        # 检查当前时间是否超过了截止时间
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        # 返回标记会话已关闭的操作
        return _Action.MARK_RENDEZVOUS_CLOSED


class _RendezvousKeepAliveOp:
    """Represents a rendezvous keep-alive update operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # 如果应该发送保持活动状态的心跳
        if _should_keep_alive(ctx):
            # 检查当前时间是否超过了截止时间
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            # 返回发送保持活动状态的心跳的操作
            return _Action.KEEP_ALIVE
        # 如果不需要发送心跳，则返回完成操作
        return _Action.FINISH


class DynamicRendezvousHandler(RendezvousHandler):
    """Represents a handler that sets up a rendezvous among a set of nodes."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: Optional[_PeriodicTimer]

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        timeout: Optional[RendezvousTimeout] = None,
    ):
        """
        创建一个新的 :py:class:`DynamicRendezvousHandler`。

        Args:
            run_id:
                会议点的运行 ID。
            store:
                作为会议点一部分返回的 C10d 存储。
            backend:
                用于保存会议点状态的后端。
            min_nodes:
                参加会议点的最小节点数。
            max_nodes:
                参加会议点的最大节点数。
            local_addr:
                本地节点地址。
            timeout:
                会议点的超时配置。
        """
        # 为每个处理程序实例关联一个唯一的节点描述符。
        node = cls._node_desc_generator.generate(local_addr)

        # 创建会议点设置对象
        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=5),
            keep_alive_max_attempt=3,
        )

        # 创建后端会议点状态持有者对象
        state_holder = _BackendRendezvousStateHolder(backend, settings)

        # 返回创建的 DynamicRendezvousHandler 实例
        return cls(node, settings, backend.name, store, state_holder)

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        state_holder: _RendezvousStateHolder,
    ) -> None:
        # 检查运行 ID 是否为空
        if not settings.run_id:
            raise ValueError("The run id must be a non-empty string.")

        # 检查最小节点数是否小于 1
        if settings.min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be greater than zero."
            )

        # 检查最大节点数是否小于最小节点数
        if settings.max_nodes < settings.min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({settings.max_nodes}) must be greater than or equal "
                f"to the minimum number of nodes ({settings.min_nodes})."
            )

        # 初始化实例变量
        self._this_node = node
        self._settings = settings
        self._backend_name = backend_name
        self._store = store
        self._state_holder = state_holder

        # 创建操作执行器对象
        self._op_executor = _DistributedRendezvousOpExecutor(
            self._this_node, self._state_holder, self._settings
        )

        # 初始化心跳锁
        self._heartbeat_lock = threading.Lock()

        self._keep_alive_timer = None

        # 缓存共享存储服务器引用
        self._shared_tcp_store_server: Optional[dist.Store] = None

        self._bootstrap_store_info: Optional[RendezvousStoreInfo] = None

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: Optional[int] = None,
    def construct_and_record_rdzv_event(self, message: str, node_state: NodeState, rank: int) -> None:
        # 构造并记录分布式同步事件
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    def _create_tcp_store_server(self, bootstrap_store_info) -> dist.TCPStore:
        # 创建 TCP 存储服务器
        return dist.TCPStore(
            bootstrap_store_info.master_addr,
            bootstrap_store_info.master_port,
            is_master=True,
            multi_tenant=True,
            use_libuv=True,
        )

    @property
    def settings(self) -> RendezvousSettings:
        """Get the settings of the rendezvous."""
        # 获取分布式同步的设置
        return self._settings

    def get_backend(self) -> str:
        """See base class."""
        # 获取后端名称
        return self._backend_name

    @property
    def use_agent_store(self) -> bool:
        """See base class."""
        # 检查是否使用代理存储
        return os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") != "1"

    def is_closed(self) -> bool:
        """See base class."""
        # 检查是否已关闭
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return self._state_holder.state.closed

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def set_closed(self) -> None:
        """See base class."""
        # 设置为已关闭状态
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def num_nodes_waiting(self) -> int:
        """See base class."""
        # 获取等待节点数量
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return len(self._state_holder.state.wait_list)

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def get_run_id(self) -> str:
        """See base class."""
        # 获取运行 ID
        return self._settings.run_id
    def shutdown(self) -> bool:
        """关闭当前节点的操作。

        停止心跳信号以确保安全关闭。
        """
        self._stop_heartbeats()  # 停止发送心跳信号

        try:
            self._close()  # 尝试执行关闭操作

            return True  # 如果成功关闭，则返回True
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to shutdown the rendezvous "
                f"'{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            logger.warning(msg)

            return False  # 如果关闭过程中发生错误，则返回False
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise  # 如果出现未处理的异常，则向上层抛出异常

    def _close(self) -> None:
        """关闭当前节点与Rendezvous的连接。

        执行关闭操作，并记录关闭信息。
        """
        op = _RendezvousCloseOp()

        deadline = self._get_deadline(self._settings.timeout.close)

        self._op_executor.run(op, deadline)  # 运行关闭操作

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        """静态方法：弱引用保持节点的心跳信号。

        弱引用节点的方法，以避免循环引用。
        """
        self = weak_self()
        if self is not None:
            self._keep_alive()  # 如果节点存在，则执行保持心跳信号的操作

    def _keep_alive(self) -> None:
        """保持当前节点的心跳信号。

        获取心跳信号的锁，执行心跳信号发送操作。
        """
        self._heartbeat_lock.acquire()  # 获取心跳信号的锁

        op = _RendezvousKeepAliveOp()

        deadline = self._get_deadline(self._settings.timeout.heartbeat)

        try:
            self._op_executor.run(op, deadline)  # 运行保持心跳信号的操作

            msg = (
                f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous "
                f"'{self._settings.run_id}'."
            )
            self._record(message=msg)  # 记录心跳信号发送的信息
            logger.debug(msg)
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the "
                f"rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            logger.warning(msg)
        finally:
            self._heartbeat_lock.release()  # 释放心跳信号的锁

    def _start_heartbeats(self) -> None:
        """开始定期发送心跳信号。

        使用定时器启动心跳信号发送。
        """
        self._keep_alive_timer = _PeriodicTimer(
            self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self)
        )

        self._keep_alive_timer.set_name(
            f"RendezvousKeepAliveTimer_{self._this_node.local_id}"
        )

        self._keep_alive_timer.start()  # 启动心跳信号定时器

    def _stop_heartbeats(self) -> None:
        """停止定期发送心跳信号。

        如果心跳信号定时器存在，则取消定时器。
        """
        if self._keep_alive_timer is None:
            return

        self._keep_alive_timer.cancel()  # 取消心跳信号定时器

    def _get_world(self) -> Tuple[int, int]:
        """获取当前节点所在的参与者信息。

        从状态保持器获取当前节点和参与者的数量信息。
        """
        state = self._state_holder.state

        return state.participants[self._this_node], len(state.participants)
    # 定义一个方法 `_wrap_store`，用于将给定的 `store` 对象包装在以特定前缀命名的分布式存储中
    def _wrap_store(self, store: Store) -> Store:
        # 构建用于存储键前缀的字符串，包括运行 ID 和当前轮次号
        key_prefix = (
            f"torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}"
        )
        # 使用给定的 `key_prefix` 和传入的 `store` 创建并返回一个 PrefixStore 对象
        return dist.PrefixStore(key_prefix, store)

    # 定义一个方法 `_get_store`，用于获取经过 `_wrap_store` 方法包装的 `_store` 对象
    def _get_store(self) -> Store:
        # 调用 `_wrap_store` 方法，传入当前对象的 `_store` 成员变量，返回处理后的 Store 对象
        return self._wrap_store(self._store)

    # 定义一个方法 `_get_deadline`，用于计算超时时间后的时间戳
    def _get_deadline(self, timeout: timedelta) -> float:
        # 计算当前时间的单调时钟时间加上传入的 `timeout` 参数的总秒数，返回浮点数时间戳
        return time.monotonic() + timeout.total_seconds()
# 定义一个函数用于从参数中获取超时时间并转换为 timedelta 对象
def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    # 从参数中获取以 key + "_timeout" 命名的超时时间，并尝试转换为整数
    timeout = params.get_as_int(key + "_timeout")
    # 如果获取的超时时间为 None，则返回 None
    if timeout is None:
        return None
    # 将获取的超时时间转换为秒数，创建一个 timedelta 对象并返回
    return timedelta(seconds=timeout)


# 定义一个函数用于创建动态会合处理程序
def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    """Create a new :py:class:`DynamicRendezvousHandler` from the specified parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.
    """
    try:
        # 从参数中获取不同类型的超时时间，并创建一个 RendezvousTimeout 对象
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),    # 获取 join_timeout 的超时时间
            _get_timeout(params, "last_call"),   # 获取 last_call_timeout 的超时时间
            _get_timeout(params, "close"),   # 获取 close_timeout 的超时时间
        )

        # 使用指定的参数创建一个 DynamicRendezvousHandler 对象并返回
        return DynamicRendezvousHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
        )
    except Exception as e:
        # 如果发生异常，记录事件并重新抛出异常
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
```