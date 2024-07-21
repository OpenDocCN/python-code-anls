# `.\pytorch\torch\distributed\elastic\agent\server\api.py`

```py
# 忽略类型检查错误（这里是针对mypy类型检查工具的声明）
# 版权声明，版权归Facebook及其关联公司所有
# 保留所有权利
# 该源代码采用BSD风格许可证，许可证文件可以在源树根目录下的LICENSE文件中找到

# 导入标准库和第三方库
import abc  # 提供抽象基类（ABC）的支持
import json  # 提供JSON编解码功能
import os  # 提供与操作系统交互的功能
import signal  # 提供与信号处理相关的功能
import socket  # 提供网络通信相关的功能
import time  # 提供时间相关的功能
import traceback  # 提供异常跟踪相关的功能
import warnings  # 提供警告处理的功能
from collections import defaultdict  # 提供默认字典功能
from contextlib import contextmanager  # 提供上下文管理相关的功能
from dataclasses import dataclass, field  # 提供数据类的支持
from enum import Enum  # 提供枚举类型的支持
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 提供类型提示的支持

# 导入torch相关的分布式elastic模块
import torch.distributed.elastic.rendezvous as rdzv  # 提供分布式会议协调的功能
import torch.distributed.elastic.utils.store as store_util  # 提供分布式elastic相关的存储工具
from torch.distributed.elastic.events import Event, EventSource, record  # 提供分布式elastic事件相关的功能
from torch.distributed.elastic.metrics import prof, put_metric  # 提供分布式elastic度量相关的功能
from torch.distributed.elastic.multiprocessing import ProcessFailure, SignalException  # 提供分布式elastic多进程相关的功能
from torch.distributed.elastic.rendezvous import RendezvousGracefulExitError  # 提供分布式elastic会议优雅退出异常的支持
from torch.distributed.elastic.utils.logging import get_logger  # 提供分布式elastic日志相关的功能

# 将以下标识添加到模块的公共接口中
__all__ = [
    "WorkerSpec",
    "Worker",
    "WorkerState",
    "WorkerGroup",
    "RunResult",
    "ElasticAgent",
    "SimpleElasticAgent",
]

# 用于终端状态同步的标识符
_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"

# 默认的角色名称
DEFAULT_ROLE = "default"

# 获取当前模块的日志记录器
logger = get_logger(__name__)

@dataclass
class WorkerSpec:
    """关于特定类型工作进程的蓝图信息。

    对于给定角色，应只存在一个工作进程规范。
    预期在所有节点（机器）上，特定规范的工作进程是同质的，
    即每个节点运行相同数量的特定规范工作进程。

    Args:
        role: 工作进程的用户定义角色
        local_world_size: 要运行的本地工作进程数
        fn:（已弃用，请改用entrypoint）
        entrypoint: 工作进程函数或命令
        args: 传递给“entrypoint”的参数
        rdzv_handler: 处理此组工作进程的rdzv
        max_restarts: 工作进程的最大重试次数
        monitor_interval: 每隔“n”秒监视工作进程状态
        master_port: 在rank 0上运行c10d存储的固定端口
                    如果未指定，则将选择一个空闲端口
        master_addr: 在rank 0上运行c10d存储的固定master_addr
                    如果未指定，则将选择代理rank 0上的主机名
        redirects: 将标准流重定向到文件，
                   通过传递映射选择性地重定向特定本地rank的流
        tee: 将指定的标准流复制到控制台+文件，
             通过传递映射选择性地在特定本地rank上复制，
             优先于“redirects”设置。

    """
    role: str  # 工作进程的角色名称
    local_world_size: int  # 要运行的本地工作进程数
    rdzv_handler: rdzv.RendezvousHandler  # 处理此组工作进程的rdzv对象
    fn: Optional[Callable] = None  # （已弃用）工作进程函数
    entrypoint: Union[Callable, str, None] = None  # 工作进程函数或命令
    args: Tuple = ()  # 传递给entrypoint的参数
    max_restarts: int = 3  # 工作进程的最大重试次数
    # 定义监控间隔，默认为0.1秒
    monitor_interval: float = 0.1
    # 主节点端口号，可选参数，默认为None
    master_port: Optional[int] = None
    # 主节点地址，可选参数，默认为None
    master_addr: Optional[str] = None
    # 本地地址，可选参数，默认为None
    local_addr: Optional[str] = None

    # 初始化函数后处理方法
    def __post_init__(self):
        # 断言本地世界大小大于0
        assert self.local_world_size > 0
        # 断言监控间隔大于0
        assert self.monitor_interval > 0

        # 如果存在fn属性，则发出警告并设置entrypoint为fn
        if self.fn:
            warnings.warn(
                "WorkerSpec.fn will be deprecated,"
                " please use WorkerSpec.entrypoint instead",
                category=DeprecationWarning,
            )
            self.entrypoint = self.fn
        # 断言entrypoint不为空
        assert self.entrypoint

    # 获取入口点名称的方法
    def get_entrypoint_name(self):
        """Get the entry point name.

        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """
        # 如果entrypoint是字符串，则返回其基本名称
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        else:
            # 否则，断言entrypoint不为None，并返回其限定名称
            assert self.entrypoint is not None
            return self.entrypoint.__qualname__
# 定义 Worker 类，表示一个工作节点实例

class Worker:
    """A worker instance.

    Contrast this with ``WorkerSpec`` that represents the specifications of a
    worker. A ``Worker`` is created from a ``WorkerSpec``. A ``Worker`` is to
    a ``WorkerSpec`` as an object is to a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Args:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """

    # 定义实例变量列表，用 __slots__ 提高内存利用效率
    __slots__ = [
        "id",
        "local_rank",
        "global_rank",
        "role_rank",
        "world_size",
        "role_world_size",
    ]

    # 初始化方法，设置工作节点的各个属性
    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        # 唯一标识符，由代理程序解释
        self.id: Any = None

        # 当前工作节点在其角色中的本地排名
        self.local_rank: int = local_rank

        # 当前工作节点在所有角色中的全局排名，可能会在重汇合时改变
        self.global_rank: int = global_rank

        # 当前工作节点在所有具有相同角色的工作节点中的排名，可能会在重汇合时改变
        self.role_rank: int = role_rank

        # 全局工作节点的总数，由于弹性，这个值可能会在重汇合时改变
        self.world_size: int = world_size

        # 具有相同角色的工作节点的总数，由于弹性，这个值可能会在重汇合时改变
        self.role_world_size: int = role_world_size

    # 返回工作节点对象的字符串表示形式
    def __str__(self):
        return (
            f"local_rank={self.local_rank},global_rank={self.global_rank}"
            f",role_rank={self.role_rank},world_size={self.world_size}"
            f",role_world_size={self.role_world_size}"
        )

    # 返回工作节点对象的字符串表示形式（用于调试）
    def __repr__(self):
        return str(self)


# 表示 WorkerState 的枚举类，描述 WorkerGroup 的状态
class WorkerState(str, Enum):
    """A state of the ``WorkerGroup``.

    Workers in a worker group change state as a unit. If a single worker
    """
        in a worker group fails the entire set is considered failed::
    
          UNKNOWN - agent lost track of worker group state, unrecoverable
          INIT - worker group object created not yet started
          HEALTHY - workers running and healthy
          UNHEALTHY - workers running and unhealthy
          STOPPED - workers stopped (interrupted) by the agent
          SUCCEEDED - workers finished running (exit 0)
          FAILED - workers failed to successfully finish (exit !0)
    
    
        A worker group starts from an initial ``INIT`` state,
        then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
        and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.
    
        Worker groups can be interrupted and temporarily put into ``STOPPED`` state
        by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
        in the near future by the agent. Some examples of workers being put into
        ``STOPPED`` state are:
    
        1. Worker group failure|unhealthy observed
        2. Membership change detected
    
        When actions (start, stop, rdzv, retry, etc) on worker group fails
        and results in the action being partially applied to the worker group
        the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
        exceptions during state change events on the agent. The agent is not
        expected to recover worker groups in ``UNKNOWN`` state and is better off
        self terminating and allowing the job manager to retry the node.
        """
    
        # 定义不同的工作状态常量
        UNKNOWN = "UNKNOWN"
        INIT = "INIT"
        HEALTHY = "HEALTHY"
        UNHEALTHY = "UNHEALTHY"
        STOPPED = "STOPPED"
        SUCCEEDED = "SUCCEEDED"
        FAILED = "FAILED"
    
        @staticmethod
        def is_running(state: "WorkerState") -> bool:
            """Return the state of the Worker.
    
            Returns:
                 True if the worker state represents workers still running
                 (e.g. that the process exists but not necessarily healthy).
            """
            # 检查给定的状态是否表示工作仍在运行
            return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}
class WorkerGroup:
    """A set of ``Worker`` instances.

    The class defines a set of ``Worker`` instances for the given ``WorkerSpec`` managed by ``ElasticAgent``. Whether the worker
    group contains cross instance workers or not depends on the implementation of the agent.
    """

    __slots__ = [
        "spec",
        "workers",
        "store",
        "group_rank",
        "group_world_size",
        "state",
        "master_addr",
        "master_port",
    ]

    def __init__(self, spec: WorkerSpec):
        # 初始化方法，接受一个 WorkerSpec 对象作为参数
        self.spec = spec
        # 创建一个包含多个 Worker 实例的列表，每个 Worker 的 local_rank 从 0 到 spec.local_world_size - 1
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None  # 用于存储对象，暂未指定
        self.group_rank = None  # 组的排名，暂未指定
        self.group_world_size = None  # 组内的世界大小，暂未指定
        self.master_addr = None  # 主节点地址，暂未指定
        self.master_port = None  # 主节点端口，暂未指定

        self.state = WorkerState.INIT  # 设置工作组的状态为初始化状态


class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role  # 工作角色
        self.rank = rank  # 代理的排名
        self.local_world_size = local_world_size  # 本地运行的工作人数

    def serialize(self) -> bytes:
        # 序列化对象为字节流
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        # 从字节流中反序列化对象
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        # 比较两个 _RoleInstanceInfo 对象的角色和排名
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        # 查找角色信息列表中指定角色的起始和结束索引
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)


@dataclass
class RunResult:
    """Return results of the worker executions.

    Run results follow an "all-or-nothing" policy where the run is successful if and
    only if ALL local workers managed by this agent complete successfully.
    """
    """
    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.
    
    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.
    
    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.
    
    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker's final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent's restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    """
    
    # 表示当前工作器的状态，可能是成功或失败
    state: WorkerState
    # 存储成功工作器返回值的字典，按照全局排名映射
    return_values: Dict[int, Any] = field(default_factory=dict)
    # 存储失败工作器的失败信息的字典，按照全局排名映射
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)
    
    def is_failed(self) -> bool:
        # 检查当前工作器状态是否为失败
        return self.state == WorkerState.FAILED
class ElasticAgent(abc.ABC):
    """An agent process responsible for managing one or more worker processes.

    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        logger.exception("worker 0 failed with exit code : %s", failure.exit_code)
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the agent.

        Supports retrying the worker group on failures up to ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """Return the ``WorkerGroup`` for the given ``role``.

        Note that the worker group is a mutable object and hence in a
        multi-threaded/process environment it may change state.
        Implementors are encouraged (but not required) to return
        a defensive read-only copy.
        """
        raise NotImplementedError


class SimpleElasticAgent(ElasticAgent):
    """An ``ElasticAgent`` that manages one particular type of worker role.

    An ``ElasticAgent`` that manages workers (``WorkerGroup``) for a single ``WorkerSpec``
    such as one particular type of worker role.
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        # Initialize the SimpleElasticAgent instance with a WorkerGroup based on spec
        self._worker_group = WorkerGroup(spec)
        # Initialize remaining restarts based on the maximum allowed restarts from spec
        self._remaining_restarts = self._worker_group.spec.max_restarts
        # Initialize _store attribute to None
        self._store = None
        # Initialize exit barrier timeout for agent
        self._exit_barrier_timeout = exit_barrier_timeout
        # Initialize total execution time to zero
        self._total_execution_time = 0
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        return self._worker_group


        # 返回当前对象的 worker_group 属性，即代理对象管理的工作组
        return self._worker_group



    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        r"""Start ``worker_group.spec.local_world_size`` number of workers.

        This is according to worker spec for the worker group .
        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError


        # 抽象方法：启动 worker_group.spec.local_world_size 数量的工作进程

        # 根据工作组规格启动一定数量的工作进程，返回一个以 local_rank 到 worker id 的映射的字典
        raise NotImplementedError



    @abc.abstractmethod
    def _stop_workers(
        self, worker_group: WorkerGroup, is_restart: bool = False
    ) -> None:
        r"""Stop all workers in the given worker group.

        Implementors must deal with workers in all states defined by
        ``WorkerState``. That is, it must gracefully handle stopping
        non-existent workers, unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError


        # 抽象方法：停止给定工作组中的所有工作进程

        # 实现者必须处理所有 WorkerState 定义的工作状态，即优雅地处理停止不存在的工作进程、不健康（阻塞）的工作进程等
        raise NotImplementedError



    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        r"""Check on the workers for the ``worker_group``.

        This function also returns the new state of the worker group.
        """
        raise NotImplementedError


        # 抽象方法：监控 worker_group 的工作进程状态

        # 检查 worker_group 中的工作进程状态，同时返回工作组的新状态
        raise NotImplementedError



    @abc.abstractmethod
    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, is_restart: bool = False
    ) -> None:
        """Clean up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
        """
        raise NotImplementedError


        # 抽象方法：清理代理工作过程中分配的所有资源

        # 清理代理工作期间分配的所有资源，包括向子进程发送的信号（默认为 SIGTERM）
        raise NotImplementedError



    @prof


        # 使用 @prof 装饰器，可能用于性能分析或记录函数执行时间等目的
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
        # 从 worker_group 中获取 worker spec 对象
        spec = worker_group.spec

        # 记录运行时间用于性能统计，记录 "RENDEZVOUS" 阶段的持续时间
        with self.record_duration("RENDEZVOUS"):
            # 使用 rdzv_handler 获取下一个 rendezvous 的信息
            rdzv_info = spec.rdzv_handler.next_rendezvous()
        # 获取 rendezvous 的 store
        store = rdzv_info.store
        # 获取 worker group 的 rank
        group_rank = rdzv_info.rank
        # 获取 worker group 的 world size
        group_world_size = rdzv_info.world_size

        # 如果指定了 master_addr，则使用指定的地址；否则使用 bootstrap 信息中的地址
        master_addr = spec.master_addr or rdzv_info.bootstrap_store_info.master_addr
        # 如果指定了 master_port，则使用指定的端口；否则使用 bootstrap 信息中的端口
        master_port = spec.master_port or rdzv_info.bootstrap_store_info.master_port

        # 将当前的 store 设置为类成员变量
        self._store = store

        # 记录分配 worker rank 的运行时间
        with self.record_duration("ASSIGN_WORKER_RANKS"):
            # 调用 _assign_worker_ranks 方法分配 worker ranks
            workers = self._assign_worker_ranks(
                store, group_rank, group_world_size, spec
            )
        # 将分配的 workers 设置到 worker_group 对象中
        worker_group.workers = workers
        # 将 store 设置到 worker_group 对象中
        worker_group.store = store
        # 将 group_rank 设置到 worker_group 对象中
        worker_group.group_rank = group_rank
        # 将 group_world_size 设置到 worker_group 对象中
        worker_group.group_world_size = group_world_size
        # 将 master_addr 设置到 worker_group 对象中
        worker_group.master_addr = master_addr
        # 将 master_port 设置到 worker_group 对象中
        worker_group.master_port = master_port

        # 计算重启次数
        restart_count = spec.max_restarts - self._remaining_restarts

        # 记录 rendezvous 完成后的详细信息到日志中
        logger.info(
            "[%(role)s] Rendezvous complete for workers. Result:\n"
            "  restart_count=%(restart_count)s\n"
            "  master_addr=%(master_addr)s\n"
            "  master_port=%(master_port)s\n"
            "  group_rank=%(group_rank)s\n"
            "  group_world_size=%(group_world_size)s\n"
            "  local_ranks=%(local_ranks)s\n"
            "  role_ranks=%(role_ranks)s\n"
            "  global_ranks=%(global_ranks)s\n"
            "  role_world_sizes=%(role_world_sizes)s\n"
            "  global_world_sizes=%(global_world_sizes)s\n",
            {
                "role": spec.role,
                "restart_count": restart_count,
                "master_addr": master_addr,
                "master_port": master_port,
                "group_rank": group_rank,
                "group_world_size": group_world_size,
                "local_ranks": [worker.local_rank for worker in workers],
                "role_ranks": [worker.role_rank for worker in workers],
                "global_ranks": [worker.global_rank for worker in workers],
                "role_world_sizes": [worker.role_world_size for worker in workers],
                "global_world_sizes": [worker.world_size for worker in workers],
            },
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        # 获取工作组的角色
        role = worker_group.spec.role
        logger.info("[%s] Rendezvous'ing worker group", role)

        # 在启动新的工作组之前进行集合操作
        self._rendezvous(worker_group)

        logger.info("[%s] Starting worker group", role)
        # 启动工作组中的所有工作进程，并将工作进程的本地排名与 ID 关联
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id

        # 设置工作组状态为健康
        worker_group.state = WorkerState.HEALTHY

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Restart (stops, rendezvous, starts) all local workers in the group."""
        # 获取工作组的角色
        role = worker_group.spec.role
        logger.info("[%s] Stopping worker group", role)
        
        # 停止工作组中的所有工作进程，准备重启
        self._stop_workers(worker_group, is_restart=True)
        # 设置工作组状态为停止
        worker_group.state = WorkerState.STOPPED
        # 初始化工作组，重新启动工作进程
        self._initialize_workers(worker_group)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            # 调用实际运行的方法
            result = self._invoke_run(role)
            # 计算执行时间
            self._total_execution_time = int(time.monotonic() - start_time)
            # 记录运行时的指标
            self._record_metrics(result)
            # 记录工作进程事件
            self._record_worker_events(result)
            return result
        except RendezvousGracefulExitError as e:
            logger.info("Rendezvous gracefully exited: %s", e)
        except SignalException as e:
            logger.warning("Received %s death signal, shutting down workers", e.sigval)
            # 处理信号异常，关闭工作进程
            self._shutdown(e.sigval)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                # 如果没有调用关闭方法，则正常关闭
                self._shutdown()
            # 记录执行时间，以防在运行过程中发生异常
            self._total_execution_time = int(time.monotonic() - start_time)
    # 返回一个表示失败事件的 Event 对象
    def get_event_failed(self) -> Event:
        # 调用内部方法构造事件，设置状态为 FAILED，来源为 AGENT，记录详细错误信息
        return self._construct_event(
            state="FAILED",
            source=EventSource.AGENT,
            raw_error=traceback.format_exc(),
        )

    # 返回一个表示成功事件的 Event 对象
    def get_event_succeeded(self) -> Event:
        # 调用内部方法构造事件，设置状态为 SUCCEEDED，来源为 AGENT，无详细错误信息
        return self._construct_event(
            state="SUCCEEDED",
            source=EventSource.AGENT,
        )

    # 记录每个 worker 的事件信息到日志
    def _record_worker_events(self, result: RunResult) -> None:
        # 遍历 worker 组中的每个 worker
        for worker in self._worker_group.workers:
            # 获取该 worker 的失败信息（如果有的话）
            failure = result.failures.get(worker.global_rank)
            # 获取该 worker 的状态信息
            state: str = self._get_worker_state(worker, result)
            # 将失败信息转换成 JSON 格式，如果失败信息存在，否则为 None
            raw_error = json.dumps(failure.error_file_data) if failure else None
            # 记录构造的事件到日志中，设置事件来源为 WORKER
            record(self._construct_event(state, EventSource.WORKER, worker, raw_error))

    # 获取 worker 的状态信息
    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        # 获取该 worker 的失败信息
        failure = result.failures.get(worker.global_rank)
        # 如果 worker 的状态为 UNHEALTHY 或 FAILED 且没有失败信息，则表示被 torchelastic agent 终止
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and not failure:
            return "TERMINATED"  # worker 被终止
        # 如果存在失败信息或 worker 在返回值中，则返回其状态值
        elif failure or worker.global_rank in result.return_values:
            return result.state.value  # 返回 worker 当前状态值
        else:
            raise ValueError(f"Unknown worker: {worker.global_rank}")  # 未知的 worker

    # 记录代码块执行时间的上下文管理器
    @contextmanager
    def record_duration(self, state: str):
        start_time = time.perf_counter()  # 记录起始时间
        try:
            yield  # 执行被装饰的代码块
        finally:
            end_time = time.perf_counter()  # 记录结束时间
            duration_ms = (end_time - start_time) * 1000  # 计算执行时间（毫秒）
            # 记录执行时间相关的事件到日志中，设置事件来源为 AGENT
            record(
                self._construct_event(
                    state=state, source=EventSource.AGENT, duration_ms=duration_ms
                )
            )

    # 构造事件对象的内部方法
    def _construct_event(
        self,
        state: str,
        source: EventSource,
        worker: Optional[Worker] = None,
        raw_error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ):
        # 返回一个包含特定属性的 Event 对象
        pass  # 方法未提供具体实现，仅定义了方法签名
    ) -> Event:
        # 获取当前工作组的引用
        wg = self._worker_group
        # 获取工作组规格的引用
        spec = wg.spec
        # 构建元数据字典，包括工作组的全局大小和入口点名称
        md = {
            "group_world_size": wg.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
        }
        # 如果有指定的 worker
        if worker:
            # 添加本地排名、角色排名和角色全局大小信息到元数据字典
            md["local_rank"] = (worker.local_rank,)
            md["role_rank"] = (worker.role_rank,)
            md["role_world_size"] = (worker.role_world_size,)
            # 获取全局排名和工作器的 ID，将其转换为字符串形式
            global_rank = worker.global_rank
            worker_id = str(worker.id)
        else:
            # 如果没有指定 worker，则将全局排名和工作器 ID 设为 None
            global_rank = None
            worker_id = None
        # 将元数据字典转换为 JSON 格式的字符串
        md_str = json.dumps(md)
        # 构建完整的元数据字典
        metadata = {
            "run_id": spec.rdzv_handler.get_run_id(),  # 获取运行 ID
            "global_rank": global_rank,  # 全局排名
            "group_rank": wg.group_rank,  # 工作组排名
            "worker_id": worker_id,  # 工作器 ID
            "role": spec.role,  # 角色
            "hostname": _get_fq_hostname(),  # 获取完全限定的主机名
            "state": state,  # 状态
            "total_run_time": self._total_execution_time,  # 总运行时间
            "rdzv_backend": spec.rdzv_handler.get_backend(),  # 获取分布式同步后端
            "raw_error": raw_error,  # 原始错误信息
            "metadata": md_str,  # 元数据字符串
            "agent_restarts": spec.max_restarts - self._remaining_restarts,  # 代理重启次数
            "duration_ms": duration_ms,  # 持续时间（毫秒）
        }
        # 返回包含所有元数据的 Event 对象
        return Event(
            f"torchelastic.worker.status.{state}", source=source, metadata=metadata
        )

    def _record_metrics(self, group_results: RunResult):
        # 检查整体运行结果是否失败
        is_failed = group_results.is_failed()
        # 记录运行的不稳定性指标
        self._record_flakiness_metric(is_failed)
        # 获取工作组的规格信息
        spec = self._worker_group.spec
        # 检查是否发生过重启
        restarts_happened = self._remaining_restarts != spec.max_restarts
        # 记录总运行次数
        put_metric(f"workers.{spec.role}.run_total", 1)
        # 根据条件记录运行成功带重试、无重试的指标
        self._record_metric_with_condition(
            "run_success_with_retries", not is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not is_failed and not restarts_happened
        )
        # 根据条件记录运行失败带重试、无重试的指标
        self._record_metric_with_condition(
            "run_failed_with_retries", is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", is_failed and not restarts_happened
        )

    def _record_metric_with_condition(self, metric_name, condition):
        # 获取工作组的规格信息
        spec = self._worker_group.spec
        # 根据条件记录指标
        if condition:
            put_metric(f"workers.{spec.role}.{metric_name}", 1)
        else:
            put_metric(f"workers.{spec.role}.{metric_name}", 0)

    def _record_flakiness_metric(self, is_failed: bool = False):
        # 如果运行失败，则设定不稳定性为 100%
        if is_failed:
            flakiness = 100.0
        else:
            # 否则，根据剩余重启次数计算不稳定性百分比
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        # 再次获取工作组的规格信息
        spec = self._worker_group.spec
        # 记录不稳定性指标
        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))
    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role
        # 定义一个方法 `_invoke_run`，接受一个字符串参数 `role`，默认为 `DEFAULT_ROLE`，返回 `RunResult` 类型的对象

        spec = self._worker_group.spec
        # 从 `_worker_group` 中获取规格信息

        role = spec.role
        # 从规格信息中获取角色

        logger.info(
            "[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name()
        )
        # 使用日志记录器输出日志信息，指示为特定角色启动工作进程，并显示入口点名称

        self._initialize_workers(self._worker_group)
        # 调用 `_initialize_workers` 方法，初始化工作进程组

        monitor_interval = spec.monitor_interval
        # 从规格信息中获取监控间隔时间

        rdzv_handler = spec.rdzv_handler
        # 从规格信息中获取协调处理程序对象

        while True:
            assert self._worker_group.state != WorkerState.INIT
            # 断言工作进程组的状态不是 INIT

            time.sleep(monitor_interval)
            # 等待指定的监控间隔时间

            run_result = self._monitor_workers(self._worker_group)
            # 调用 `_monitor_workers` 方法，监控工作进程组，返回运行结果对象

            state = run_result.state
            # 从运行结果对象中获取状态信息

            self._worker_group.state = state
            # 将工作进程组的状态更新为当前状态

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            # 记录指标数据，表示剩余重新启动次数

            put_metric(f"workers.{role}.{state.name.lower()}", 1)
            # 记录指标数据，表示特定状态的工作进程数为 1

            if state == WorkerState.SUCCEEDED:
                logger.info(
                    "[%s] worker group successfully finished."
                    " Waiting %s seconds for other agents to finish.",
                    role,
                    self._exit_barrier_timeout,
                )
                # 如果状态为 SUCCEEDED，则使用日志记录器输出日志信息，表示工作组成功完成，并等待其他代理程序完成

                self._exit_barrier()
                # 调用 `_exit_barrier` 方法，执行退出屏障操作

                return run_result
                # 返回运行结果对象

            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    logger.info(
                        "[%s] Worker group %s. "
                        "%s/%s attempts left;"
                        " will restart worker group",
                        role,
                        state.name,
                        self._remaining_restarts,
                        spec.max_restarts,
                    )
                    # 如果状态为 UNHEALTHY 或 FAILED，并且剩余重新启动次数大于 0，则使用日志记录器输出日志信息，表示将重新启动工作组

                    self._remaining_restarts -= 1
                    # 减少剩余重新启动次数计数

                    self._restart_workers(self._worker_group)
                    # 调用 `_restart_workers` 方法，重新启动工作进程组

                else:
                    self._stop_workers(self._worker_group)
                    # 调用 `_stop_workers` 方法，停止工作进程组

                    self._worker_group.state = WorkerState.FAILED
                    # 将工作进程组的状态更新为 FAILED

                    return run_result
                    # 返回运行结果对象

            elif state == WorkerState.HEALTHY:
                # 如果状态为 HEALTHY，则执行以下操作

                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                # 获取等待的节点数量，通过调用 `rdzv_handler` 的 `num_nodes_waiting` 方法

                group_rank = self._worker_group.group_rank
                # 获取工作进程组的组等级

                if num_nodes_waiting > 0:
                    logger.info(
                        "[%s] Detected %s "
                        "new nodes from group_rank=%s; "
                        "will restart worker group",
                        role,
                        num_nodes_waiting,
                        group_rank,
                    )
                    # 如果检测到等待的节点数大于 0，则使用日志记录器输出日志信息，表示将重新启动工作组

                    self._restart_workers(self._worker_group)
                    # 调用 `_restart_workers` 方法，重新启动工作进程组

            else:
                raise Exception(  # noqa: TRY002
                    f"[{role}] Worker group in {state.name} state"
                )
                # 如果状态不是预期的 SUCCEEDED、UNHEALTHY、FAILED、HEALTHY 中的任何一种，则抛出异常
    def _exit_barrier(self):
        """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
        # 记录日志：显示本地工作组的状态及等待其他代理完成的时间
        logger.info(
            "Local worker group finished (%s). "
            "Waiting %s seconds for other agents to finish",
            self._worker_group.state,
            self._exit_barrier_timeout,
        )
        start = time.time()  # 记录当前时间，用于计算等待时间
        try:
            # 使用存储工具的 barrier 方法等待所有代理完成
            store_util.barrier(
                store=self._store,
                world_size=self._worker_group.group_world_size,
                key_prefix=_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            # 记录日志：显示等待其他代理完成的总时间
            logger.info(
                "Done waiting for other agents. Elapsed: %s seconds",
                time.time() - start,
            )
        except SignalException as e:
            # 如果收到信号异常，记录警告日志并抛出异常
            logger.warning("Got termination signal: %s", e.sigval)
            raise
        except Exception:
            # 捕获所有异常，记录异常日志并显示等待过程中的总时间
            logger.exception(
                "Error waiting on exit barrier. Elapsed: %s seconds",
                time.time() - start,
            )
```