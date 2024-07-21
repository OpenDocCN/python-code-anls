# `.\pytorch\torch\distributed\distributed_c10d.py`

```py
# mypy: allow-untyped-defs
"""Distributed Collective Communication (c10d)."""

import collections.abc  # 导入标准库中的抽象基类集合
import contextlib  # 导入上下文管理工具
import hashlib  # 导入用于计算哈希的模块
import io  # 导入用于处理二进制数据的核心工具
import itertools  # 导入用于创建迭代器的工具
import logging  # 导入用于记录日志的模块
import os  # 导入与操作系统交互的功能
import pickle  # 导入用于序列化和反序列化Python对象的模块
import sys  # 导入与Python解释器交互的功能
import time  # 导入处理时间的模块
import warnings  # 导入用于处理警告的模块
from collections import namedtuple  # 导入命名元组
from datetime import timedelta  # 导入处理时间间隔的模块
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union  # 导入类型提示工具
from typing_extensions import deprecated  # 导入用于标记已弃用功能的工具

import torch  # 导入PyTorch深度学习库
from torch._C import _DistStoreError as DistStoreError  # 导入分布式存储错误
from torch._C._distributed_c10d import (
    _DistributedBackendOptions,  # 导入分布式后端选项
    _register_process_group,  # 导入注册进程组的功能
    _resolve_process_group,  # 导入解析进程组的功能
    _unregister_all_process_groups,  # 导入注销所有进程组的功能
    _unregister_process_group,  # 导入注销单个进程组的功能
    AllgatherOptions,  # 导入全聚集操作选项
    AllreduceCoalescedOptions,  # 导入累积全reduce操作选项
    AllreduceOptions,  # 导入全reduce操作选项
    AllToAllOptions,  # 导入全到全操作选项
    BarrierOptions,  # 导入屏障操作选项
    BroadcastOptions,  # 导入广播操作选项
    DebugLevel,  # 导入调试级别
    GatherOptions,  # 导入聚集操作选项
    get_debug_level,  # 导入获取调试级别的功能
    PrefixStore,  # 导入前缀存储
    ProcessGroup,  # 导入进程组
    ReduceOp,  # 导入reduce操作
    ReduceOptions,  # 导入reduce操作选项
    ReduceScatterOptions,  # 导入reduce散布操作选项
    ScatterOptions,  # 导入散布操作选项
    Store,  # 导入存储
    Work,  # 导入工作
)
from torch._utils_internal import set_pytorch_distributed_envs_from_justknobs  # 导入设置PyTorch分布式环境的内部工具
from torch.utils._typing_utils import not_none  # 导入类型工具，确保值不为None

from .c10d_logger import _exception_logger, _time_logger  # 从当前目录下的c10d_logger模块导入异常和时间记录器
from .constants import default_pg_nccl_timeout, default_pg_timeout  # 从当前目录下的constants模块导入默认的进程组超时时间
from .rendezvous import register_rendezvous_handler, rendezvous  # 从当前目录下的rendezvous模块导入注册会合处理器和会合函数，忽略F401警告

__all__ = [  # 定义当前模块公开的所有符号列表
    "Backend",  # 分布式后端
    "BackendConfig",  # 后端配置
    "GroupMember",  # 组成员
    "P2POp",  # 点对点操作
    "all_gather",  # 所有聚集操作
    "all_gather_coalesced",  # 累积所有聚集操作
    "all_gather_object",  # 所有聚集对象操作
    "all_reduce",  # 所有reduce操作
    "all_reduce_coalesced",  # 累积所有reduce操作
    "all_to_all",  # 所有到所有操作
    "all_to_all_single",  # 单个所有到所有操作
    "barrier",  # 屏障操作
    "batch_isend_irecv",  # 批量发送和接收操作
    "broadcast",  # 广播操作
    "send_object_list",  # 发送对象列表
    "recv_object_list",  # 接收对象列表
    "broadcast_object_list",  # 广播对象列表
    "destroy_process_group",  # 销毁进程组
    "gather",  # 聚集操作
    "gather_object",  # 聚集对象操作
    "get_backend_config",  # 获取后端配置
    "get_backend",  # 获取后端
    "get_rank",  # 获取当前进程在进程组中的等级
    "get_world_size",  # 获取进程组的进程数量
    "get_pg_count",  # 获取进程组的计数
    "group",  # 创建进程组
    "init_process_group",  # 初始化进程组
    "irecv",  # 异步接收
    "is_gloo_available",  # 检查是否可用Gloo后端
    "is_initialized",  # 检查是否已初始化
    "is_mpi_available",  # 检查是否可用MPI后端
    "is_backend_available",  # 检查指定后端是否可用
    "is_nccl_available",  # 检查是否可用NCCL后端
    "is_torchelastic_launched",  # 检查是否已启动TorchElastic
    "is_ucc_available",  # 检查是否可用UCC后端
    "isend",  # 异步发送
    "monitored_barrier",  # 监控屏障操作
    "new_group",  # 创建新组
    "new_subgroups",  # 通过枚举创建新子组
    "new_subgroups_by_enumeration",  # 通过枚举创建新子组
    "recv",  # 接收
    "reduce",  # reduce操作
    "reduce_scatter",  # reduce散布操作
    "scatter",  # 散布操作
    "scatter_object_list",  # 散布对象列表
    "send",  # 发送
    "supports_complex",  # 支持复杂操作
    "AllreduceCoalescedOptions",  # 累积全reduce操作选项
    "AllreduceOptions",  # 全reduce操作选项
    "AllToAllOptions",  # 全到全操作选项
    "BarrierOptions",  # 屏障操作选项
    "BroadcastOptions",  # 广播操作选项
    "GatherOptions",  # 聚集操作选项
    "PrefixStore",  # 前缀存储
    "ProcessGroup",  # 进程组
    "ReduceOp",  # reduce操作
    "ReduceOptions",  # reduce操作选项
    "ReduceScatterOptions",  # reduce散布操作选项
    "ScatterOptions",  # 散布操作选项
    "Store",  # 存储
    "DebugLevel",  # 调试级别
    "get_debug_level",  # 获取调试级别
    "Work",  # 工作
    "default_pg_timeout",  # 默认进程组超时时间
    "get_group_rank",  # 获取组内等级
    "get_global_rank",  # 获取全局等级
    "get_process_group_ranks",  # 获取进程组等级
    "reduce_op",  # reduce操作
    "all_gather_into_tensor",  # 将结果聚集到张量中
    "reduce_scatter_tensor",  # 张量reduce散布
    "get_node_local_rank",  # 获取节点本地等级
]

_MPI_AVAILABLE = True  # 表示MPI可用
_NCCL_AVAILABLE = True  # 表示NC
# 标志位，指示是否支持 UCC 后端
_UCC_AVAILABLE = True

# 导入 pickle 模块的 Pickler 和 Unpickler 类
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


# 改变来自 torch._C._distributed_c10d 中所有公共类型的 __module__ 属性
def _export_c_types() -> None:
    # 需要改变 __module__ 的公共类型列表
    _public_types_to_change_module = [
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        BarrierOptions,
        BroadcastOptions,
        GatherOptions,
        PrefixStore,
        ProcessGroup,
        ReduceOp,
        ReduceOptions,
        ReduceScatterOptions,
        ScatterOptions,
        Store,
        DebugLevel,
        get_debug_level,
        Work,
    ]
    # 遍历列表中的每个类型，改变其 __module__ 属性
    for type in _public_types_to_change_module:
        type.__module__ = "torch.distributed.distributed_c10d"


# 调用函数，改变公共类型的 __module__ 属性
_export_c_types()

# 尝试导入 MPI 后端，并设定其 __module__ 属性
try:
    from torch._C._distributed_c10d import ProcessGroupMPI

    ProcessGroupMPI.__module__ = "torch.distributed.distributed_c10d"
    # 将 ProcessGroupMPI 添加到 __all__ 列表中
    __all__ += ["ProcessGroupMPI"]
except ImportError:
    _MPI_AVAILABLE = False

# 尝试导入 NCCL 后端，并设定其 __module__ 属性
try:
    from torch._C._distributed_c10d import ProcessGroupNCCL

    ProcessGroupNCCL.__module__ = "torch.distributed.distributed_c10d"
    # 将 ProcessGroupNCCL 添加到 __all__ 列表中
    __all__ += ["ProcessGroupNCCL"]
except ImportError:
    _NCCL_AVAILABLE = False

# 尝试导入 Gloo 后端，并设定其 __module__ 属性
try:
    from torch._C._distributed_c10d import _ProcessGroupWrapper, ProcessGroupGloo

    ProcessGroupGloo.__module__ = "torch.distributed.distributed_c10d"
    # 将 ProcessGroupGloo 添加到 __all__ 列表中
    __all__ += ["ProcessGroupGloo"]
except ImportError:
    _GLOO_AVAILABLE = False

# 尝试导入 UCC 后端，并设定其 __module__ 属性
try:
    from torch._C._distributed_c10d import ProcessGroupUCC

    ProcessGroupUCC.__module__ = "torch.distributed.distributed_c10d"
    # 将 ProcessGroupUCC 添加到 __all__ 列表中
    __all__ += ["ProcessGroupUCC"]
except ImportError:
    _UCC_AVAILABLE = False

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 进程组包装器存储前缀
PG_WRAPPER_STORE_PREFIX = "pg_wrapper"


# 检查某个 ReduceOp 是否支持复数计算
def supports_complex(reduceOp: ReduceOp) -> bool:
    """
    返回 True 表示支持该 ReduceOp，否则返回 False。
    某些 ReduceOp 不支持复数计算，调用时可能返回错误的值。
    """
    denyList = [
        ReduceOp.MAX,
        ReduceOp.MIN,
        ReduceOp.PRODUCT,
        ReduceOp.BAND,
        ReduceOp.BOR,
        ReduceOp.BXOR,
    ]
    return reduceOp not in denyList


class Backend(str):
    """
    用于表示后端的枚举类。

    可用的后端有：GLOO、NCCL、UCC、MPI，以及其他注册的后端。

    该类的值为小写字符串，如 ``"gloo"``。可以作为属性访问，例如 ``Backend.NCCL``。

    该类可以直接调用以解析字符串，例如 ``Backend(backend_str)`` 将检查 ``backend_str`` 是否有效，
    如果有效则返回解析后的小写字符串。也接受大写字符串。

    """
    pass
    # 用于表示未定义的后端名称
    UNDEFINED = "undefined"
    # GLOO 后端的名称
    GLOO = "gloo"
    # NCCL 后端的名称
    NCCL = "nccl"
    # UCC 后端的名称
    UCC = "ucc"
    # MPI 后端的名称
    MPI = "mpi"

    # 定义了一个命名元组 _BackendPlugin，包含两个字段：creator_fn 和 extended_api
    _BackendPlugin = namedtuple("_BackendPlugin", ["creator_fn", "extended_api"])

    # 字典，存储后端名称到 _BackendPlugin 命名元组的映射关系
    _plugins: Dict[str, _BackendPlugin] = {}

    # 包含所有后端名称的列表
    backend_list = [UNDEFINED, GLOO, NCCL, UCC, MPI]

    # 字典，将设备类型映射到默认的后端名称
    default_device_backend_map: Dict[str, str] = {
        "cpu": GLOO,
        "cuda": NCCL,
    }

    # 字典，存储每个后端能够支持的设备类型列表
    backend_capability: Dict[str, List[str]] = {
        GLOO: ["cpu", "cuda"],
        NCCL: ["cuda"],
        UCC: ["cpu", "cuda"],
        MPI: ["cpu", "cuda"],
    }

    # 字典，将后端名称映射到 ProcessGroup.BackendType 枚举类型
    backend_type_map: Dict[str, ProcessGroup.BackendType] = {
        UNDEFINED: ProcessGroup.BackendType.UNDEFINED,
        GLOO: ProcessGroup.BackendType.GLOO,
        NCCL: ProcessGroup.BackendType.NCCL,
        UCC: ProcessGroup.BackendType.UCC,
    }

    def __new__(cls, name: str):
        """创建并返回类的新实例。"""
        # 如果 name 不是字符串类型，抛出 ValueError 异常
        if not isinstance(name, str):
            raise ValueError("Backend constructor parameter must be string-ish")
        # 获取名为 name 的属性值，如果不存在则使用 UNDEFINED
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        # 如果值等于 UNDEFINED，则将 name 转换为小写作为值
        if value == Backend.UNDEFINED:
            value = name.lower()
        return value

    @classmethod
    def register_backend(
        cls,
        name,
        func,
        extended_api=False,
        devices: Optional[Union[str, List[str]]] = None,
# 定义一个后端配置类
class BackendConfig:
    """Backend configuration class."""

    # 返回所有设备与后端的键值对，用逗号分隔
    def __repr__(self):
        """Return all the device:backend pairs separated by commas."""
        return ",".join(
            f"{device}:{backend}" for device, backend in self.device_backend_map.items()
        )

    # 返回设备与后端的映射字典
    def get_device_backend_map(self) -> Dict[str, Backend]:
        """Return backend map of the device."""
        return self.device_backend_map


# 定义一个废弃的枚举类 _reduce_op
class _reduce_op:
    """
    Deprecated enum-like class.

    For reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """

    # 初始化函数，从 ReduceOp.RedOpType.__members__ 中获取键值对，并添加为实例属性
    def __init__(self):
        # __members__ 是一个存储枚举类键值对的字典
        for k, v in ReduceOp.RedOpType.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.RedOpType.__members__

    # 重载 __getattribute__ 方法，返回对象的属性
    @deprecated(
        "`torch.distributed.reduce_op` is deprecated, "
        "please use `torch.distributed.ReduceOp` instead",
        category=FutureWarning,
    )
    def __getattribute__(self, key):
        return object.__getattribute__(self, key)


# 创建 _reduce_op 的实例对象 reduce_op
reduce_op = _reduce_op()


# 定义一个点对点操作类 P2POp
class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
    """

    # 初始化函数，接收操作函数、张量、对等进程、进程组和标签作为参数
    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        peer: int,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
    ):
        """Init."""
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag

    # 创建并返回一个 P2POp 类的新实例对象
    def __new__(
        cls,
        op: Callable,
        tensor: torch.Tensor,
        peer: int,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
    ):
        """Create and return a new instance of the class."""
        _check_op(op)  # 检查操作函数的有效性
        _check_single_tensor(tensor, "tensor")  # 检查张量的有效性
        return object.__new__(cls)
    # 定义对象的字符串表示形式的特殊方法 __repr__
    def __repr__(self):
        # 获取当前对象所属组的排名
        my_group_rank = get_rank(self.group)
        # 如果存在组，则获取与同行组相关的排名；否则使用同行对象的排名
        peer_group_rank = (
            get_group_rank(self.group, self.peer) if self.group else self.peer
        )
        # 获取操作函数的名称
        op_name = self.op.__name__
        # 获取当前对象所属组的名称，如果不存在组则使用默认名称 "default_pg"
        group_name = self.group.group_name if self.group else "default_pg"
        
        # 根据操作函数的名称确定发送或接收的角色
        if "send" in op_name:
            # 发送操作中，当前对象为发送方，同行对象为接收方
            s = my_group_rank
            d = peer_group_rank
        elif "recv" in op_name:
            # 接收操作中，当前对象为接收方，同行对象为发送方
            s = peer_group_rank
            d = my_group_rank
        else:
            # 若操作名称既非 "send" 也非 "recv"，则调用父类的 __repr__ 方法返回默认字符串表示
            return super().__repr__()

        # 返回对象的字符串表示形式，包括操作名称、组名称、发送方、接收方、张量形状和数据类型
        return f"P2POp({op_name} pg={group_name}, s={s}, d={d},  {self.tensor.shape}, {self.tensor.dtype})"
    def __init__(self):
        """
        Initialize _World instance.

        Sets up default values for process group state and mappings.
        """
        # 默认的进程组对象，用于包含整个集群的所有进程
        self._default_pg = None
        # 存储每个进程组的协同操作状态，映射为进程组到协同操作列表的字典
        self._pg_coalesce_state: Dict[ProcessGroup, List[_CollOp]] = {}
        # 存储每个进程组的默认设备，映射为进程组到 torch.device 对象的字典
        self._pg_default_device: Dict[ProcessGroup, torch.device] = {}

    @property
    def default_pg(self) -> Optional[ProcessGroup]:
        """
        Get or set the default ProcessGroup.

        This ProcessGroup includes all ranks of the cluster and is used
        when a ProcessGroup is required but none is provided explicitly.
        """
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value) -> None:
        self._default_pg = value

    @property
    def pg_map(self) -> Dict[ProcessGroup, Tuple[str, Store]]:
        """
        Mapping from ProcessGroup to backend name and store.

        Provides a dictionary mapping each ProcessGroup to a tuple
        containing its backend name and associated store.
        """
        global _pg_map
        return _pg_map

    @property
    def pg_names(self) -> Dict[ProcessGroup, str]:
        """
        Mapping from ProcessGroup to its name.

        Provides a dictionary mapping each ProcessGroup to its corresponding name.
        """
        global _pg_names
        return _pg_names
    def pg_group_ranks(self) -> Dict[ProcessGroup, Dict[int, int]]:
        """
        返回进程组的全局排名到本地排名的映射。

        TODO 不要暴露映射，改为暴露细粒度操作
        """
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_backend_config(self) -> Dict[ProcessGroup, str]:
        """
        返回进程组的后端配置。

        TODO 不要暴露映射，改为暴露细粒度操作
        """
        global _pg_backend_config
        return _pg_backend_config

    @property
    def group_count(self) -> int:
        """
        返回默认命名时的进程组数量。

        TODO 不要暴露 group_count，考虑使用其他方式
        """
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value: int) -> None:
        """用于计算使用全局同步时进程组名称的方法。"""
        global _group_count
        _group_count = value

    @property
    def tags_to_pg(self) -> Dict[str, List[ProcessGroup]]:
        """
        返回标签到进程组的映射。

        """
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> Dict[ProcessGroup, str]:
        """
        返回进程组到标签的映射。
        """
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_coalesce_state(self) -> Dict[ProcessGroup, List[_CollOp]]:
        """
        返回进程组的合并状态。
        """
        return self._pg_coalesce_state

    @property
    def pg_default_device(self) -> Dict[ProcessGroup, torch.device]:
        """
        返回进程组的默认设备。
        """
        return self._pg_default_device

    @property
    def pg_config_info(self) -> List[Dict[str, Any]]:
        """
        返回包含进程组和后端配置信息的字典列表。

        包括它们的唯一标识和配置（类型和排名）。
        """
        config_info: List[Dict[str, Any]] = []
        default_pg_size = _get_group_size(None)
        for pg in self.pg_map.keys():
            ranks = self.pg_group_ranks[pg]
            config_info.append(
                {
                    "pg_name": self.pg_names[pg],
                    "pg_desc": pg.group_desc,
                    "backend_config": self.pg_backend_config[pg],
                    "ranks": list(ranks.keys())
                    if len(ranks) != default_pg_size
                    else [],  # 当所有排名参与进程组时，'ranks' 是空列表
                    "group_size": len(ranks),
                    "group_count": self.group_count,
                }
            )
        return config_info
# 创建一个名为 `_world` 的全局变量，并初始化为 `_World` 类的实例。
_world = _World()
"""Holds the singleton instance of ``_World`` used by c10. Experimental extension point to override it"""

# 定义 `_WorldMeta` 类，作为 `group` 和 `GroupMember` 的元类。
class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """

    # 类属性 `WORLD`，返回当前 `_world.default_pg`，即默认的进程组对象。
    @property
    def WORLD(cls) -> Optional[ProcessGroup]:
        return _world.default_pg

    # 设置类属性 `WORLD`，将 `_world.default_pg` 设置为给定的进程组对象 `pg`。
    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):
        _world.default_pg = pg


# 定义 `group` 类，使用 `_WorldMeta` 作为元类，表示一个组对象的占位符。
class group(metaclass=_WorldMeta):
    """Group class. Placeholder."""
    pass


# 定义 `GroupMember` 类，使用 `_WorldMeta` 作为元类，表示一个组成员对象。
class GroupMember(metaclass=_WorldMeta):
    """Group member class."""
    NON_GROUP_MEMBER = -100


def _get_default_timeout(backend: Backend) -> timedelta:
    # 根据后端类型确定默认超时时间。
    # 如果后端是 `Backend.NCCL`，则返回 `default_pg_nccl_timeout`；
    # 否则返回通用的 `default_pg_timeout`。
    if backend == Backend.NCCL:
        if not isinstance(default_pg_nccl_timeout, timedelta):
            # 如果 `default_pg_nccl_timeout` 不是 `timedelta` 类型，则发出警告。
            warnings.warn(
                "Attempted to get default timeout for nccl backend, but NCCL support is not compiled"
            )
            return default_pg_timeout
        return default_pg_nccl_timeout
    else:
        return default_pg_timeout


def _check_valid_timeout(timeout: Any) -> None:
    # 检查超时时间 `timeout` 的类型是否为 `timedelta`，如果不是则抛出 `TypeError`。
    if not isinstance(timeout, timedelta):
        raise TypeError(
            f"Expected timeout argument to be of type datetime.timedelta, got {timeout}"
        )


# 默认的进程组状态初始化方法
_default_pg_init_method: Optional[str] = None

# 存储基于障碍的键前缀
STORE_BASED_BARRIER_PREFIX = "store_based_barrier_key"


def _get_pg_default_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """
    Return the device to use with ``group`` for control flow usage (object collectives, barrier).

    There are selection rules:
        1. If user specifies exactly one backend in ``init_process_group`` call:
            use that backend
        2. Else if user specifies multiple "device:backend" pairs in init_process_group:
            If "cpu" is among those pairs, use "cpu" (because the object is in cpu memory);
            Otherwise, use the first backend (sort of a random pick).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.device: The device to use with ``group``.

    """
    # 如果未提供特定的 `group`，则使用默认的进程组。
    group = group or _get_default_group()
    if group in _world.pg_default_device:
        # 如果已经在之前的搜索中找到并缓存，直接返回缓存的设备。
        return _world.pg_default_device[group]
    """
    如果 `group` 不是 `ProcessGroup` 类型的实例：
    - 为了向后兼容，处理 `group` 实际上是一个后端（如 `ProcessGroupGloo`）而不是 PyTorch 2.0 中定义的 `ProcessGroup`。
    """
    if not isinstance(group, ProcessGroup):
        # 提供向后兼容性，当传入的 `group` 实际上是一个后端（如 `ProcessGroupGloo`）而不是 `ProcessGroup`。
        warnings.warn(
            f"You are using a Backend {type(group)} as a ProcessGroup. "
            "This usage is deprecated since PyTorch 2.0. Please use a public API "
            "of PyTorch Distributed instead.",
            FutureWarning,
            stacklevel=3,
        )
        # 大多数用户使用私有 API 创建 Gloo 以进行对象集合操作
        _world.pg_default_device[group] = torch.device("cpu")
        return _world.pg_default_device[group]

    """
    ``group._device_types`` 是一个 pybind 属性，返回 `group` 支持的设备列表
    （如 "cpu", "cuda" 等）。如果 `group` 支持多个设备，此列表可能包含多个设备。
    """
    devices = group._device_types

    if len(devices) == 1:
        # 用户在 `init_process_group` 中固定了一个后端
        _world.pg_default_device[group] = devices[0]
    elif len(devices) == 0:
        # 没有后端注册到此 ProcessGroup 中（可能因为没有进行集合操作？）我们选择 cpu 作为默认值，希望这会延迟初始化 Gloo 或其他可用的 cpu 后端。
        _world.pg_default_device[group] = torch.device("cpu")
    elif torch.device("cpu") in devices:
        # 此 ProcessGroup 中有多个后端，其中包含 cpu。
        # 由于对象位于 cpu 内存中，优先选择 cpu，无需设备复制。
        _world.pg_default_device[group] = torch.device("cpu")
    else:
        # 后端列表中没有 cpu。随机选择第一个后端。
        _world.pg_default_device[group] = devices[0]

    logger.info(
        "Using device %s for object " "collectives.", _world.pg_default_device[group]
    )
    return _world.pg_default_device[group]
@_time_logger
def _store_based_barrier(
    rank,
    store,
    group_name,
    rendezvous_count,
    timeout,
    logging_interval=timedelta(seconds=10),
) -> None:
    """
    Store based barrier for synchronizing processes.

    Barrier based on store which is used for synchronizing processes after
    ``init_process_group`` or ``new_group``. Intended to be used only with
    those two methods and is not a generic alternative to ``barrier()``.
    """
    # 构建存储键名，用于在存储中同步进程
    store_key = f"{STORE_BASED_BARRIER_PREFIX}:{group_name}"
    # 向存储中添加键，表示进程已经开始同步
    store.add(store_key, 1)
    # 记录调试信息，标记添加的键名和当前进程的排名
    logger.debug("Added key: %s to store for rank: %s", store_key, rank)

    # 等待所有工作进程检查存储中的键
    world_size = rendezvous_count
    worker_count = store.add(store_key, 0)

    last_worker_key = f"{store_key}:last_worker"
    # 如果所有工作进程都已经检查完毕，则设置最后一个工作进程的标记
    if worker_count == world_size:
        store.set(last_worker_key, "1")

    # 调整超时时间，确保至少是10秒加上每千个进程1秒的时间间隔，以减少超时的可能性
    # 此值经过了规模测试的实证发现
    logging_interval = max(logging_interval, timedelta(seconds=10 + world_size / 1000))

    start = time.time()
    while True:
        try:
            # 等待存储中的键发生变化，超时后抛出异常
            store.wait([last_worker_key], logging_interval)
            break
        except RuntimeError as e:
            worker_count = store.add(store_key, 0)
            # 定期打印状态信息以便跟踪进度
            logger.debug(
                "Waiting in store based barrier to initialize process group for "
                "rank: %s, key: %s (world_size=%s, num_workers_joined=%s, timeout=%s error=%s)",
                rank,
                store_key,
                world_size,
                worker_count,
                timeout,
                e,
            )

            # 如果超时时间已过，则抛出异常
            if timedelta(seconds=(time.time() - start)) > timeout:
                raise DistStoreError(
                    "Timed out initializing process group in store based barrier on "
                    f"rank {rank}, for key: {store_key} (world_size={world_size}, "
                    f"num_workers_joined={worker_count}, timeout={timeout} error={e})"
                )

    # 记录信息，标记存储基础障碍已经完成
    logger.info(
        "Rank %s: Completed store-based barrier for key:%s with %s nodes.",
        rank,
        store_key,
        world_size,
    )


def _rank_not_in_group(group: Optional[ProcessGroup]) -> bool:
    """Check if the current process's rank is not in a given group."""
    # 检查当前进程的排名是否不在给定组中
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


def _warn_not_in_group(op_name) -> None:
    global_rank = -1 if GroupMember.WORLD is None else GroupMember.WORLD.rank()
    # 发出警告，表示当前操作在不属于给定组的全局排名上运行
    warnings.warn(
        f"Running {op_name} on global rank {global_rank} which does not "
        "belong to the given group."
    )
def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    # 若 group 是全局的 WORLD 组，则直接返回全局排名 global_rank
    if group is GroupMember.WORLD:
        return global_rank
    # 如果 group 不在已注册的 _world.pg_group_ranks 中，抛出异常
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    # 从 _world.pg_group_ranks[group] 中获取组内各个全局排名的字典 group_ranks
    group_ranks = _world.pg_group_ranks[group]
    # 如果 global_rank 不在 group_ranks 中，抛出异常
    if global_rank not in group_ranks:
        raise ValueError(f"Global rank {global_rank} is not part of group {group}")

    # 返回 global_rank 对应的组内排名
    return group_ranks[global_rank]


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """
    Translate a group rank into a global rank.

    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the global rank from.
        group_rank (int): Group rank to query.

    Returns:
        Global rank of ``group_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    # 若 group 是全局的 WORLD 组，则直接返回组内排名 group_rank
    if group is GroupMember.WORLD:
        return group_rank
    # 如果 group 不在已注册的 _world.pg_group_ranks 中，抛出异常
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    # 遍历 _world.pg_group_ranks[group]，找到与 group_rank 对应的全局排名
    for rank, grp_rank in _world.pg_group_ranks[group].items():
        if grp_rank == group_rank:
            return rank
    # 如果找不到对应的全局排名，抛出异常
    raise ValueError(f"Group rank {group_rank} is not part of group {group}")


# TODO: remove this once the ecosystem moves away from it.
@deprecated(
    "`torch.distributed.distributed_c10d._get_global_rank` is deprecated, "
    "please use `torch.distributed.distributed_c10d.get_global_rank` instead",
    category=FutureWarning,
)
def _get_global_rank(group, rank) -> int:
    """Use get_global_rank as this method is deprecated."""
    # 直接调用 get_global_rank 函数
    return get_global_rank(group, rank)


def get_process_group_ranks(group: ProcessGroup) -> List[int]:
    """
    Get all ranks associated with ``group``.

    Args:
        group (ProcessGroup): ProcessGroup to get all ranks from.

    Returns:
        List of global ranks ordered by group rank.
    """
    # 返回与 group 相关的所有全局排名列表，按照组内排名排序
    return list(_world.pg_group_ranks[group].keys())


def _get_group_size(group) -> int:
    """Get a given group's world size."""
    # 如果 group 是全局的 WORLD 或者为 None，则获取默认进程组的大小
    if group is GroupMember.WORLD or group is None:
        default_pg = _get_default_group()
        return default_pg.size()
    # 否则返回 group 的大小
    return group.size()


def _get_group_size_by_name(group_name: str) -> int:
    # 根据给定的 group_name 解析出相应的进程组，然后返回其大小
    group = _resolve_process_group(group_name)
    return group.size()
# 按照给定的排名列表和标签确定进程组名
def _resolve_group_name_by_ranks_and_tag(ranks: List[int], tag: str) -> str:
    # TODO(yifu): remove this function once ranks + tag is not a supported
    # identifier for process group for functional collectives.
    # 使用给定的标签和排名列表查找进程组
    group = _find_pg_by_ranks_and_tag(tag, ranks)
    # 如果未找到对应的进程组，抛出数值错误
    if group is None:
        raise ValueError("")
    # 返回找到的进程组的组名
    return group.group_name


# 检查参数是否为单个张量
def _check_single_tensor(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a single tensor."""
    # 如果参数不是 torch.Tensor 类型，抛出类型错误
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type torch.Tensor
             but got {type(param)} instead."""
        )


# 检查参数是否为张量列表
def _check_tensor_list(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a list of tensors."""
    # 如果参数不是 list 类型，抛出类型错误
    if not isinstance(param, list):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} instead."""
        )
    # 如果参数列表中有任何元素不是 torch.Tensor 类型，抛出类型错误
    elif not all(isinstance(p, torch.Tensor) for p in param):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} with elements of type {[type(p) for p in param]}."""
        )


# 将对象转换为可迭代对象
def _as_iterable(obj) -> collections.abc.Iterable:
    return obj if isinstance(obj, list) else (obj,)


# 确保所有张量具有相同的数据类型
def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    for tensor in itertools.chain.from_iterable(map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        # 允许混合复数及其元素类型
        if tensor_dtype.is_complex:
            tensor_dtype = (
                torch.float32 if tensor_dtype == torch.complex64 else torch.complex128
            )
        
        # 检查所有张量是否具有相同的数据类型
        if last_dtype is None:
            last_dtype = tensor_dtype
        else:
            if last_dtype != tensor_dtype:
                raise ValueError(
                    "Invalid usage of tensors with different dtypes"
                    f"Found {last_dtype} and  {tensor.dtype}"
                )


# 检查操作是否为 isend 或 irecv
def _check_op(op) -> None:
    """Check that the ``op`` is either isend or irecv."""
    # 如果操作不是 isend 或 irecv，抛出数值错误
    if op not in [isend, irecv]:
        raise ValueError(
            "Invalid ``op``. Expected ``op`` "
            "to be of type ``torch.distributed.isend`` or "
            "``torch.distributed.irecv``."
        )


# 检查 P2POp 列表是否有效
def _check_p2p_op_list(p2p_op_list) -> None:
    """
    Check that the ``p2p_op_list`` is a list of P2POp instances.

    Also, check that all ops use the same group.
    """
    # 如果 p2p_op_list 不是列表类型，或者列表中的任意元素不是 P2POp 类型，抛出数值错误
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list
    ):
        raise ValueError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "to be of type ``torch.distributed.P2POp``."
        )
    
    # 获取第一个 P2POp 实例的组信息
    group = p2p_op_list[0].group
    # 检查所有的 p2p_op 对象是否属于同一组
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        # 如果不是同一组，则抛出值错误异常
        raise ValueError("All ops need to use the same group.")
def is_mpi_available() -> bool:
    """
    Check if the MPI backend is available.

    Returns:
        bool: True if the MPI backend is available, otherwise False.
    """
    return _MPI_AVAILABLE


def is_nccl_available() -> bool:
    """
    Check if the NCCL backend is available.

    Returns:
        bool: True if the NCCL backend is available, otherwise False.
    """
    return _NCCL_AVAILABLE


def is_gloo_available() -> bool:
    """
    Check if the Gloo backend is available.

    Returns:
        bool: True if the Gloo backend is available, otherwise False.
    """
    return _GLOO_AVAILABLE


def is_ucc_available() -> bool:
    """
    Check if the UCC backend is available.

    Returns:
        bool: True if the UCC backend is available, otherwise False.
    """    
    return _UCC_AVAILABLE


def is_backend_available(backend: str) -> bool:
    """
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """
    # If the backend has an ``is_backend_available`` function, return the result of that function directly
    available_func = getattr(torch.distributed, f"is_{backend.lower()}_available", None)
    if available_func:
        return available_func()

    # Otherwise, check if the backend name is in the list of available backends
    return backend.lower() in Backend.backend_list


def is_initialized() -> bool:
    """
    Check if the default process group has been initialized.

    Returns:
        bool: True if the default process group has been initialized, otherwise False.
    """
    return GroupMember.WORLD is not None


def is_torchelastic_launched() -> bool:
    """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    Returns:
        bool: True if TORCHELASTIC_RUN_ID environment variable is set, indicating torchelastic launch, otherwise False.
    """
    return os.getenv("TORCHELASTIC_RUN_ID") is not None


def _is_barrier_after_init() -> int:
    """
    Get the setting for whether a barrier should be performed after process group initialization.

    Returns:
        int: 0 if no barrier is requested, 1 if barrier is requested (controlled by TORCH_DIST_INIT_BARRIER environment variable).
    """
    return int(os.getenv("TORCH_DIST_INIT_BARRIER", "0"))


def _get_default_group() -> ProcessGroup:
    """
    Get the default process group created by init_process_group.

    Returns:
        ProcessGroup: Default process group.
    Raises:
        ValueError: If the default process group has not been initialized.
    """
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    if TYPE_CHECKING:
        return not_none(GroupMember.WORLD)
    else:
        return GroupMember.WORLD


def _get_default_store() -> Store:
    """
    Get the default store created by init_process_group.

    Returns:
        Store: Default store.
    Raises:
        ValueError: If the default process group has not been initialized.
    """
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    default_pg = _get_default_group()
    _, default_store = _world.pg_map[default_pg]
    return default_store
def _update_default_pg(pg) -> None:
    # 更新全局变量 `_world.default_pg` 为指定的进程组 `pg`
    _world.default_pg = pg
    # 如果 `pg` 不为 None 并且不是非组成员，则获取其排名；否则设为 -1
    rank = pg.rank() if pg is not None and pg != GroupMember.NON_GROUP_MEMBER else -1
    # 设置全局排名
    torch._C._distributed_c10d._set_global_rank(rank)


def get_backend_config(group: Optional[ProcessGroup] = None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
    # 如果未指定 `group`，获取默认进程组 `_get_default_group()`
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    # 如果当前进程不在 `pg` 中，抛出异常
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")
    # 获取进程组的后端配置信息
    backend_config = _world.pg_backend_config.get(pg)
    return str(not_none(backend_config))


def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
    """
    Return the backend of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    # 如果未指定 `group`，获取默认进程组 `_get_default_group()`
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    # 如果当前进程不在 `pg` 中，抛出异常
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")
    # 获取进程组的存储信息
    pg_store = _world.pg_map[pg] if pg in _world.pg_map else None
    return Backend(not_none(pg_store)[0])


def _get_process_group_uid(pg: ProcessGroup) -> int:
    # 尝试获取指定进程组 `pg` 的后端信息
    backend = None
    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        pass
    # 如果 NCCL 可用且后端为 NCCL，则返回其 UID；否则返回 -1
    if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
        return backend.uid
    return -1


def _get_pg_config(group: Optional[ProcessGroup] = None) -> Dict[str, Any]:
    """
    Return the pg configuration of the given process group.

    """
    # 如果未指定 `group`，获取默认进程组 `_get_default_group()`
    if group is None:
        pg = _get_default_group()
    else:
        pg = group
    # 返回进程组的配置信息字典
    return {
        "pg_name": _get_process_group_name(pg),
        "pg_desc": pg.group_desc,
        "backend_config": get_backend_config(pg),
        "pg_size": _get_group_size(pg),
        "ranks": get_process_group_ranks(pg),
    }


def _get_all_pg_configs() -> List[Dict[str, Any]]:
    """
    Return the pg configuration of all the process groups.

    """
    # 初始化配置信息列表
    config_info: List[Dict[str, Any]] = []
    # 遍历所有进程组，获取每个进程组的配置信息并添加到列表中
    for pg in _world.pg_map.keys():
        config_info.append(_get_pg_config(pg))
    return config_info


def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
    # 返回全局变量 `_world` 中的进程组数量
    return _world.group_count


def get_node_local_rank(fallback_rank: Optional[int] = None) -> int:
    """
    Return the local rank of the node.

    Args:
        fallback_rank (int, optional): The fallback rank to return if the
            local rank cannot be determined. Defaults to None.

    Returns:
        int: The local rank of the node.

    """
    """
    返回当前进程相对于节点的本地排名。

    在语义上，这是一个有用的概念，用于将进程映射到设备。
    例如，在具有 8 个加速器的节点上，您可以使用节点本地排名来决定将进程绑定到哪个加速器设备。

    在实践中，节点本地排名的实际分配由进程启动器在 pytorch 外部处理，并通过 `LOCAL_RANK` 环境变量通信。

    Torchrun 将自动填充 `LOCAL_RANK`，但其他启动器可能不会。如果未指定 `LOCAL_RANK`，则此 API 将根据提供的关键字参数 'fallback_rank' 进行回退，如果指定了的话，否则将引发错误。目的是允许编写既可以在单设备环境下运行也可以在多设备环境下运行的应用程序而不报错。

    """
    # 如果当前环境变量中包含 `LOCAL_RANK`，则返回其整数值
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # 否则，如果提供了 fallback_rank 参数，则返回其整数值
    elif fallback_rank is not None:
        return int(fallback_rank)
    # 如果上述条件都不满足，则引发运行时错误，指示 `LOCAL_RANK` 未在环境变量中设置
    raise RuntimeError(
        "LOCAL_RANK is not in the environment. Consider passing fallback_rank to allow `get_node_local_rank` to work, "
        "assuming you are not running in a multi-device context and want the code to run locally instead."
    )
# 设置给定进程组的超时时间，允许用户使用不同于默认值的超时时间。

@_exception_logger
@_time_logger
def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[torch.device] = None,
) -> None:
    """
    初始化默认的分布式进程组。

    这也将初始化分布式包。

    有两种主要初始化进程组的方法：
        1. 明确指定 ``store``、``rank`` 和 ``world_size``。
        2. 指定 ``init_method``（一个 URL 字符串），指示在哪里/如何发现对等体。
           可选地指定 ``rank`` 和 ``world_size``，或在 URL 中编码所有必需的参数并省略它们。
    """
    如果未指定任何一种方法，则假定“init_method”为“env://”。

    .. note:: 要启用 ``backend == Backend.MPI``，PyTorch 需要在支持 MPI 的系统上从源代码构建。

    .. note:: 对多个后端的支持是实验性的。当前当未指定后端时，将创建 ``gloo`` 和 ``nccl`` 后端。``gloo`` 后端用于 CPU 张量的集合操作，``nccl`` 后端用于 CUDA 张量的集合操作。可以通过传递格式为 "<device_type>:<backend_name>,<device_type>:<backend_name>" 的字符串来指定自定义后端，例如 "cpu:gloo,cuda:custom_backend"。
    """

    global _world

    global _backend
    global _default_pg_init_method

    如果已经初始化了默认进程组（GroupMember.WORLD 不为空），则抛出 ValueError。

    从 justknobs 设置 PyTorch 分布式环境变量。

    # 根据导入顺序的不同，一些 trace_rules 函数可能在导入阶段被评估。
    # 在这种情况下，由于导入循环依赖，这些函数可能无法正确添加分布式相关的规则。
    # 我们需要在运行时清除 lru_cache，以确保这些 trace_rules 的正确性。
    #
    # 由于此 API 必须在编译所有分布式代码之前调用，因此在这里清除缓存应该是安全的。
    如果 "torch._dynamo" 在 sys.modules 中：
        清除 torch._dynamo.trace_rules 的 lru_cache。

    确保不能同时指定 store 和 init_method。

    如果使用 store：
        确保 world_size 大于 0，如果使用 store。
        确保 rank 大于等于 0，如果使用 store。
    否则，如果未指定 init_method：
        将 init_method 设置为 "env://”。

    如果指定了 backend：
        将 backend 转换为 Backend 对象。
    否则：
        将 backend 设置为未定义的 Backend。

    如果未指定 timeout：
        使用默认超时时间，根据 backend 来获取。

    检查 timeout 是否有效。

    """
    Group name is not visible to users unless they access
    internals of c10d. This means we can ignore the value
    they provide as it not exposed in a public way.
    """
    根据空列表和非哈希名称创建进程组名称。

    如果 backend 为 Backend.MPI：
        如果 world_size 和 rank 均不为 -1，则发出警告，因为它们将由 MPI 运行时分配。

        使用 _new_process_group_helper 创建默认进程组，并更新默认进程组。
    else:
        # 兼容老版本 API
        if store is None:
            # 如果未提供存储对象，则使用 rendezvous 函数创建一个迭代器
            rendezvous_iterator = rendezvous(
                not_none(init_method), rank, world_size, timeout=timeout
            )
            # 从迭代器中获取存储对象、排名和世界大小，并设置超时时间
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)

            # 使用 PrefixStore 避免不同系统（如 RPC）中的键名冲突，特别是在多租户存储场景下
            store = PrefixStore("default_pg", store)

        # 创建一个新的进程组，并获取默认进程组和未使用的返回值
        default_pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name,
            pg_options=pg_options,
            timeout=timeout,
            device_id=device_id,
            group_desc="default_pg",
        )
        # 更新默认进程组
        _update_default_pg(default_pg)

    # 将 GroupMember.WORLD 的大小映射为排名到排名的字典，设置给 _world.pg_group_ranks[GroupMember.WORLD]
    _world.pg_group_ranks[GroupMember.WORLD] = {i: i for i in range(GroupMember.WORLD.size())}  # type: ignore[attr-defined, index]
    
    # 获取全局变量中 GroupMember.WORLD 的首个后端并设置给 _backend
    _backend = _world.pg_map[not_none(GroupMember.WORLD)][0]
    
    # 将 init_method 设置为默认进程组初始化方法
    _default_pg_init_method = init_method

    # 保存旧的异常处理钩子
    old_hook = sys.excepthook
    # 创建异常处理前缀
    excepthook_prefix = f"[rank{get_rank()}]"

    # 定义分布式异常处理钩子函数
    def _distributed_excepthook(*args):
        # 保存旧的标准错误流并替换为新的字符串 IO 流
        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            # 调用旧的异常处理钩子
            old_hook(*args)
        finally:
            # 恢复标准错误流
            sys.stderr = old_stderr
        # 获取捕获的异常信息并添加前缀
        msg = buf.getvalue()
        msg = "\n".join(
            f"{excepthook_prefix}: {s}" if s != "" else "" for s in msg.split("\n")
        )
        # 将格式化后的异常信息写入标准错误流并刷新
        sys.stderr.write(msg)
        sys.stderr.flush()

    # 将分布式异常处理钩子设置为当前的异常处理钩子
    sys.excepthook = _distributed_excepthook

    # 如果初始化后需要屏障
    if _is_barrier_after_init() == 1:
        # 在方法返回后确保所有进程组（包括可能的全局变量）在所有排名上都正确更新的屏障
        logger.debug(
            "Performing barrier after ProcessGroup initialization since "
            "TORCH_DIST_INIT_BARRIER = 1"
        )
        # 如果使用 MPI 后端，则执行 MPI 屏障
        if backend == Backend.MPI:
            barrier()
        else:
            # 否则使用基于存储的屏障，因为 barrier() 使用了大量默认设备并可能破坏 NCCL 内部状态
            _store_based_barrier(rank, store, group_name, world_size, timeout)
def _get_split_source(pg):
    split_from = None  # 初始化变量 split_from 为 None
    if pg.bound_device_id:  # 如果 pg 绑定了设备 ID
        split_from = pg._get_backend(pg.bound_device_id)  # 获取绑定设备 ID 的后端
    elif pg is _world.default_pg:  # 如果 pg 是默认的全局进程组
        try:
            split_from = pg._get_backend(torch.device("cuda"))  # 尝试获取 CUDA 设备的后端
        except RuntimeError:
            # 如果没有与此后端关联的 CUDA 设备
            pass

    if not split_from or not split_from.supports_splitting:
        return None  # 如果 split_from 不存在或不支持分割，则返回 None

    # 如果需要，通过剥离进程组包装器来找到可以分割的后端
    while _GLOO_AVAILABLE and isinstance(split_from, _ProcessGroupWrapper):
        split_from = split_from.wrapped_pg  # 获取被包装的进程组

    return split_from  # 返回可以分割的后端


def _shutdown_backend(pg):
    """
    Try to shut down the backend of a process group.
    Currently, only ProcessGroupNCCL backend is supported.
    No op for other backends.
    """
    backend = None  # 初始化变量 backend 为 None
    try:
        backend = pg._get_backend(torch.device("cuda"))  # 尝试获取 CUDA 设备的后端
    except RuntimeError:
        pass

    if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
        # 如果 NCCL 可用并且后端是 ProcessGroupNCCL 类型
        # 显式调用 shutdown 来确保释放 NCCL 资源
        backend._shutdown()


def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    pg_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world  # 声明使用全局变量 _world

    if group_name in _world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )  # 如果指定的组名已经被创建，则抛出 ValueError

    if device_id is not None and (device_id.index is None or device_id.type != "cuda"):
        raise ValueError(
            "init_process_group device_id parameter must be a cuda device with an "
            "id, e.g. cuda:0, not just cuda or cpu"
        )  # 如果 device_id 参数不是 CUDA 设备或没有指定 id，则抛出 ValueError

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)  # 检查 timeout 是否有效

    if pg_tag not in [None, ""]:
        # 使用相同的标签和排名集创建结果相同的底层进程组
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc  # 如果 group_desc 为 None，则设为 "undefined"

    # 如果全局组中的 group_ranks_in_group 列表为空，则创建默认组
    is_default_group = len(global_ranks_in_group) == 0
    # 如果当前进程已经初始化并且满足以下条件之一：
    # 1. 全局秩列表的长度等于默认组的大小
    # 2. 默认组绑定了设备ID，导致早期连接初始化
    # 则从默认组中获取分裂源
    if is_initialized() and (
        len(global_ranks_in_group) == _get_default_group().size()
        or _get_default_group().bound_device_id
    ):
        split_from = _get_split_source(_get_default_group())
    else:
        split_from = None

    # 如果不是默认组（即指定了group_ranks），检查当前进程是否是新组的成员
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # 如果使用`ncclCommSplit`（或类似的API）创建通信器，需要在
            # 新组的父组中的所有秩上调用`ncclCommSplit`，即使这些秩不在新组中也是如此。
            # 这是NCCL API的要求，否则会出现同步问题。
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    # 创建前缀存储对象，用于存储指定组的前缀
    prefix_store = PrefixStore(f"{group_name}/", store)
    
    # 设置基础进程组选项，指定后端类型和超时时间
    base_pg_options = ProcessGroup.Options(backend=str(backend))
    base_pg_options._timeout = timeout
    
    # 创建进程组对象
    pg: ProcessGroup = ProcessGroup(
        prefix_store, group_rank, group_size, base_pg_options
    )
    
    # 如果指定了设备ID，将其绑定到进程组对象中
    if device_id:
        pg.bound_device_id = device_id
    
    # 创建后端配置对象，用于指定后端类型
    backend_config = BackendConfig(backend)
    backend_class: torch._C._distributed_c10d.Backend
    
    # 设置组名和组描述信息到后端
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)
    
    # 如果指定了设备ID，并且该设备支持分裂操作，则连接该设备
    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)
    
    # 更新全局状态信息
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)
    
    _world.pg_backend_config[pg] = str(backend_config)
    
    # 如果pg_tag未指定或为空字符串，则使用默认标签
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"
    
    # 更新全局标签映射
    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    
    # 返回创建的进程组对象和前缀存储对象
    return pg, prefix_store
# 销毁给定的进程组，并且反初始化分布式包。

def destroy_process_group(group: Optional[ProcessGroup] = None):
    # 全局变量，用于存储当前的世界进程组状态
    global _world

    # 如果传入的进程组是非组成员，则直接返回
    if group == GroupMember.NON_GROUP_MEMBER:
        return

    # 如果未传入进程组参数，则使用默认的 WORLD 进程组
    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    # 断言进程组不为空
    assert pg is not None

    # 检查指定的进程组是否存在于 _world.pg_map 中，否则抛出异常
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # 当用户注册 Python onCompletion 钩子时，这些钩子会在不同的线程上运行。
    # 当前，ProcessGroup 的析构函数会等待这些钩子完成，但析构函数可能会在
    # Python 解释器退出后完成。此时再尝试获取 GIL 会导致崩溃。我们可以在运行
    # 钩子时重新激活解释器，或者保持主解释器活动直到所有工作和钩子完成。当前
    # 的实现选择了后者。因此，这里显式调用 _wait_for_pending_works() 等待待定
    # 的钩子完成。
    if pg.name().lower() == "nccl" and pg._has_hooks():
        pg._wait_for_pending_works()

    # 如果没有传入进程组参数，或者传入的是 WORLD 进程组，则依次关闭所有后端
    # 按 pg 名称的顺序关闭，因为在某些版本的 NCCL 中 ncclCommAbort() 是一个集体
    # 调用。
    for pg_to_shutdown in sorted(
        _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
    ):
        _shutdown_backend(pg_to_shutdown)

    # 更新默认进程组为 None
    _update_default_pg(None)

    # 清空 _world 中的各种映射和配置
    _world.pg_map.clear()
    _world.pg_names.clear()
    _world.pg_group_ranks.clear()
    _world.pg_backend_config.clear()
    _world.pg_to_tag.clear()
    _world.tags_to_pg.clear()
    _world.pg_coalesce_state.clear()
    _world.pg_default_device.clear()

    # 取消所有进程组的注册
    _unregister_all_process_groups()

    # 当进程组没有显式名称时（只有 WORLD 进程组可以有显式名称），我们使用全局
    # 变量 _world.group_count 生成名称。在销毁时需要重置计数器，以便在某些训练
    # 器因故障恢复后重新创建进程组时生成一致的值。
    #
    # 仅在销毁 WORLD 进程组时重置计数器，因为如果此进程组处于良好状态，我们不
    # 需要处理故障。
    _world.group_count = 0
    else:
        # 关闭后端连接
        _shutdown_backend(pg)
        # 从全局变量中删除相关映射
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        # 如果存在默认设备映射，则删除
        if pg in _world.pg_default_device:
            del _world.pg_default_device[pg]
        # 如果进程组在协同状态字典中，则发出警告并清除
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned."
            )
            del _world.pg_coalesce_state[pg]

        # 获取并删除与进程组相关的标签
        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        # 如果存在标签，则尝试从标签映射中删除该进程组
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                # 如果标签以"ptd:"开头，则从默认标签映射中删除
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        # 注销进程组
        _unregister_process_group(pg.group_name)
# 返回当前进程在给定进程组中的排名，若未提供进程组则使用默认进程组
def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    # 检查当前进程是否不在指定的进程组中，如果是则返回-1
    if _rank_not_in_group(group):
        return -1

    # 获取默认进程组
    default_pg = _get_default_group()
    # 如果未提供进程组或者提供的是全局组，则返回默认进程组的排名
    if group is None or group is GroupMember.WORLD:
        return default_pg.rank()

    # 否则返回指定进程组中的排名
    return get_group_rank(group, default_pg.rank())


# 返回当前进程组中的进程数目
def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    # 检查当前进程是否不在指定的进程组中，如果是则返回-1
    if _rank_not_in_group(group):
        return -1

    # 返回指定进程组的进程数目
    return _get_group_size(group)


# 发送张量数据的异步操作
def isend(
    tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0
) -> Optional[Work]:
    """
    Send a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    # 检查待发送的张量是否符合单一张量的要求
    _check_single_tensor(tensor, "tensor")
    # 检查当前进程是否不在指定的进程组中，如果是则发出警告并返回None
    if _rank_not_in_group(group):
        _warn_not_in_group("isend")
        return None

    # 如果张量是复数类型，则将其转换为实数类型
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    # 如果未提供进程组或者提供的是全局组，则使用默认进程组
    if group is None or group is GroupMember.WORLD:
        pg = _get_default_group()
    else:
        pg = group
        # 获取目标进程在指定进程组中的排名
        dst = get_group_rank(pg, dst)

    # 发送张量到指定目标进程，返回分布式请求对象
    return pg.send([tensor], dst, tag)


# 接收张量数据的异步操作
def irecv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
) -> Optional[Work]:
    """
    Receives a tensor asynchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to receive into.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    # 检查当前进程是否不在指定的进程组中，如果是则返回None
    if _rank_not_in_group(group):
        return None

    # 如果未提供进程组或者提供的是全局组，则使用默认进程组
    if group is None or group is GroupMember.WORLD:
        pg = _get_default_group()
    else:
        pg = group

    # 接收来自指定源进程的张量数据，返回分布式请求对象
    return pg.recv([tensor], src, tag)
    Args:
        tensor (Tensor): 要填充接收数据的张量。
        src (int, optional): 全局进程组中的源排名（与 ``group`` 参数无关）。
            如果未指定，则从任何进程接收。
        group (ProcessGroup, optional): 要操作的进程组。如果为 None，
            将使用默认进程组。
        tag (int, optional): 用于匹配接收与远程发送的标签。

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    # 检查张量是否符合单一张量的要求
    _check_single_tensor(tensor, "tensor")
    # 如果当前进程不在指定的进程组中，发出警告并返回 None
    if _rank_not_in_group(group):
        _warn_not_in_group("irecv")
        return None

    # 如果张量是复数类型，将其转换为实数视图
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    # 确定要使用的进程组
    if group is None or group is GroupMember.WORLD:
        pg = _get_default_group()
    else:
        pg = group

    # 根据参数选择接收操作
    if src is None:
        # 从任意源接收张量数据
        return pg.recv_anysource([tensor], tag)
    else:
        if pg is GroupMember.WORLD:
            # 从指定源接收张量数据（在全局进程组中）
            return pg.recv([tensor], src, tag)
        else:
            # 从指定源接收张量数据（在指定的进程组中）
            group_src_rank = get_group_rank(pg, src)
            return pg.recv([tensor], group_src_rank, tag)
@_exception_logger
def send(
    tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0
) -> None:
    """
    Send a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument).
            Destination rank should not be the same as the rank of the current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv

    """
    # 检查当前进程是否为目标进程，如果是则抛出异常
    if get_rank() == dst:
        raise ValueError(
            "Invalid destination rank: destination rank should not be the same as "
            "the rank of the current process."
        )

    # 检查并确保要发送的是单个张量
    _check_single_tensor(tensor, "tensor")
    # 如果当前进程不在指定的进程组中，发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("send")
        return None

    # 如果张量是复数类型，将其视为实数处理
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    # 根据进程组发送张量到目标进程
    if group is None or group is GroupMember.WORLD:
        default_pg = _get_default_group()
        default_pg.send([tensor], dst, tag).wait()
    else:
        group_dst_rank = get_group_rank(group, dst)
        group.send([tensor], group_dst_rank, tag).wait()


@_exception_logger
def recv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
) -> int:
    """
    Receives a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    """
    # 检查并确保要接收的是单个张量
    _check_single_tensor(tensor, "tensor")
    # 如果当前进程不在指定的进程组中，发出警告并返回-1
    if _rank_not_in_group(group):
        _warn_not_in_group("recv")
        return -1

    # 如果张量是复数类型，将其视为实数处理
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    # 确定要使用的进程组
    if group is None:
        pg = _get_default_group()
    else:
        pg = group

    # 根据是否指定了源进程，接收来自指定源或任意源的张量数据
    if src is None:
        work = pg.recv_anysource([tensor], tag)
        work.wait()
        src_rank = work._source_rank()
        if group is None or group is GroupMember.WORLD:
            return src_rank
        else:
            return get_global_rank(pg, src_rank)
    else:
        if group is None or group is GroupMember.WORLD:
            pg.recv([tensor], src, tag).wait()
        else:
            group_src_rank = get_group_rank(pg, src)
            pg.recv([tensor], group_src_rank, tag).wait()
        return src
    # 重载 __getattribute__ 方法，用于获取对象的属性
    def __getattribute__(self, name):
        # 如果属性名在指定的列表中
        if name in [
            "is_success",
            "exception",
            "wait",
            "source_rank",
            "_source_rank",
            "result",
            "synchronize",
        ]:
            # 抛出值错误异常，提示调用了不合法的属性名
            raise ValueError(f"Illegal to call {name} on IllegalWork object")
class _CoalescingManager:
    # 定义一个私有类 `_CoalescingManager`
    def __init__(self):
        # 初始化方法，创建一个空的工作列表 `works`，类型为 `List[Work]`
        self.works: List[Work] = []

    # 添加工作到工作列表的方法
    def append(self, work: Work):
        # 如果传入的工作对象有效（非空），则将其添加到工作列表中
        if work:
            self.works.append(work)

    # 等待所有工作完成的方法
    def wait(self):
        # 遍历工作列表中的每一个工作对象，并等待其完成
        for work in self.works:
            work.wait()


@contextlib.contextmanager
def _coalescing_manager(
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    async_ops: Optional[bool] = False,
):
    """
    Context manager used to coalesce collectives or P2P operations when possible.

    Args:
        group (`ProcessGroup`, optional): The process group to work on. If None,
            the default process group will be used.
        device (`torch.device`, optional): Default is None, set to a device if
            there isn't a `**_coalesced` implementation by the backend.
        async_ops (`bool`, optional): whether the coalesced ops are async ops.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # Synchronous ops
        >>> with _coalescing_manager():
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> # Asynchronous ops
        >>> with _coalescing_manager(async_ops=True) as cm:
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> cm.wait()

    .. warning::
       :func:`_coalescing_manager` currently do not support coalescing
       all-reduces with different reduce operators, e.g.  `ReduceOp.SUM` mixed
       with `ReduceOp.PRODUCT`.
    """
    # 获取默认的进程组，如果未指定，则使用默认进程组
    group = group or _get_default_group()
    # 检查并获取指定进程组的协同操作列表
    op_list = _world.pg_coalesce_state.setdefault(group, [])
    # 如果协同操作列表不为空，则抛出错误
    if op_list:
        raise ValueError(
            "ProcessGroup has non-empty op list at the start of coalescing"
        )
    # 如果设备有效，则启动设备的协同操作
    if device:
        group._start_coalescing(device)
    # 创建一个 `_CoalescingManager` 实例 `cm`
    cm = _CoalescingManager()
    # 使用 yield 将 `cm` 作为上下文管理器的返回值
    yield cm
    # 在上下文管理器结束后，移除指定进程组的协同操作列表
    op_list = _world.pg_coalesce_state.pop(group)
    if op_list:
        # 如果操作列表不为空，则执行以下操作
        # 支持“快速路径”合并的集合被捕获
        # 请参考相应集合API的实现
        # 当前支持：
        # - 合并的`all_reduce`
        # - 合并的`all_gather_into_tensor`
        # - 合并的`reduce_scatter_tensor`
        op0 = op_list[0].op
        # 获取第一个操作的类型
        if op0 == all_reduce:
            # 如果第一个操作是all_reduce
            tensors = []
            for op in op_list:
                tensors.append(op.tensor)
            # 创建AllreduceCoalescedOptions对象
            all_reduce_opts = AllreduceCoalescedOptions()
            all_reduce_opts.reduceOp = not_none(op_list[0].redop)
            # 执行集合操作，将数据tensors使用all_reduce_opts进行集合
            work = group.allreduce_coalesced(tensors, all_reduce_opts)
        elif op0 == all_gather_into_tensor:
            # 如果第一个操作是all_gather_into_tensor
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(not_none(op.dst_tensor))
            # 执行集合操作，将数据inputs收集到outputs中
            work = group.allgather_into_tensor_coalesced(outputs, inputs)
        elif op0 == reduce_scatter_tensor:
            # 如果第一个操作是reduce_scatter_tensor
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(not_none(op.dst_tensor))
            # 创建ReduceScatterOptions对象
            reduce_opts = ReduceScatterOptions()
            reduce_opts.reduceOp = not_none(op_list[0].redop)
            # 执行集合操作，将数据inputs分散到outputs中
            work = group.reduce_scatter_tensor_coalesced(outputs, inputs, reduce_opts)
        else:
            # 如果操作类型不支持快速路径合并，则引发断言错误
            raise AssertionError(
                f"Coalescing manager does not support fast-path coalescing of {op0}, "
                f"yet {op0} is still recorded in op list. This is an internal error of c10d."
            )

    if device:
        # 如果设备存在，则执行以下操作
        # 旧的方式允许上下文管理器中的每个coll通过Python绑定调用C++对应部分
        work = group._end_coalescing(device)

    if async_ops:
        # 如果存在异步操作，则将work添加到上下文管理器中
        cm.append(work)  # type: ignore[possibly-undefined]
    else:
        # 否则，等待work完成
        work.wait()  # type: ignore[possibly-undefined]
# 异步发送或接收一批张量，并返回请求列表
def batch_isend_irecv(p2p_op_list):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. NCCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
        >>> recv_tensor = torch.randn(2, dtype=torch.float32)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the NCCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    """
    # 检查 p2p_op_list 的有效性
    _check_p2p_op_list(p2p_op_list)
    # 获取操作列表中第一个操作的通信组
    group = p2p_op_list[0].group
    # 获取第一个操作使用的设备
    device = p2p_op_list[0].tensor.device
    # 如果设备类型是 cuda
    if device.type == "cuda":
        # 使用 NCCL 风格的合并管理器，支持异步操作
        with _coalescing_manager(group, device, async_ops=True) as cm:
            # 对每个 p2p 操作执行相应的发送或接收操作
            for p2p_op in p2p_op_list:
                p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        # 返回合并管理器中的任务列表
        return cm.works
    else:
        # 对于 Gloo 后端的向后兼容支持
        reqs = []
        # 遍历 p2p_op_list 中的每个操作
        for p2p_op in p2p_op_list:
            # 执行操作并获取请求对象
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            # 如果存在工作对象，则添加到请求列表中
            if work:
                reqs.append(work)
        # 返回请求对象列表
        return reqs


@_exception_logger
def broadcast(tensor, src, group=None, async_op=False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.
    
    Args:
        tensor: The tensor to broadcast.
        src: Source rank from which the tensor is broadcasted.
        group: The group where the collective operation is applied.
        async_op: Flag indicating if the operation should be asynchronous.
    Args:
        tensor (Tensor): 要发送的数据，如果当前进程的排名等于 src 参数指定的排名，则发送数据；否则用于保存接收到的数据。
        src (int): 全局进程组中的源排名（与 group 参数无关）。
        group (ProcessGroup, optional): 要操作的进程组。如果为 None，则使用默认进程组。
        async_op (bool, optional): 是否将此操作设置为异步操作。

    Returns:
        如果 async_op 设置为 True，则返回异步工作句柄。
        如果 async_op 为 False 或者不属于该进程组，则返回 None。

    """
    # 检查输入的 tensor 是否符合单个 tensor 的要求
    _check_single_tensor(tensor, "tensor")
    # 检查当前进程是否不在指定的进程组中
    if _rank_not_in_group(group):
        # 如果不在指定的进程组中，则发出警告并返回
        _warn_not_in_group("broadcast")
        return

    # 创建 BroadcastOptions 对象，并设置相关参数
    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0  # 广播的根张量索引设置为 0
    opts.asyncOp = async_op

    # 如果 group 为 None 或者是默认的 WORLD 进程组
    if group is None or group is GroupMember.WORLD:
        # 获取默认的进程组
        default_pg = _get_default_group()
        # 在默认进程组上进行广播操作
        work = default_pg.broadcast([tensor], opts)
    else:
        # 获取在指定进程组中的源排名
        group_src_rank = get_group_rank(group, src)
        opts.rootRank = group_src_rank
        # 在指定进程组上进行广播操作
        work = group.broadcast([tensor], opts)

    # 如果设置为异步操作，则返回工作句柄
    if async_op:
        return work
    else:
        # 如果不是异步操作，则等待广播完成
        work.wait()
# 使用装饰器将异常记录器应用于下面定义的函数
@_exception_logger
# 定义一个函数 all_reduce，用于在分布式环境中对张量进行全局归约操作
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces the tensor data across all machines in a way that all get the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat, device=device) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j], device='cuda:0') # Rank 0
        tensor([4.+4.j, 6.+6.j], device='cuda:1') # Rank 1

    """
    # 检查传入的张量是否符合要求
    _check_single_tensor(tensor, "tensor")
    # 如果当前进程组不在指定的群组中，则发出警告并返回空值
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce")
        return

    # 如果张量是复数类型，将其视为实数处理
    if tensor.is_complex():
        # 如果所选操作不支持复数类型的张量，则引发错误
        if not supports_complex(op):
            raise ValueError(f"all_reduce does not support {op} on complex tensors")
        tensor = torch.view_as_real(tensor)

    # 设置全局归约操作的选项
    opts = AllreduceOptions()
    opts.reduceOp = op
    # 如果未指定进程组，则使用默认进程组
    if group is None:
        group = _get_default_group()

    # 如果进程组存在于全局协调状态中，则将当前操作添加到协调状态中
    if group in _world.pg_coalesce_state.keys():
        # 在协调的上下文中，不执行单个操作，仅追加集体表示
        coll = _CollOp(all_reduce, tensor, None, op, None)
        _world.pg_coalesce_state[group].append(coll)
        # 如果是异步操作，则返回非法工作句柄，否则返回空值
        if async_op:
            return _IllegalWork()
        else:
            return None

    # 执行全局归约操作
    work = group.allreduce([tensor], opts)

    # 如果是异步操作，则返回工作句柄，否则等待操作完成
    if async_op:
        return work
    else:
        work.wait()
    "`torch.distributed.all_reduce_coalesced` will be deprecated. If you must "
    "use it, please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,


    # 发出关于将要被弃用的警告信息，指出 `torch.distributed.all_reduce_coalesced` 函数将会被弃用
    # 如果必须使用该函数，请稍后查阅我们的文档
    # 文档链接：https://pytorch.org/docs/main/distributed.html#collective-functions
    # 警告的类别为 FutureWarning
# 使用torch.distributed.ReduceOp枚举中的默认值ReduceOp.SUM，对输入的张量列表进行全局减少操作，确保所有进程最终获得相同的结果
def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=None, async_op=False):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.

    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Complex tensors are supported.

    Args:
        tensors (Union[List[Tensor], Tensor]): Input and output of the collective.
            The function operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    # 如果输入是单个张量，则转换为张量列表
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    # 检查张量列表的形状和数据类型是否符合要求
    _check_tensor_list(tensors, "tensor")
    # 确保所有张量具有相同的数据类型
    _ensure_all_tensors_same_dtype(tensors)
    # 如果当前进程组不在指定的分组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce_coalesced")
        return

    # 如果任何张量是复数，并且操作不支持复数，则抛出异常
    if any(t.is_complex() for t in tensors) and not supports_complex(op):
        raise ValueError(f"all_reduce does not support {op} on complex tensors")

    # 将复数张量转换为视图实数张量
    tensors = [t if not t.is_complex() else torch.view_as_real(t) for t in tensors]

    # 设置AllreduceCoalescedOptions选项
    opts = AllreduceCoalescedOptions()
    opts.reduceOp = op
    # 根据是否提供了分组信息选择默认进程组或指定的进程组进行全局减少操作
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allreduce_coalesced(tensors, opts)
    else:
        work = group.allreduce_coalesced(tensors, opts)

    # 如果选择异步操作，则返回异步操作的工作句柄
    if async_op:
        return work.get_future()
    else:
        # 等待操作完成
        work.wait()


# 通过异常记录装饰器包装的reduce函数，用于在所有机器上减少张量数据
def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.
    """
    Args:
        tensor (Tensor): Collective input and output tensor. Operates in-place.
        dst (int): Destination rank in the global process group (regardless of `group`).
        op (optional): Specifies the reduction operation from `torch.distributed.ReduceOp` enum.
        group (ProcessGroup, optional): Process group to perform the operation on. Defaults to the default process group if None.
        async_op (bool, optional): Whether the operation should be asynchronous.

    Returns:
        Async work handle if `async_op` is True.
        None if `async_op` is False or if the process is not part of the specified group.
    """
    # Check if the tensor is valid for collective operations
    _check_single_tensor(tensor, "tensor")
    
    # Check if the current process is not in the specified group
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce")
        return
    
    # Initialize options for the reduction operation
    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    
    # Determine the process group to use
    if group is None or group is GroupMember.WORLD:
        # Use the default process group if no specific group is provided
        default_pg = _get_default_group()
        work = default_pg.reduce([tensor], opts)
    else:
        # Adjust the root rank for the specified group
        group_dst_rank = get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce([tensor], opts)
    
    # Return the async work handle if async_op is True; otherwise, wait for completion
    if async_op:
        return work
    else:
        work.wait()
# 将 Python 对象序列化为字节流，并转换为 Torch 的 ByteStorage 对象
def _object_to_tensor(obj, device, group):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # 不要使用 torch.tensor 指定 dtype 替换 torch.ByteTensor 或 torch.LongTensor，
    # 否则会导致性能下降 100 倍。
    # 参见：https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    # 如果调试级别为 DETAIL，并且支持 NCCL，则记录哈希值和字节张量大小
    if get_debug_level() == DebugLevel.DETAIL and is_nccl_available():
        backend = get_backend(group)
        if backend == Backend.NCCL:
            hash = torch._C._distributed_c10d._hash_tensors([byte_tensor])
            logger.warning(
                "_object_to_tensor size: %s hash value: %s", byte_tensor.numel(), hash
            )
    # 创建表示本地大小的 LongTensor，以便在分布式环境中广播
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


# 将字节张量转换回 Python 对象
def _tensor_to_object(tensor, tensor_size, group):
    # 如果调试级别为 DETAIL，并且支持 NCCL，则记录哈希值和张量大小
    if get_debug_level() == DebugLevel.DETAIL and is_nccl_available():
        backend = get_backend(group)
        if backend == Backend.NCCL:
            hash = torch._C._distributed_c10d._hash_tensors([tensor])
            logger.warning(
                "_tensor_to_object size: %s hash value: %s", tensor.numel(), hash
            )
    # 将张量移回 CPU，并从中提取字节流，然后反序列化为 Python 对象
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


# 异常记录装饰器，用于记录异常信息
@_exception_logger
# 将对象从整个进程组中收集到列表中
def all_gather_object(object_list, obj, group=None):
    """
    Gathers picklable objects from the whole group into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.
    """
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_object")
        return
    检查当前进程是否在给定的分组中，如果不在，则发出警告并返回

    current_device = _get_pg_default_device(group)
    获取当前进程组的默认设备

    input_tensor, local_size = _object_to_tensor(obj, current_device, group)
    将输入对象转换为张量并获取其本地大小

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    获取组内的进程数量

    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    创建一个全零张量，用于存储各进程对象的大小信息

    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    创建一个列表，包含每个进程对象大小信息的张量视图

    # Allgather tensor sizes
    all_gather(object_size_list, local_size, group=group)
    使用 all_gather 将各进程的对象大小信息聚合到 object_size_list 中

    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    计算所有进程中对象大小的最大值

    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    调整输入张量的大小，以适应所有进程中最大的对象大小

    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )
    创建一个空张量，用于存储所有进程聚合后的输出数据

    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    创建一个列表，包含每个进程输出数据的张量视图

    all_gather(output_tensors, input_tensor, group=group)
    使用 all_gather 将所有进程的输入张量聚合到 output_tensors 中

    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size, group)
    将输出张量反序列化为对象列表中的对象
    """
# 带异常记录的装饰器，用于函数gather_object
@_exception_logger
# 从整个进程组中收集可picklable对象的函数
def gather_object(obj, object_gather_list=None, dst=0, group=None):
    """
    Gathers picklable objects from the whole group in a single process.

    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument). (default is 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
        ...     gather_objects[dist.get_rank()],
        ...     output if dist.get_rank() == 0 else None,
        ...     dst=0
        ... )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    # 如果当前进程不在指定的进程组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("gather_object")
        return

    # 确保object_gather_list在当前进程的rank正确地指定
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    # 获取默认的分组设备
    current_device = _get_pg_default_device(group)
    # 将对象转换为张量，并获取本地大小
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # 收集所有本地大小。这样我们可以找到最大大小，并在反序列化张量时正确索引到适当的大小。
    group_size = get_world_size(group=group)
    # 创建一个全零张量，用于存储对象大小
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    # 创建一个对象大小列表，每个元素是一个张量
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    
    # 所有节点广播张量大小。这里需要进行全收集，尽管这是一个收集操作，因为每个进程都需要广播相同（最大）大小的张量。
    all_gather(object_size_list, local_size, group=group)
    
    # 计算所有节点中最大的对象大小
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    
    # 调整输入张量的大小，使其达到所有节点中的最大大小
    input_tensor.resize_(max_object_size)
    
    # 如果当前节点不是目标节点，避免填充输出张量
    if my_rank == dst:
        # 创建一个空的合并输出张量
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=current_device
        )
        # 输出张量是合并输出张量的非重叠视图
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    
    # 所有节点使用相同大小的张量进行收集
    gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,  # type: ignore[possibly-undefined]
        dst=dst,
        group=group,
    )
    
    # 如果当前节点不是目标节点，则直接返回
    if my_rank != dst:
        return
    
    # 对于每个输出张量，将其转换为相应的对象
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size, group)
    # 带异常日志的装饰器，用于捕获函数内部可能出现的异常并记录
    @_exception_logger
    # 向目标进程发送包含在object_list中的可序列化对象，同步执行
    def send_object_list(object_list, dst, group=None, device=None):
        """
        Sends picklable objects in ``object_list`` synchronously.

        Similar to :func:`send`, but Python objects can be passed in.
        Note that all objects in ``object_list`` must be picklable in order to be
        sent.

        Args:
            object_list (List[Any]): List of input objects to sent.
                Each object must be picklable. Receiver must provide lists of equal sizes.
            dst (int): Destination rank to send ``object_list`` to.
                Destination rank is based on global process group (regardless of ``group`` argument)
            group: (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used. Default is ``None``.
            device (``torch.device``, optional): If not None, the objects are
                serialized and converted to tensors which are moved to the
                ``device`` before sending. Default is ``None``.

        Returns:
            ``None``.

        .. note:: For NCCL-based process groups, internal tensor representations
            of objects must be moved to the GPU device before communication takes
            place. In this case, the device used is given by
            ``torch.cuda.current_device()`` and it is the user's responsibility to
            ensure that this is set so that each rank has an individual GPU, via
            ``torch.cuda.set_device()``.

        .. warning::
            :func:`send_object_list` uses ``pickle`` module implicitly, which
            is known to be insecure. It is possible to construct malicious pickle
            data which will execute arbitrary code during unpickling. Only call this
            function with data you trust.

        .. warning::
            Calling :func:`send_object_list` with GPU tensors is not well supported
            and inefficient as it incurs GPU -> CPU transfer since tensors would be
            pickled. Please consider using :func:`send` instead.

        Example::
            >>> # xdoctest: +SKIP("need process group init")
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> # Assumes backend is not NCCL
            >>> device = torch.device("cpu")
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 2.
            >>>     objects = ["foo", 12, {1: 2}] # any picklable object
            >>>     dist.send_object_list(objects, dst=1, device=device)
            >>> else:
            >>>     objects = [None, None, None]
            >>>     dist.recv_object_list(objects, src=0, device=device)
            >>> objects
            ['foo', 12, {1: 2}]
        """
        # 如果目标进程的rank与当前进程的rank相同，抛出数值错误异常
        if get_rank() == dst:
            raise ValueError(
                "Invalid destination rank: destination rank should not be the same as "
                "the rank of the current process."
            )

        # 如果当前进程不在指定的进程组中，则发出警告并返回
        if _rank_not_in_group(group):
            _warn_not_in_group("send_object_list")
            return

        # 当前设备的选择。
    # 为了保持向后兼容性，如果未指定 ``device``，则默认为 ``None``
    # 在这种情况下，我们运行当前设备选择逻辑，即：
    # 如果后端是 NCCL，则 ``current_device`` 是 CUDA，否则是 CPU 设备。
    # 如果 ``device`` 不是 ``None``，则将要发送到该设备的大小和对象张量移动到该设备上。
    current_device = device or _get_pg_default_device(group)

    # 将 object_list 中的每个对象序列化为张量，并在当前设备上处理
    tensor_list, size_list = zip(
        *[_object_to_tensor(obj, current_device, group) for obj in object_list]
    )

    # 将所有对象的大小信息拼接成一个张量
    object_sizes_tensor = torch.cat(size_list)

    # 发送对象大小信息到目标节点
    send(object_sizes_tensor, dst=dst, group=group)

    # 拼接并发送序列化后的对象张量
    # 注意: 如果 tensor_list 只有一个元素，torch.cat 将在当前设备上执行额外的内存拷贝，可以跳过这一步。
    if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
        object_tensor = tensor_list[0]
    else:
        object_tensor = torch.cat(tensor_list)

    send(object_tensor, dst=dst, group=group)
# 带异常日志记录的装饰器，用于函数recv_object_list
@_exception_logger
# 接收一个对象列表的函数，实现同步接收picklable对象
def recv_object_list(object_list, src=None, group=None, device=None):
    """
    Receives picklable objects in ``object_list`` synchronously.

    Similar to :func:`recv`, but can receive Python objects.

    Args:
        object_list (List[Any]): List of objects to receive into.
            Must provide a list of sizes equal to the size of the list being sent.
        src (int, optional): Source rank from which to recv ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
            Will receive from any rank if set to None. Default is ``None``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, receives on this device.
            Default is ``None``.

    Returns:
        Sender rank. -1 if rank is not part of the group. If rank is part of the group,
        ``object_list`` will contain the sent objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`recv_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`recv_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`recv` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     dist.send_object_list(objects, dst=1, device=device)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     dist.recv_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    # 如果当前进程不在指定的进程组中，则警告并返回-1
    if _rank_not_in_group(group):
        _warn_not_in_group("recv_object_list")
        return -1

    # 当前设备选择逻辑。
    # 为了向后兼容性，``device`` 默认为 ``None``
    # 在这种情况下，我们继续当前设备选择逻辑，即：
    # ``current_device`` 是 CUDA 如果后端是 NCCL，否则是 CPU 设备。如果 ``device`` 不为 ``None``，则将尺寸和对象张量移动到该设备上。
    current_device = device or _get_pg_default_device(group)

    # 创建一个空的张量来存储对象的大小，张量的设备是 ``current_device``
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )

    # 接收对象的大小信息
    rank_sizes = recv(object_sizes_tensor, src=src, group=group)

    # 创建一个空的张量来接收序列化后的对象数据
    object_tensor = torch.empty(
        torch.sum(object_sizes_tensor).item(),
        dtype=torch.uint8,
        device=current_device,
    )

    # 接收对象的数据
    rank_objects = recv(object_tensor, src=src, group=group)

    # 检查接收的对象大小与对象数据的返回值是否匹配
    assert (
        rank_sizes == rank_objects
    ), "对象大小和对象数据返回的排名不匹配."

    # 使用存储的大小信息反序列化对象
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        # 获取当前对象在 object_tensor 中的视图
        obj_view = object_tensor[offset : offset + obj_size]
        # 将视图类型转换为 uint8
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        # 将反序列化后的对象存储在 object_list 中
        object_list[i] = _tensor_to_object(obj_view, obj_size, group)

    # 返回对象的排名信息
    return rank_objects
# 为了方便异常记录，在函数调用时添加异常日志记录的装饰器
@_exception_logger
# 广播给定对象列表到整个组中
def broadcast_object_list(object_list, src=0, group=None, device=None):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    # 如果当前排名不在指定的组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast_object_list")
        return

    # 当前设备选择。
    # 为了保持向后兼容性，如果 ``device`` 默认为 ``None``，
    # 则运行当前设备选择逻辑，即如果后端是 NCCL，则 ``current_device`` 是 CUDA，否则是 CPU 设备。
    # 如果 ``device`` 不是 ``None``，则将大小和对象张量广播到指定设备。
    current_device = device or _get_pg_default_device(group)
    my_rank = get_rank()

    # 将 object_list 中的对象序列化为张量（tensor）并在源排名（src）上执行
    if my_rank == src:
        # 使用列表推导式将 object_list 中的每个对象转换为张量，并获取其大小信息
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device, group) for obj in object_list]
        )
        # 将所有对象的大小信息拼接成一个张量
        object_sizes_tensor = torch.cat(size_list)
    else:
        # 如果当前排名不是源排名，则创建一个空张量来存储对象大小信息
        object_sizes_tensor = torch.empty(
            len(object_list), dtype=torch.long, device=current_device
        )

    # 将对象大小信息进行广播
    broadcast(object_sizes_tensor, src=src, group=group)

    # 拼接并广播序列化后的对象张量
    # 注意：如果 tensor_list 只有一个元素，torch.cat 会在当前设备上做一次额外的内存拷贝，可以跳过这个拷贝操作。
    if my_rank == src:
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)
    else:
        # 如果当前排名不是源排名，则创建一个空张量来接收广播的对象张量
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device,
        )

    # 对象张量进行广播
    broadcast(object_tensor, src=src, group=group)

    # 使用存储的大小信息反序列化对象
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            # 从 object_tensor 中取出当前对象的视图，并将其类型转换为 uint8
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            # 将反序列化后的对象重新存入 object_list 中
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)
# 在异常处理日志装饰器下定义函数，用于将可序列化对象从 scatter_object_input_list 散布到整个组中
@_exception_logger
def scatter_object_list(
    scatter_object_output_list, scatter_object_input_list, src=0, group=None
):
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole group.

    Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any]): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        src (int): Source rank from which to scatter ``scatter_object_input_list``.
            Source rank is based on global process group (regardless of ``group`` argument).
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`scatter_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`scatter` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """
    # 如果当前进程不在指定的进程组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter_object_list")
        return
    # 检查 scatter_object_output_list 是否为列表且至少包含一个元素，若不是则引发 ValueError
    if (
        not isinstance(scatter_object_output_list, list)
        or len(scatter_object_output_list) < 1
    ):
        raise ValueError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    # 获取当前进程的排名
    my_rank = get_rank()
    # 获取指定分组的默认设备
    pg_device = _get_pg_default_device(group)

    # 如果当前进程是发送方（src），将 scatter_object_input_list 中的对象转换为张量列表
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[
                _object_to_tensor(obj, pg_device, group)
                for obj in scatter_object_input_list
            ]
        )
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

    # 如果当前进程是发送方（src），广播最大张量大小，以确保所有进程调用 scatter() 时使用相同大小的张量
    if my_rank == src:
        # 计算张量列表中的最大大小
        max_tensor_size = max(tensor_sizes)  # type: ignore[possibly-undefined]
        # 调整张量列表中的每个张量大小为最大张量大小
        for tensor in tensor_list:  # type: ignore[possibly-undefined]
            tensor.resize_(max_tensor_size)
    else:
        # 如果当前进程不是发送方（src），则初始化最大张量大小为零
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    # 使用广播函数将最大张量大小发送给所有进程
    broadcast(max_tensor_size, src=src, group=group)

    # 创建一个空张量 output_tensor，用于接收分散的序列化对象
    output_tensor = torch.empty(
        max_tensor_size.item(), dtype=torch.uint8, device=pg_device
    )
    # 调用 scatter() 函数，将序列化的对象分散到各个进程
    scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,  # type: ignore[possibly-undefined]
        src=src,
        group=group,
    )

    # 创建一个张量 obj_tensor_size，用于接收每个对象的大小信息，以便在反序列化时调整张量大小
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    # 调用 scatter() 函数，将每个对象的大小信息分散到各个进程
    scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,  # type: ignore[possibly-undefined]
        src=src,
        group=group,
    )

    # 将 output_tensor 反序列化为对象，并存储到 scatter_object_output_list 中的第一个元素
    scatter_object_output_list[0] = _tensor_to_object(
        output_tensor, obj_tensor_size, group
    )
# 定义一个装饰器函数，用于捕获并记录异常信息
@_exception_logger
# 定义函数all_gather，用于从整个进程组中收集张量到列表中
def all_gather(tensor_list, tensor, group=None, async_op=False):
    """
    Gathers tensors from the whole group in a list.

    Complex and uneven sized tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
            Uneven sized tensors are supported.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor_list = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(2)]
        >>> tensor_list
        [tensor([0, 0], device='cuda:0'), tensor([0, 0], device='cuda:0')] # Rank 0
        [tensor([0, 0], device='cuda:0'), tensor([0, 0], device='cuda:1')] # Rank 1
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2], device='cuda:0'), tensor([3, 4], device='cuda:0')] # Rank 0
        [tensor([1, 2], device='cuda:1'), tensor([3, 4], device='cuda:1')] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.cfloat, device=device) for _ in range(2)]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j], device='cuda:0'), tensor([0.+0.j, 0.+0.j], device='cuda:0')] # Rank 0
        [tensor([0.+0.j, 0.+0.j], device='cuda:1'), tensor([0.+0.j, 0.+0.j], device='cuda:1')] # Rank 1
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat, device=device) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j], device='cuda:0'), tensor([3.+3.j, 4.+4.j], device='cuda:0')] # Rank 0
        [tensor([1.+1.j, 2.+2.j], device='cuda:1'), tensor([3.+3.j, 4.+4.j], device='cuda:1')] # Rank 1

    """
    # 检查tensor_list是否为张量列表
    _check_tensor_list(tensor_list, "tensor_list")
    # 检查tensor是否为单个张量
    _check_single_tensor(tensor, "tensor")
    # 确保所有张量具有相同的数据类型
    _ensure_all_tensors_same_dtype(tensor_list, tensor)
    # 如果当前进程不在指定的进程组中，警告并返回None
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather")
        return
    # 对于 tensor_list 中的每个张量 t，如果 t 是复数类型，则使用 torch.view_as_real 转换为实部张量，并将其放入列表中
    tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in tensor_list
    ]
    # 如果 tensor 是复数类型，则使用 torch.view_as_real 转换为实部张量
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    # 如果 group 为 None，则获取默认的处理组
    if group is None:
        default_pg = _get_default_group()
        # 对于默认处理组 default_pg，执行 allgather 操作，将 tensor_list 和 tensor 广播到所有进程
        work = default_pg.allgather([tensor_list], [tensor])
    else:
        # 对于给定的 group，执行 allgather 操作，将 tensor_list 和 tensor 广播到所有进程
        work = group.allgather([tensor_list], [tensor])

    # 如果 async_op 为 True，则直接返回 work 对象
    if async_op:
        return work
    else:
        # 如果 async_op 为 False，则等待 work 完成
        work.wait()
# 定义一个装饰器函数，用于记录异常信息
@_exception_logger
# 定义函数 all_gather_into_tensor，用于从各个进程中收集张量并放入单个输出张量中
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    """
    Gather tensors from all ranks and put them in a single output tensor.

    This function requires all tensors to be the same size on each process.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            Different from the ``all_gather`` API, the input tensors in this
            API must have the same size across all ranks.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f'cuda:{rank}')
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device='cuda:0') # Rank 0
        tensor([1, 2, 3, 4], device='cuda:1') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device='cuda:0') # Rank 0
        tensor([[1, 2],
                [3, 4]], device='cuda:1') # Rank 1

    .. warning::
        The Gloo backend does not support this API.

    """
    # 检查输入的张量是否为单个张量
    _check_single_tensor(input_tensor, "input_tensor")
    # 检查输出的张量是否为单个张量
    _check_single_tensor(output_tensor, "output_tensor")
    # 如果当前进程不在指定的进程组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_into_tensor")
        return

    # 如果输出张量是复杂类型，则将其视为实数张量
    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    # 如果输入张量是复数类型，则将其转换为实部张量
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )

    # 创建Allgather的选项对象
    opts = AllgatherOptions()
    opts.asyncOp = async_op  # 设置异步操作选项

    # 如果未指定通信组，则使用默认组
    group = group or _get_default_group()

    # 如果当前通信组在全局的pg_coalesce_state字典中
    if group in _world.pg_coalesce_state.keys():
        # 在合并上下文中，不执行单个操作，只追加集体表示
        coll = _CollOp(all_gather_into_tensor, input_tensor, output_tensor)
        _world.pg_coalesce_state[group].append(coll)
        # 如果是异步操作，返回非法工作对象_IllegalWork
        if async_op:
            return _IllegalWork()
        else:
            return None

    # 在指定通信组上执行基本的allgather操作，并返回工作对象
    work = group._allgather_base(output_tensor, input_tensor, opts)

    # 如果是异步操作，直接返回工作对象
    if async_op:
        return work
    else:
        # 如果是同步操作，等待工作完成
        work.wait()
# 应用异常日志装饰器和过时警告装饰器到函数 `_all_gather_base` 上
@_exception_logger
@deprecated(
    "`torch.distributed._all_gather_base` is a private function and will be deprecated. "
    "Please use `torch.distributed.all_gather_into_tensor` instead.",
    category=FutureWarning,
)
def _all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. warning::
        `_all_gather_base` is a private function. Users should use
        `all_gather_into_tensor` instead.

    """
    # 调用 `all_gather_into_tensor` 函数，将输出和输入张量从所有进程中聚合到输出张量中
    return all_gather_into_tensor(output_tensor, input_tensor, group, async_op)


# 应用异常日志装饰器和过时警告装饰器到函数 `all_gather_coalesced` 上
@_exception_logger
@deprecated(
    "`torch.distributed.all_gather_coalesced` will be deprecated. If you must use it, "
    "please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,
)
def all_gather_coalesced(
    output_tensor_lists, input_tensor_list, group=None, async_op=False
):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Complex tensors are supported.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    # 调用 `all_gather_coalesced` 函数，以合并方式从整个进程组中收集输入张量列表
    return all_gather_coalesced(output_tensor_lists, input_tensor_list, group, async_op)
    # 我们只在这里对与 C++ 参数的基本兼容性进行检查，C++ 代码将进行形状和类型检查。
    if _rank_not_in_group(group):
        # 如果当前进程不在指定的进程组中，发出警告并返回
        _warn_not_in_group("all_gather_coalesced")
        return
    
    # 检查输入张量列表的形状和类型是否符合预期
    _check_tensor_list(input_tensor_list, "input_tensor_list")
    # 确保所有输入张量具有相同的数据类型
    _ensure_all_tensors_same_dtype(input_tensor_list)
    
    # 检查输出张量列表是否为列表类型
    if not isinstance(output_tensor_lists, list):
        # 如果输出张量列表不是列表类型，抛出类型错误异常
        raise TypeError(
            "Invalid function argument: output_tensor_lists should be a list"
        )
    
    # 对每个输出张量列表进行形状和类型检查
    for output_tensor_list in output_tensor_lists:
        _check_tensor_list(output_tensor_list, "output_tensor_lists")
        _ensure_all_tensors_same_dtype(output_tensor_list)
    
    # 将复数张量转换为实数张量
    output_tensor_lists = [
        [t if not t.is_complex() else torch.view_as_real(t) for t in l]
        for l in output_tensor_lists
    ]
    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list
    ]
    
    # 如果未指定进程组，则获取默认进程组，并执行 allgather_coalesced 操作
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.allgather_coalesced(output_tensor_lists, input_tensor_list)
    else:
        # 使用指定的进程组执行 allgather_coalesced 操作
        work = group.allgather_coalesced(output_tensor_lists, input_tensor_list)
    
    # 如果设置了异步操作标志，则返回操作的 future 对象
    if async_op:
        return work.get_future()
    else:
        # 等待操作完成
        work.wait()
# 验证输出列表是否符合排名要求
def _validate_output_list_for_rank(my_rank, dst, gather_list):
    # 如果目标排名与当前排名相同
    if dst == my_rank:
        # 如果 gather_list 为空，则抛出数值错误
        if not gather_list:
            raise ValueError(
                "Argument ``gather_list`` must be specified on destination rank."
            )
    # 如果目标排名与当前排名不同
    elif gather_list:
        # 如果 gather_list 不为空，则抛出数值错误
        raise ValueError(
            "Argument ``gather_list`` must NOT be specified "
            "on non-destination ranks."
        )


# 异常日志记录装饰器
@_exception_logger
def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    """
    Gathers a list of tensors in a single process.

    This function requires all tensors to be the same size on each process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately,
            same-sized tensors to use for gathered data
            (default is None, must be specified on the destination rank)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument). (default is 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    # 检查输入张量是否单一
    _check_single_tensor(tensor, "tensor")

    # 如果 gather_list 已指定，则检查张量列表
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        # 如果未指定 gather_list，则将其置为空列表
        gather_list = []
    # 确保所有张量具有相同的数据类型
    _ensure_all_tensors_same_dtype(tensor, gather_list)

    # 如果当前进程不在指定的进程组中
    if _rank_not_in_group(group):
        # 发出警告并返回
        _warn_not_in_group("gather")
        return

    # 获取当前进程的排名
    my_rank = get_rank()
    # 验证输出列表的有效性
    _validate_output_list_for_rank(my_rank, dst, gather_list)
    # 根据目标排名确定输出张量
    output_tensors = [gather_list] if dst == my_rank else []
    input_tensors = [tensor]

    # 设置 GatherOptions 选项
    opts = GatherOptions()
    opts.rootRank = dst

    # 如果未指定进程组或进程组为 WORLD
    if group is None or group is GroupMember.WORLD:
        # 获取默认进程组
        default_pg = _get_default_group()
        # 执行全局收集操作
        work = default_pg.gather(output_tensors, input_tensors, opts)
    else:
        # 获取指定进程组中目标排名
        group_dst_rank = get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        # 在指定进程组中执行收集操作
        work = group.gather(output_tensors, input_tensors, opts)

    # 如果是异步操作，则返回工作句柄
    if async_op:
        return work
    else:
        # 否则等待操作完成
        work.wait()


# 异常日志记录装饰器
@_exception_logger
def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Complex tensors are supported.
    
    Args:
        tensor (Tensor): Input tensor.
        scatter_list (list[Tensor], optional): List of appropriately,
            same-sized tensors containing data to scatter
            (default is None, must be specified on the source rank)
        src (int, optional): Source rank on global process group (regardless of ``group`` argument). (default is 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank on global process group (regardless of ``group`` argument).
            Default is 0
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in scatter_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> tensor_size = 2
        >>> t_ones = torch.ones(tensor_size)
        >>> t_fives = torch.ones(tensor_size) * 5
        >>> output_tensor = torch.zeros(tensor_size)
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     # Only tensors, all of which must be the same size.
        >>>     scatter_list = [t_ones, t_fives]
        >>> else:
        >>>     scatter_list = None
        >>> dist.scatter(output_tensor, scatter_list, src=0)
        >>> # Rank i gets scatter_list[i]. For example, on rank 1:
        >>> output_tensor
        tensor([5., 5.])

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []  # 如果scatter_list未指定，则设为空列表

    _ensure_all_tensors_same_dtype(tensor, scatter_list)  # 确保tensor和scatter_list中所有张量的数据类型相同

    if _rank_not_in_group(group):  # 如果当前进程不在指定的进程组中
        _warn_not_in_group("scatter")  # 输出警告信息
        return  # 返回空值，不执行scatter操作

    # 处理复数张量，将复数张量转换为实部张量
    scatter_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list
    ]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    my_rank = get_rank()  # 获取当前进程的排名

    if src == my_rank:  # 如果当前进程是源进程
        if not scatter_list:  # 如果scatter_list为空
            raise ValueError(
                "Argument ``scatter_list`` must be specified on source rank."
            )
        input_tensors = [scatter_list]  # 将scatter_list作为输入张量列表
        output_tensors = [tensor]  # 将tensor作为输出张量列表
    else:
        if scatter_list:  # 如果scatter_list不为空
            raise ValueError(
                "Argument ``scatter_list`` must NOT be specified "
                "on non-source ranks."
            )
        input_tensors = []  # 非源进程时输入张量列表为空
        output_tensors = [tensor]  # 将tensor作为输出张量列表

    opts = ScatterOptions()
    opts.rootRank = src  # 设置scatter的根节点排名为src
    opts.asyncOp = async_op  # 设置是否使用异步操作的选项

    if group is None or group is GroupMember.WORLD:  # 如果未指定进程组或者进程组为WORLD
        default_pg = _get_default_group()  # 获取默认的进程组
        work = default_pg.scatter(output_tensors, input_tensors, opts)  # 执行scatter操作
    # 如果不是异步操作，获取组内源的排名并设置为根排名
    else:
        group_src_rank = get_group_rank(group, src)
        opts.rootRank = group_src_rank
        # 将输出张量和输入张量分散到组中的所有进程
        work = group.scatter(output_tensors, input_tensors, opts)

    # 如果是异步操作，直接返回工作对象
    if async_op:
        return work
    # 否则，等待工作完成
    else:
        work.wait()
# 定义一个装饰器函数，用于记录异常并打印日志
@_exception_logger
# 定义函数 reduce_scatter，将一个张量列表进行归约后再分散到组内所有进程
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    # 检查输出张量是否为单个张量
    _check_single_tensor(output, "output")
    # 检查输入张量列表是否合法
    _check_tensor_list(input_list, "input_list")
    # 确保所有输入张量的数据类型相同
    _ensure_all_tensors_same_dtype(output, input_list)
    # 如果当前进程不在指定的进程组内，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter")
        return

    # 创建 ReduceScatterOptions 对象
    opts = ReduceScatterOptions()
    opts.reduceOp = op

    # 如果未指定进程组，则使用默认进程组进行归约分散操作
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.reduce_scatter([output], [input_list], opts)
    else:
        # 否则，使用指定的进程组进行归约分散操作
        work = group.reduce_scatter([output], [input_list], opts)

    # 如果设置为异步操作，则返回异步工作句柄
    if async_op:
        return work
    else:
        # 否则，等待归约分散操作完成
        work.wait()


# 定义一个装饰器函数，用于记录异常并打印日志
@_exception_logger
# 定义函数 reduce_scatter_tensor，将一个张量归约后再分散到组内所有进程
def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a tensor to all ranks in a group.

    Args:
        output (Tensor): Output tensor. It should have the same size across all
            ranks.
        input (Tensor): Input tensor to be reduced and scattered. Its size
            should be output tensor size times the world size. The input tensor
            can have one of the following shapes:
            (i) a concatenation of the output tensors along the primary
            dimension, or
            (ii) a stack of the output tensors along the primary dimension.
            For definition of "concatenation", see ``torch.cat()``.
            For definition of "stack", see ``torch.stack()``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    # 如果当前进程不在指定的进程组内，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter")
        return

    # 创建 ReduceScatterOptions 对象
    opts = ReduceScatterOptions()
    opts.reduceOp = op

    # 如果未指定进程组，则使用默认进程组进行归约分散操作
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.reduce_scatter([output], [input], opts)
    else:
        # 否则，使用指定的进程组进行归约分散操作
        work = group.reduce_scatter([output], [input], opts)

    # 如果设置为异步操作，则返回异步工作句柄
    if async_op:
        return work
    else:
        # 否则，等待归约分散操作完成
        work.wait()
    # 检查输出张量和输入张量是否为单个张量，否则引发异常
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    
    # 如果当前进程组中的排名不在指定的组中，发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter_tensor")
        return
    
    # 创建ReduceScatterOptions对象，并设置操作类型和异步操作标志
    opts = ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    
    # 如果未提供组，则使用默认组
    group = group or _get_default_group()
    
    # 检查是否在合并上下文中
    # 如果是，则不执行单个操作，只是追加一个集体表示
    if group in _world.pg_coalesce_state.keys():
        # 创建一个_CollOp对象，表示reduce_scatter_tensor的集体操作
        coll = _CollOp(reduce_scatter_tensor, input, output, op, None)
        # 将集体操作添加到合并状态中
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            # 如果是异步操作，则返回非法工作对象_IllegalWork
            return _IllegalWork()
        else:
            # 否则返回空值
            return None
    
    # 对组执行基本的reduce_scatter操作，返回工作对象
    work = group._reduce_scatter_base(output, input, opts)
    
    if async_op:
        # 如果是异步操作，则返回工作对象
        return work
    else:
        # 否则等待工作完成
        work.wait()
# 标记函数为过时，提醒用户使用替代函数`torch.distributed.reduce_scatter_tensor`
@deprecated(
    "`torch.distributed._reduce_scatter_base` is a private function and will be deprecated. "
    "Please use `torch.distributed.reduce_scatter_tensor` instead.",
    category=FutureWarning,
)
def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input (Tensor): Input tensor that is of size output tensor size times world size
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `_reduce_scatter_base` is a private function. Users should use
        `reduce_scatter_tensor` instead.

    """
    # 调用替代函数`reduce_scatter_tensor`进行实际的张量减少和散播操作
    return reduce_scatter_tensor(output, input, op, group, async_op)


# 使用异常日志装饰器装饰函数
@_exception_logger
def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    """
    Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group
    and returned as a single output tensor.

    Complex tensors are supported.

    Args:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    """
    # 如果进程组中不包含当前进程，发出警告并返回空
    if _rank_not_in_group(group):
        _warn_not_in_group("all_to_all_single")
        return

    # 创建`AllToAllOptions`对象
    opts = AllToAllOptions()
    # 检查输出张量是否为单个张量
    _check_single_tensor(output, "output")
    # 检查输入张量是否为单个张量
    _check_single_tensor(input, "input")
    # 确保所有张量具有相同的数据类型
    _ensure_all_tensors_same_dtype(output, input)

    # 如果输入张量是复数类型，将其视为实数处理
    if input.is_complex():
        input = torch.view_as_real(input)
    # 如果输出张量是复数类型，将其视为实数处理
    if output.is_complex():
        output = torch.view_as_real(output)

    # 如果未提供输出分割大小，则设为空列表
    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    # 如果未提供输入分割大小，则设为空列表
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes
    # 如果未指定通信组（group），则获取默认通信组
    if group is None:
        # 调用 _get_default_group() 函数获取默认通信组对象
        default_pg = _get_default_group()
        # 使用默认通信组对象执行 alltoall_base 操作，并获取返回的工作对象
        work = default_pg.alltoall_base(
            output, input, output_split_sizes, input_split_sizes, opts
        )
    else:
        # 使用指定的通信组对象执行 alltoall_base 操作，并获取返回的工作对象
        work = group.alltoall_base(
            output, input, output_split_sizes, input_split_sizes, opts
        )

    # 如果设置了异步操作标志 async_op，则直接返回工作对象
    if async_op:
        return work
    else:
        # 如果未设置异步操作标志，则等待工作对象的完成
        work.wait()
@_exception_logger
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    """
    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Complex tensors are supported.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    """
    # 如果当前进程不在指定的进程组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("all_to_all")
        return

    # 创建用于 all_to_all 操作的选项对象
    opts = AllToAllOptions()
    # 检查输出和输入张量列表是否有效
    _check_tensor_list(output_tensor_list, "output_tensor_list")
    _check_tensor_list(input_tensor_list, "input_tensor_list")
    # 确保输出和输入张量列表中的张量数据类型相同
    _ensure_all_tensors_same_dtype(output_tensor_list, input_tensor_list)

    # 如果输入张量是复数张量，则将其转换为实数张量
    input_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in input_tensor_list
    ]
    # 如果输出张量是复数张量，则将其转换为实数张量
    output_tensor_list = [
        t if not t.is_complex() else torch.view_as_real(t) for t in output_tensor_list
    ]

    # 如果未指定进程组，则使用默认进程组
    if group is None:
        default_pg = _get_default_group()
        # 执行 alltoall 操作并获取异步操作句柄
        work = default_pg.alltoall(output_tensor_list, input_tensor_list, opts)
    else:
        # 执行指定进程组的 alltoall 操作并获取异步操作句柄
        work = group.alltoall(output_tensor_list, input_tensor_list, opts)

    # 如果设置为异步操作，则返回异步操作句柄
    if async_op:
        return work
    else:
        # 否则等待操作完成
        work.wait()


@_exception_logger
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
    """
    Synchronize all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        device_ids ([int], optional): List of device/GPU ids.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: `ProcessGroupNCCL` now relies on stream synchronization instead of
              device synchronization to block the CPU. Thus, please do not assume that
              `barrier()` would perform a device synchronization.
    """
    # 如果当前进程不在指定的进程组中，则发出警告并返回
    if _rank_not_in_group(group):
        _warn_not_in_group("barrier")
        return

    # 创建用于 barrier 操作的选项对象
    opts = BarrierOptions()
    # 设置设备为默认进程组的默认设备
    opts.device = _get_pg_default_device(group)
    # 如果传入了设备ID列表，则设置opts对象的device_ids属性为传入的设备ID列表
    if device_ids is not None:
        # 检查device_ids是否为列表类型
        if isinstance(device_ids, list):
            opts.device_ids = device_ids
        else:
            # 如果device_ids不是列表类型，则抛出类型错误异常
            raise TypeError(
                "Invalid function argument: device_ids type should be List[int]"
            )

    # 如果未提供group参数，则获取默认进程组，调用其barrier方法并返回结果到work变量
    if group is None:
        default_pg = _get_default_group()
        work = default_pg.barrier(opts=opts)
    else:
        # 如果提供了group参数，则调用该group的barrier方法并返回结果到work变量
        work = group.barrier(opts=opts)

    # 如果async_op为True，则直接返回work
    if async_op:
        return work
    else:
        # 如果async_op为False，则等待work完成
        work.wait()
# 定义了一个函数 monitored_barrier，用于实现类似于 torch.distributed.barrier 的进程同步操作，但支持可配置的超时时间。
# 该函数能够报告未能在指定超时时间内通过障碍的进程编号。
# 对于非零进程编号，将会阻塞直到从进程编号为 0 的进程收到发送/接收处理。
# 进程编号为 0 的进程将会阻塞直到所有其他进程的发送/接收处理完成，并将报告超时未响应的进程编号。
# 需要注意的是，如果一个进程未能到达 monitored_barrier（例如由于 hang），所有其他进程也会失败在 monitored_barrier 中。

# 该集体操作会阻塞组内的所有进程/进程编号，直到整个组成功退出函数，因此在调试和同步时非常有用。
# 但是它可能会对性能产生影响，应仅用于调试或需要主机端完全同步点的场景。
# 用于调试目的时，可以在应用程序的集体调用之前插入此障碍以检查是否存在任何进程不同步的情况。

# 注意：该集体操作仅支持 GLOO 后端。

# 定义函数 monitored_barrier，接受如下参数：
# - group（ProcessGroup，可选）：要处理的进程组。如果为 None，则使用默认进程组。
# - timeout（datetime.timedelta，可选）：monitored_barrier 的超时时间。如果为 None，则使用默认进程组超时。
# - wait_all_ranks（bool，可选）：是否收集所有失败的进程编号。默认情况下为 False，monitored_barrier 在进程编号为 0 时会在遇到第一个失败的进程编号时抛出异常以快速失败。
#   如果设置 wait_all_ranks=True，则 monitored_barrier 将收集所有失败的进程编号，并抛出一个包含所有失败进程信息的错误。

# 返回 None。

# 示例：
# >>> # xdoctest: +SKIP("need process group init")
# >>> # 注意：省略了每个进程的进程组初始化。
# >>> import torch.distributed as dist
# >>> if dist.get_rank() != 1:
# >>>     dist.monitored_barrier() # 抛出异常指示进程编号为 1 的进程未调用 monitored_barrier。
# >>> # 使用 wait_all_ranks=True 的示例
# >>> if dist.get_rank() == 0:
# >>>     dist.monitored_barrier(wait_all_ranks=True) # 抛出异常指示进程编号为 1、2、...、world_size - 1 的进程未调用 monitored_barrier。

def monitored_barrier(group=GroupMember.WORLD, timeout=None, wait_all_ranks=False):
    # 在使用进程组之前，需要先检查 rank 是否不在 group 中，否则会引发 "Invalid process group" 错误。
    if _rank_not_in_group(group):
        _warn_not_in_group("monitored_barrier")
        return

    # 检查使用的后端是否为 GLOO，如果不是则抛出 ValueError 异常。
    if get_backend(group) != Backend.GLOO:
        raise ValueError("monitored_barrier is only implemented for GLOO backend.")
    # 如果 timeout 参数为 None，则从 group 参数获取默认的超时时间
    if timeout is None:
        timeout = _get_default_timeout(get_backend(group))
    # 如果 timeout 参数为 float 类型
    elif isinstance(timeout, float):
        # 发出警告，提醒用户应该使用 timedelta 格式来指定超时时间
        warnings.warn(
            "Please specify timeout arg as a timedelta. "
            f"Converting current value of {timeout} assuming it represents seconds",
        )
        # 将 timeout 转换为 timedelta 类型，假设其表示的是秒数
        timeout = timedelta(seconds=timeout)

    # 检查 timeout 的有效性
    _check_valid_timeout(timeout)

    # 如果 group 参数为 None，则使用默认的 group
    group_to_use = _get_default_group() if group is None else group
    # 调用 group_to_use 对象的 monitored_barrier 方法，执行监视的屏障操作
    return group_to_use.monitored_barrier(timeout, wait_all_ranks=wait_all_ranks)
# 创建一个包装的进程组的包装器函数
def _create_process_group_wrapper(
    wrapped_pg: torch._C._distributed_c10d.Backend,
    store_prefix: str,
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta = default_pg_timeout,
):
    # 确保 GLOO 后端可用，否则抛出异常
    assert _GLOO_AVAILABLE, "ProcessGroupWrapper unsupported without GLOO backend."

    # 创建一个帮助进程组的专用前缀存储
    prefix = f"{PG_WRAPPER_STORE_PREFIX}:{store_prefix}"
    store = PrefixStore(prefix, store)
    # 使用 GLOO 后端创建一个帮助进程组
    helper_pg = ProcessGroupGloo(store, rank, world_size, timeout=timeout)
    # 使用 ProcessGroupWrapper 包装传入的进程组对象
    wrapped_pg = _ProcessGroupWrapper(wrapped_pg, helper_pg)
    return wrapped_pg


# 用于确定性地对一组排名进行哈希的辅助函数
def _hash_ranks(ranks: List[int]):
    return hashlib.sha1(bytes("_".join(map(str, ranks)), "utf-8")).hexdigest()


# 接受一组排名，并计算一个整数颜色
def _process_group_color(ranks: List[int]) -> int:
    # 将哈希转换为整数，但通过位移来避免负数
    return int(_hash_ranks(ranks), 16) % (sys.maxsize >> 1)


def _process_group_name(ranks, use_hashed_name):
    global _world
    if use_hashed_name:
        # 如果使用哈希名称，生成基于排名的哈希名，并确保唯一性
        pg_name = _hash_ranks(ranks)
        while pg_name in _world.pg_names.values():
            pg_name = hashlib.sha1(bytes(pg_name + "_", "utf-8")).hexdigest()
    else:
        # 如果不使用哈希名称，生成一个顺序递增的名称
        pg_name = str(_world.group_count)
        _world.group_count += 1
    return pg_name


def _get_backend_from_str(backend: Optional[str] = None) -> Backend:
    # 如果未指定后端，则默认使用全局进程组的后端
    if not backend:
        backend = get_backend(_get_default_group())
    return Backend(backend)


@_time_logger
def new_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """
    创建一个新的分布式组。

    这个函数要求所有主组中的进程（即所有分布式作业的一部分的进程）都进入此函数，
    即使它们不会成为组的成员。此外，组应该在所有进程中按相同的顺序创建。
    """
    # 默认使用全局进程组的后端，如果未指定后端
    if not backend:
        backend = get_backend(_get_default_group())
    return Backend(backend)
    # 引入必要的库
    .. warning::
        Using multiple process groups with the ``NCCL`` backend concurrently
        is not safe and the user should perform explicit synchronization in
        their application to ensure only one process group is used at a time.
        This means collectives from one process group should have completed
        execution on the device (not just enqueued since CUDA execution is
        async) before collectives from another process group are enqueued.
        See `Using multiple NCCL communicators concurrently <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently>`_ for more details.
    
    # 定义一个函数，用于创建分布式进程组
    def new_group(
        # 成员组的排名列表。如果为 ``None``，将设置为所有排名。默认为 ``None``。
        ranks (list[int]): 
        # 超时时间，详见 `init_process_group` 获取详细信息和默认值。
        timeout (timedelta, optional): 
        # 使用的后端。根据编译时配置，有效值为 ``gloo`` 和 ``nccl``。
        # 默认使用与全局组相同的后端。此字段应作为小写字符串（例如， ``"gloo"``）给出，
        # 也可以通过 :class:`Backend` 属性访问（例如， ``Backend.GLOO``）。
        # 如果传入 ``None``，将使用与默认进程组对应的后端。默认为 ``None``。
        backend (str or Backend, optional): 
        # 进程组选项，指定在特定进程组构建期间需要传递的其他选项。
        # 例如对于 ``nccl`` 后端，可以指定 ``is_high_priority_stream``，以便进程组可以选择高优先级的 cuda 流。
        pg_options (ProcessGroupOptions, optional): 
        # 执行组局部同步。这在于非成员排名不需要调用 API 并且不参加障碍。
        use_local_synchronization (bool, optional): 
        # 描述进程组的字符串。
        group_desc (str, optional): 
    
    # 返回一个分布式组的句柄，可以提供给集体调用，或者如果排名不在 ``ranks`` 中则为 GroupMember.NON_GROUP_MEMBER。
    Returns:
        A handle of distributed group that can be given to collective calls or
        GroupMember.NON_GROUP_MEMBER if the rank is not part of ``ranks``.
    
    # 使用 use_local_synchronization=True 时不适用于 MPI。
    
    # 使用 use_local_synchronization=True 可以在更大的集群和小进程组中显著提高速度，但必须小心，
    # 因为它会改变集群行为，因为非成员排名不参加组障碍()。
    
    # 当每个排名创建多个重叠的进程组时，使用 use_local_synchronization=True 可能会导致死锁。
    # 为了避免这种情况，请确保所有排名遵循相同的全局创建顺序。
    # 使用给定的参数调用 _new_group_with_tag 函数，并返回其结果
    return _new_group_with_tag(
        ranks,  # 参数：用于组成新组的排名列表
        timeout,  # 参数：操作超时时限
        backend,  # 参数：后端处理器
        pg_options,  # 参数：PG选项（可能是PostgreSQL选项的缩写）
        None,  # 参数：未指定的额外参数
        use_local_synchronization=use_local_synchronization,  # 参数：是否使用本地同步
        group_desc=group_desc,  # 参数：组描述信息
    )
def _new_group_with_tag(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    pg_tag=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """
    Variant of ``new_group`` that exposes tag creation.

    :: N.B. The mechanism is experimental and tied to the functional collectives effort, see
    ``torch.distributed._functional_collectives`` for reference on how to use it.
    """
    global _world  # 全局变量 _world

    # 获取默认进程组，并从中获取绑定设备的设备 ID
    default_pg = _get_default_group()
    device_id = default_pg.bound_device_id
    # 获取默认进程组的默认后端和存储
    default_backend, default_store = _world.pg_map[default_pg]
    # 获取默认进程组的全局秩和全局世界大小
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()

    # 如果未指定后端，则默认使用与全局进程组相同的后端
    if not backend:
        backend = default_backend
    backend = Backend(backend)  # 将后端名称转换为 Backend 对象

    # 设置超时时间，如果未指定，则使用默认后端的默认超时时间
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)  # 检查超时时间的有效性

    # 如果启用了局部同步且使用 MPI 后端，则抛出异常
    if use_local_synchronization:
        if backend == Backend.MPI:
            raise ValueError(
                "MPI backend doesn't support use_local_synchronization=True"
            )
        # 如果指定了 ranks，并且当前全局秩不在 ranks 中，则返回 None
        if ranks is not None and get_rank() not in ranks:
            return None

    # 检查输入的 ranks 是否有效
    if ranks is not None:
        ranks = sorted(ranks)  # 对 ranks 进行排序
        group_world_size = len(ranks)  # 获取组的世界大小
        # 新组的世界大小不能大于全局世界大小
        if group_world_size > global_world_size:
            raise ValueError(
                "the new group's world size should be less or "
                "equal to the world size set by "
                "init_process_group"
            )
        # 检查每个 rank 的有效性
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise ValueError(
                    "The new group's rank should be within "
                    "the world_size set by init_process_group"
                )
        # 如果当前全局秩在 ranks 中，则确定组秩
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        # 如果未指定 ranks，则创建包含全局世界大小的 ranks 列表，并设置组秩为全局秩
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    # 根据 ranks 和 use_local_synchronization 标志生成进程组名
    group_name = _process_group_name(ranks, use_hashed_name=use_local_synchronization)

    # 调用辅助函数创建新的进程组
    pg, pg_store = _new_process_group_helper(
        group_world_size,
        group_rank,
        ranks,
        backend,
        default_store,
        group_name,
        pg_options=pg_options,
        timeout=timeout,
        pg_tag=pg_tag,
        device_id=device_id,
        group_desc=group_desc,
    )

    # 创建全局秩到组秩的映射
    _world.pg_group_ranks[pg] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }
    # 检查是否需要在初始化后执行屏障操作
    if _is_barrier_after_init() == 1:
        # 在方法返回后确保所有进程组（包括全局变量）在所有进程上正确更新。
        # 更新于 2023 年 4 月：对于大规模运行，这种屏障（特别是基于存储的屏障）可能成本高且/或不可伸缩。
        # 同时，很多情况下这些屏障可能是不必要的，这由移除后的绿色 CI 所证明。
        # 添加了环境变量 `TORCH_DIST_INIT_BARRIER`，设置为 1 时启用此屏障。
        logger.info(
            "Performing barrier after ProcessGroup initialization since "
            "TORCH_DIST_INIT_BARRIER = 1"
        )
        # 如果使用 MPI 后端
        if backend == Backend.MPI:
            # MPI 没有存储。
            barrier()
        else:
            # 如果不使用本地同步，则使用默认存储；否则使用本地同步存储。
            barrier_store = pg_store if use_local_synchronization else default_store
            # 如果使用本地同步，则世界大小为 ranks 的长度；否则获取全局世界大小。
            world_size = len(ranks) if use_local_synchronization else get_world_size()
            # 在这里使用基于存储的屏障，因为 barrier() 使用了大量默认设备并且会干扰 NCCL 的内部状态。
            _store_based_barrier(
                global_rank, barrier_store, group_name, world_size, timeout
            )

    # 返回 ProcessGroup 对象
    return pg
# 定义一个函数，用于创建等大小的子组。

def new_subgroups(
    group_size=None,  # 子组的大小，如果未指定则为 None
    group=None,  # 主组对象，用于创建子组
    timeout=None,  # 超时时间，如果未指定则为 None
    backend=None,  # 后端类型，用于创建子组
    pg_options=None,  # 进程组选项，用于创建子组
    group_desc=None,  # 组描述，用于创建子组
):
    """
    Create subgroups of equal size.

    By default, it creates intra-machine subgroups,
    where each of which contains all the ranks of a machine, based on the assumption
    that each machine has the same number of devices.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        If ``group_size`` is passed in, the world size must be divisible by ``group_size``.
        If no ``group_size`` is passed in, it believe that you are creating a group based
        on CUDA and determining the group size by number of CUDA devices, and if not all
        the machines have the same number of devices, the subgroup division will be
        different across nodes and can cause unexpected behaviors. Therefore, if you are
        creating a subgroup that does not depend on CUDA (such as Gloo on CPU), please
        pass in ``group_size`` correctly.

    .. warning::
        Using multiple process groups with the ``NCCL`` backend concurrently
        is not safe and the user should perform explicit synchronization in
        their application to ensure only one process group is used at a time.
        This means collectives from one process group should have completed
        execution on the device (not just enqueued since CUDA execution is
        async) before collectives from another process group are enqueued.
        See `Using multiple NCCL communicators concurrently <https://docs.nvid
        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using
        -multiple-nccl-communicators-concurrently>`_ for more details.
    """
    # 如果未指定子组大小，则根据CUDA设备数设置默认子组大小
    if group_size is None:
        # 检查是否可用CUDA，否则抛出数值错误
        if not torch.cuda.is_available():
            raise ValueError(
                "Default group size only takes effect when CUDA is available."
                "If your subgroup using a backend that does not depend on CUDA,"
                "please pass in 'group_size' correctly."
            )
        # 设置默认子组大小为当前CUDA设备数
        group_size = torch.cuda.device_count()
    
    # 检查指定的子组大小是否为正数
    if group_size <= 0:
        raise ValueError(f"The arg 'group_size' ({group_size}) must be positive")

    # 获取当前环境下的进程总数
    world_size = get_world_size()
    
    # 检查指定的子组大小是否超过了当前环境下的进程总数
    if world_size < group_size:
        raise ValueError(
            f"The arg 'group_size' ({group_size}) must not exceed the world size ({world_size})"
        )
    
    # 检查当前环境下的进程总数是否能被指定的子组大小整除
    if world_size % group_size != 0:
        raise ValueError("The world size must be divisible by 'group_size'")
    
    # 初始化空的子组列表和当前子组变量
    subgroups = []
    cur_subgroup = None
    # 循环创建子组，每个子组包含一部分进程
    for subgroup_id in range(world_size // group_size):
        # 计算当前子组的起始和结束进程编号
        start_rank = subgroup_id * group_size
        end_rank = start_rank + group_size
        # 生成当前子组的进程编号列表
        ranks_in_subgroup = list(range(start_rank, end_rank))
        # 创建新的子组，包括指定的进程列表和其他参数
        subgroup = new_group(
            ranks=ranks_in_subgroup,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            group_desc=group_desc,
        )
        # 将创建的子组添加到子组列表中
        subgroups.append(subgroup)

        # 获取当前进程的编号
        rank = get_rank()
        # 如果当前进程编号在当前子组的进程列表中，则记录当前子组
        if rank in ranks_in_subgroup:
            cur_subgroup = subgroup
            # 输出日志，显示当前进程编号分配给哪个子组
            logger.info("Rank %s is assigned to subgroup %s", rank, ranks_in_subgroup)

    # 返回当前进程所属的子组和所有创建的子组列表
    return cur_subgroup, subgroups
# 定义一个函数用于枚举生成新的子组。

def new_subgroups_by_enumeration(
    ranks_per_subgroup_list,  # 接收一个嵌套列表，每个子列表包含组成子组的进程 ranks
    timeout=None,             # 超时时间，用于初始化进程组，默认为 None
    backend=None,             # 指定使用的后端，可选项为 "gloo" 或 "nccl"，默认为全局组使用的后端
    pg_options=None,          # 进程组选项，用于构建特定进程组时传递额外选项，例如在 "nccl" 后端中指定高优先级流
    group_desc=None,          # 描述组的字符串，每个子组将继承其 group_desc

):
    """
    Create subgroups by dividing the global world.

    The division is specified by a nested list of ranks. The subgroups cannot have
    overlap, and some ranks may not have to be in any subgroup.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        Using multiple process groups with the ``NCCL`` backend concurrently
        is not safe and the user should perform explicit synchronization in
        their application to ensure only one process group is used at a time.
        This means collectives from one process group should have completed
        execution on the device (not just enqueued since CUDA execution is
        async) before collectives from another process group are enqueued.
        See `Using multiple NCCL communicators concurrently <https://docs.nvid
        ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using
        -multiple-nccl-communicators-concurrently>`_ for more details.

    Args:
        ranks_per_subgroup_list (list[list[int]]): A nested list of ranks of
            group members.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
             build-time configurations, valid values are ``gloo`` and ``nccl``.
             By default uses the same backend as the global group. This field
             should be given as a lowercase string (e.g., ``"gloo"``), which can
             also be accessed via :class:`Backend` attributes (e.g.,
             ``Backend.GLOO``). If ``None`` is passed in, the backend
             corresponding to the default process group will be used. Default is
             ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams.
        group_desc (str, optional): A string describing the group. Each subgroup will
            inherit its group_desc.

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.
    """
    """
    如果 ranks_per_subgroup_list 为 None 或空列表，则抛出数值错误异常。
    """
    if ranks_per_subgroup_list is None or len(ranks_per_subgroup_list) == 0:
        raise ValueError("The arg 'ranks_per_subgroup_list' cannot be empty")

    subgroups = []
    cur_subgroup = None
    """
    为了检查是否存在子组重叠，创建一个从排名到子组的映射。
    """
    rank_to_ranks_dict = {}  # type: ignore[var-annotated]
    """
    遍历 ranks_per_subgroup_list 中的每个 ranks，创建新的分组。
    """
    for ranks in ranks_per_subgroup_list:
        """
        使用给定的 ranks 创建新的分组。
        """
        subgroup = new_group(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            group_desc=group_desc,
        )
        """
        将新创建的子组添加到 subgroups 列表中。
        """
        subgroups.append(subgroup)
        """
        获取当前进程的排名。
        """
        my_rank = get_rank()
        """
        遍历 ranks 中的每个 rank。
        """
        for rank in ranks:
            """
            检查该 rank 是否已经存在于 rank_to_ranks_dict 中，如果存在则抛出数值错误异常。
            """
            if rank in rank_to_ranks_dict:
                raise ValueError(
                    f"Rank {rank} has appeared in both subgroup {rank_to_ranks_dict[rank]} and {ranks}"
                )
            """
            将该 rank 映射到当前 ranks，并记录在 rank_to_ranks_dict 中。
            """
            rank_to_ranks_dict[rank] = ranks
            """
            如果当前进程的排名等于该 rank，则将 cur_subgroup 设置为当前的 subgroup，并记录日志信息。
            """
            if my_rank == rank:
                cur_subgroup = subgroup
                logger.info("Rank %s is assigned to subgroup %s", rank, ranks)

    """
    返回当前的子组 cur_subgroup 和所有创建的子组 subgroups 列表。
    """
    return cur_subgroup, subgroups
# 根据标签和排名查找对应的进程组，返回找到的进程组对象或None
def _find_pg_by_ranks_and_tag(tag: str, ranks: List[int]) -> Optional[ProcessGroup]:
    # 如果标签非空且不以"ptd:"或"user:"开头，则加上"user:"前缀
    if len(tag) > 0 and not tag.startswith("ptd:") and not tag.startswith("user:"):
        tag = f"user:{tag}"

    # 遍历标签映射到进程组的字典中查找对应标签的进程组列表
    for group in _world.tags_to_pg.get(tag, []):
        # 如果进程组的大小与给定排名列表的长度不同，则跳过此进程组
        if group.size() != len(ranks):
            continue
        
        # 获取当前进程组的排名列表
        group_ranks = get_process_group_ranks(group)
        # 检查给定的排名列表是否完全包含在当前进程组的排名列表中
        good = all(r in group_ranks for r in ranks)
        if good:
            return group  # 如果找到符合条件的进程组，则返回该进程组对象
    return None  # 如果未找到符合条件的进程组，则返回None


# 根据标签、排名列表和步幅查找或创建对应的进程组，返回对应的进程组对象
def _find_or_create_pg_by_ranks_and_tag(
    tag: str, ranks: List[int], stride: int
) -> ProcessGroup:
    # 确保排名列表的长度能被步幅整除
    assert (
        len(ranks) % stride == 0
    ), f"Ranks length ({len(ranks)}) must be divisible by stride ({stride})"

    # 获取当前节点的排名
    my_rank = get_rank()
    my_ranks = None

    # 如果步幅等于排名列表的长度，则直接使用整个排名列表
    if stride == len(ranks):
        my_ranks = ranks.copy()
        assert my_rank in my_ranks, "rankset doesn't include the current node"
    else:
        # 否则，按步幅划分排名列表，找到包含当前节点排名的子列表
        for i in range(0, len(ranks), stride):
            rank_set = ranks[i : i + stride]
            if my_rank in rank_set:
                my_ranks = rank_set
        assert my_ranks is not None, "rankset doesn't include the current node"

    # 对当前排名列表进行排序
    my_ranks.sort()

    # 根据标签和当前排名列表查找对应的进程组
    pg = _find_pg_by_ranks_and_tag(tag, my_ranks)
    if pg is not None:
        return pg  # 如果找到符合条件的进程组，则返回该进程组对象
    
    # 如果未找到符合条件的进程组，且标签为空，则抛出数值错误异常
    if tag == "":
        raise ValueError("Cannot automatically create PG with empty tag")
    
    # TODO: 从默认进程组复制设置和超时参数
    # 否则，调用函数创建一个新的带有指定标签的进程组，并返回该进程组对象
    return _new_group_with_tag(my_ranks, pg_tag=tag)


# 获取给定进程组关联的标签字符串
def _get_group_tag(pg: ProcessGroup) -> str:
    """Return the tag associated with ``pg``."""
    tag = _world.pg_to_tag[pg]
    # 如果标签以"user:"开头，则去掉"user:"前缀后返回
    if tag.startswith("user:"):
        tag = tag[5:]
    return tag


# 获取给定进程组的名称字符串，若未找到则返回默认字符串"None"
def _get_process_group_name(pg: ProcessGroup) -> str:
    return _world.pg_names.get(pg, "None")


# 获取给定进程组关联的存储对象
def _get_process_group_store(pg: ProcessGroup) -> Store:
    return _world.pg_map[pg][1]


# 不兼容TorchDynamo的分布式c10d操作列表，在FX图中禁用这些操作
# 允许在eager模式下使用torch.compile执行这些操作
dynamo_unsupported_distributed_c10d_ops = [
    recv,
    all_gather_object,
    all_gather_coalesced,
    all_to_all_single,
    all_reduce,
    gather_object,
    all_to_all,
    all_reduce_coalesced,
    gather,
    send_object_list,
    recv_object_list,
    broadcast_object_list,
    barrier,
    scatter,
    scatter_object_list,
    reduce,
    all_gather,
    reduce_scatter,
    all_gather_into_tensor,
    broadcast,
    reduce_scatter_tensor,
    send,
]
```