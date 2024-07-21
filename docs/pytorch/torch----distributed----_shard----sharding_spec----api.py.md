# `.\pytorch\torch\distributed\_shard\sharding_spec\api.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块，用于高阶函数操作
import functools
# 导入 operator 模块，提供了许多有用的运算符和函数
import operator
# 导入 ABC 抽象基类，用于定义抽象基类
from abc import ABC, abstractmethod
# 导入 dataclass 装饰器，用于轻松创建不可变的数据类
from dataclasses import dataclass
# 导入 Callable 和 Dict 类型，用于类型注解
from typing import Callable, Dict, List, TYPE_CHECKING

# 导入 PyTorch 深度学习框架
import torch
# 导入分布式相关模块，用于分布式计算
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
# 导入分片元数据类，描述张量的分片方式
from torch.distributed._shard.metadata import ShardMetadata
# 导入操作注册工具函数
from torch.distributed._shard.op_registry_utils import _decorator_func

# 导入内部模块，包括一些实用函数和验证工具
from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata,
)

# 当进行类型检查时执行以下代码块，主要用于解决循环依赖问题
if TYPE_CHECKING:
    # 导入 ShardedTensor 类型，仅在类型检查时使用，运行时排除以避免循环依赖
    from torch.distributed._shard.sharded_tensor import ShardedTensor


# 表示实体放置的基类，用于指定自定义放置方式
class PlacementSpec(ABC):  # noqa: B024
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    pass


# 数据类，关联实体与单个设备的放置方式
@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """
    device: torch.distributed._remote_device

    # 初始化方法，在验证设备类型后，将其包装成 _remote_device 对象
    def __post_init__(self):
        if not isinstance(self.device, torch.distributed._remote_device):
            self.device = torch.distributed._remote_device(self.device)


# 表示分片规范的基类
class ShardingSpec(ABC):
    """
    Base class representing sharding specifications.
    """

    # 抽象方法，根据全局张量大小构建元数据，描述如何分片张量
    @abstractmethod
    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """
        pass

    # 抽象方法，执行张量的分片操作
    @abstractmethod
    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ):
        pass
    ) -> "ShardedTensor":
        """
        返回类型声明为 'ShardedTensor' 的函数声明。
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): 需要分片的张量。
        Keyword args:
            src_rank (int, optional): 源排名，用作数据的真实来源，会被分片并散布到其余排名中。
                默认值为 0。
            process_group (ProcessGroup, optional): 要操作的进程组。如果为 None，
                将使用默认进程组。
        Returns:
            返回从给定张量分片生成的 :class:`ShardedTensor`。
        """
# 自定义的特定 ShardingSpec 的操作集合，使用字典存储每个操作对应的处理函数
_CUSTOM_SHARDING_SPEC_OPS: Dict[str, Dict[Callable, Callable]] = {}


def _has_custom_op(sharding_spec, op):
    """
    检查给定的 ShardingSpec 是否有特定操作的自定义实现。
    """
    # 获取 ShardingSpec 类名的全限定名
    class_name = type(sharding_spec).__qualname__
    # 检查类名是否在 _CUSTOM_SHARDING_SPEC_OPS 中，并且该操作是否在类名对应的字典中
    return (
        class_name in _CUSTOM_SHARDING_SPEC_OPS
        and op in _CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


def _dispatch_custom_op(
    sharding_spec, op: Callable, types, args, kwargs, process_group
):
    """
    如果存在，调用此 ShardingSpec 的自定义操作。
    """
    # 获取 ShardingSpec 类名的全限定名
    class_name = type(sharding_spec).__qualname__
    # 如果没有注册该操作的自定义实现，则抛出 RuntimeError 异常
    if not _has_custom_op(sharding_spec, op):
        raise RuntimeError(f"Custom op: {op} not registered for {class_name}")
    # 获取注册的自定义操作函数
    func = _CUSTOM_SHARDING_SPEC_OPS[class_name][op]
    # 调用自定义操作函数，并返回其结果
    return func(types, args, kwargs, process_group)


def custom_sharding_spec_op(sharding_spec_class, func):
    """
    用于注册操作的装饰器。
    Args:
        sharding_spec_class(type): 需要为其添加自定义操作的 ShardingSpec 类型。
        func(Callable): 要重写的操作函数（例如：torch.bmm）。
    """
    # 获取 ShardingSpec 类名的全限定名
    class_name = sharding_spec_class.__qualname__
    # 如果类名不在 _CUSTOM_SHARDING_SPEC_OPS 中，则添加空字典作为值
    if class_name not in _CUSTOM_SHARDING_SPEC_OPS:
        _CUSTOM_SHARDING_SPEC_OPS[class_name] = {}
    # 返回部分应用的装饰器函数 _decorator_func，其中包括操作函数和操作表
    return functools.partial(
        _decorator_func, op=func, op_table=_CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    这是一种 PlacementSpec 类型，允许用户通过枚举每个分片的布局方式来指定通用分片方案。
    
    Args:
        shards(List[ShardMetadata]): 包含每个分片的 :class:`ShardMetadata` 对象列表。注意，分片不能重叠。
    """

    shards: List[ShardMetadata]

    def __post_init__(self):
        if len(self.shards) == 0:
            raise ValueError(f"Empty shard list provided: {self.shards}")

        # 验证每个分片是否具有相同的秩（rank）
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(
                    f"Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}"
                )
            rank = len(shard.shard_offsets)

        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        # 检查分片是否构成有效的张量
        check_tensor(self.shards, tensor_sizes)
        return sharded_tensor_meta.ShardedTensorMetadata(
            self.shards, tensor_sizes, tensor_properties
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ):
    ) -> "ShardedTensor":
        # 定义函数签名，该函数接受任意参数并返回一个字符串 "ShardedTensor"
        # TODO: figure out a generic and efficient way to scatter the shards for EnumerableShardingSpec
        # 抛出未实现错误，提示尚未实现对 EnumerableShardingSpec 的 sharding 功能
        raise NotImplementedError("EnumerableShardingSpec.shard not implemented yet!")
def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    从ShardedTensor的每个分片的元数据推断分片规范。
    如果张量仅在一个维度上分片，我们可以验证它是否是ChunkShardingSpec。
    验证的方法是首先获取总长度，并使用给定的放置方案执行块分片，看看是否可以得到与给定shards_metadata相同的块大小。
    如果不能，则假定它是枚举分片。

    Args:
        shards_metadata (List[ShardMetadata]): 本地分片的元数据列表。

    Returns:
        torch.distributed._shard.sharding_spec.ShardingSpec的对象，表示一个分片张量的分片规范。
    """
    placements = []  # 存储分片的放置位置信息
    chunk_sharding_dim = None  # 初始化块分片的维度为None
    chunk_offset_list = []  # 存储每个块的偏移量列表
    shard_size_list = []  # 存储每个分片的大小列表
    shard_offset_list = []  # 存储每个分片的偏移量列表

    # 从全局的sharded_tensor_metadata中收集本地分片的元数据
    for shard_metadata in shards_metadata:  # 遍历每个分片的元数据
        placements.append(shard_metadata.placement)  # 添加分片的放置位置信息
        local_offsets = shard_metadata.shard_offsets  # 获取分片的本地偏移量
        chunk_offset_list.append(sum(local_offsets))  # 计算并添加所有本地偏移量的总和
        shard_size_list.append(shard_metadata.shard_sizes)  # 添加分片的大小信息
        shard_offset_list.append(shard_metadata.shard_offsets)  # 添加分片的偏移量信息
        shard_dims = [idx for idx, e in enumerate(local_offsets) if e != 0]  # 找到非零偏移量对应的维度

        # 如果偏移量全部为零，则无法确定张量的分片方式
        if len(shard_dims) == 0:
            continue

        # 如果偏移量在多个维度上不为零，则张量肯定是多维分片的
        if len(shard_dims) != 1:
            chunk_sharding_dim = None
            break

        # 如果偏移量在仅一个维度上不为零，则需要确保所有的秩共享相同的维度
        if not chunk_sharding_dim:
            chunk_sharding_dim = shard_dims[0]
        elif chunk_sharding_dim != shard_dims[0]:
            chunk_sharding_dim = None
            break
    # 如果指定了块分片维度，则需要确保从偏移量推断出正确的放置顺序
    placements = [
        x
        for _, x in sorted(
            zip(chunk_offset_list, placements), key=operator.itemgetter(0)
        )
    ]

    # 导入块分片规范模块
    from .chunk_sharding_spec import ChunkShardingSpec

    # 创建块分片规范对象，指定分片维度和放置顺序
    chunk_spec = ChunkShardingSpec(
        dim=chunk_sharding_dim,
        placements=placements,
    )

    # 计算分片大小的排序列表和总长度
    shard_sizes = sorted([x[chunk_sharding_dim] for x in shard_size_list])
    shard_total_length = sum(shard_sizes)
    
    # 计算分片偏移量的排序列表
    shard_offsets = sorted([x[chunk_sharding_dim] for x in shard_offset_list])

    # 获取放置的块数
    chunks = len(placements)

    # 计算分片的分割大小
    split_size = get_split_size(shard_total_length, chunks)

    # 计算每个块的分片大小列表
    chunk_shard_sizes = sorted(
        [
            get_chunked_dim_size(shard_total_length, split_size, idx)
            for idx in range(chunks)
        ]
    )

    # 计算每个块的分片偏移量，应该与 ChunkShardingSpec 的偏移量计算匹配
    chunk_shard_offsets = [split_size * idx for idx in range(chunks)]

    # 如果计算出的块分片大小和偏移量与原始的分片大小和偏移量匹配，则返回块分片规范
    if shard_sizes == chunk_shard_sizes and shard_offsets == chunk_shard_offsets:
        return chunk_spec

    # 如果条件不满足，则返回一个默认的 EnumerableShardingSpec 对象
    return EnumerableShardingSpec(shards_metadata)
```