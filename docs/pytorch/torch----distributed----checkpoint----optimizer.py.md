# `.\pytorch\torch\distributed\checkpoint\optimizer.py`

```
# 导入必要的库
import dataclasses
from typing import cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as ShardTensorProperties,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    create_read_items_for_chunk_list,
)
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import (
    _element_wise_add,
    _element_wise_sub,
    _normalize_device_info,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device


STATE_DICT_2D_LAYOUT = Dict[str, Tuple[Optional[Sequence[int]], Sequence[int]]]


# TODO: Update docstrings for optimizer.py
# 仅将下列函数导出
__all__ = [
    "load_sharded_optimizer_state_dict",
]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    # 如果设备类型是 CPU，直接返回 "cpu"
    if device_type == "cpu":
        return "cpu"
    # 否则，获取设备模块，并检查设备是否可用，返回规范化后的设备信息
    device_module = _get_device_module(device_type)
    if device_module.is_available():
        return _normalize_device_info(
            device_type, global_rank % device_module.device_count()
        )
    # 如果设备不可用，则返回 "cpu"
    return "cpu"


def _create_colwise_spec(
    pg: Optional[dist.ProcessGroup] = None,
) -> ChunkShardingSpec:
    # 获取默认设备类型
    pg_device_type = dist.distributed_c10d._get_pg_default_device(pg).type
    # 根据是否存在进程组来选择设备放置策略
    if pg is None:
        # 对于无进程组情况，生成每个进程的设备放置字符串列表
        placements = [
            f"rank:{idx}/{_gen_rank_device(idx, pg_device_type)}"
            for idx in range(dist.get_world_size())
        ]
    else:
        # 对于有进程组情况，生成每个进程的设备放置字符串列表
        placements = [
            f"rank:{idx}/{_gen_rank_device(dist.get_global_rank(pg, idx), pg_device_type)}"
            for idx in range(pg.size())
        ]
    # 创建基于列划分的分片规格对象并返回
    return ChunkShardingSpec(
        dim=0,
        placements=cast(List[Union[_remote_device, str]], placements),
    )


def _is_nested_tensor(val: torch.Tensor) -> bool:
    # 检查给定的张量是否为嵌套张量
    # 检查变量 val 是否为 ShardedTensor 类型
    if type(val) is ShardedTensor:
        # 如果 val 是 ShardedTensor 类型，则进入条件判断
        # 检查 ShardedTensor 对象的本地分片是否为空
        if len(val.local_shards()) == 0:
            # 如果本地分片为空，返回 False
            return False
        # 检查 ShardedTensor 对象的第一个本地分片的 tensor 是否为 ShardedTensor 类型
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            # 如果第一个本地分片的 tensor 是 ShardedTensor 类型，返回 True
            return True
        # 检查 ShardedTensor 对象的第一个本地分片的 tensor 是否为 DTensor 类型
        if type(val.local_shards()[0].tensor) is DTensor:
            # 如果第一个本地分片的 tensor 是 DTensor 类型，抛出异常
            raise ValueError("Cannot handle DTensor nested insided ShardedTensor")
    # 如果 val 不是 ShardedTensor 类型，则进入 elif 分支
    elif type(val) is DTensor and (
        # 检查 val 是否为 DTensor 类型，并且其 _local_tensor 是 DTensor 或 ShardedTensor 类型
        type(val._local_tensor) is DTensor or type(val._local_tensor) is ShardedTensor
    ):
        # 如果满足上述条件，抛出异常
        raise ValueError("Cannot handle nested DTensor")
    # 默认情况下返回 False
    return False
# 分配一个张量并返回，根据给定的属性和大小
def _alloc_tensor(
    props: TensorProperties, size: Sequence[int], device_type: str = "cuda"
) -> torch.Tensor:
    return torch.empty(
        size=size,
        dtype=props.dtype,           # 张量的数据类型
        layout=props.layout,         # 张量的布局
        requires_grad=props.requires_grad,  # 是否需要梯度
        pin_memory=props.pin_memory,        # 是否固定在内存中
        device=cast(torch.device, _get_device_module(device_type).current_device()),  # 分配的设备
    )


# 获取二维布局状态字典
def _get_state_dict_2d_layout(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_2D_LAYOUT, Optional[dist.ProcessGroup]]:
    """
    加载正确的优化器状态的 TP 切片。

    这并不容易，因为无法从检查点元数据推断出每个张量的切片。
    我们利用模型的状态字典生成切片的 ST 来确定需要加载的内容。
    这非常脆弱，也许让 FSDP 计算这些信息会更容易。
    返回一个字典，其中键与 state_dict 的键相同，值是当前等级 TP 切片的 (偏移量, 大小) 元组。
    注意：state_dict *必须* 来自 FSDP.sharded_state_dict。
    """
    specs: STATE_DICT_2D_LAYOUT = {}  # 初始化状态字典二维布局
    dp_pg: Optional[dist.ProcessGroup] = None  # 分布式进程组，可选类型
    for key, value in state_dict.items():
        specs[key] = (None, value.size())  # 将每个键值对的大小加入 specs 中
        if _is_nested_tensor(value):  # 如果值是嵌套张量
            assert (
                len(value.local_shards()) == 1
            ), "Cannot handle ST with multiple shards"  # 不能处理具有多个分片的 ST
            assert isinstance(
                value, ShardedTensor
            ), "Can only handle nested ShardedTensor"  # 只能处理嵌套的 ShardedTensor
            shard = value.local_shards()[0]  # 获取本地分片
            specs[key] = (
                shard.metadata.shard_offsets,   # 分片的偏移量
                shard.metadata.shard_sizes,     # 分片的大小
            )
            dp_pg = shard.tensor._process_group  # 获取张量的进程组

    return (
        specs,    # 返回状态字典的二维布局
        dp_pg,    # 返回分布式进程组
    )


# 有偏移量的读取器类，继承自 DefaultLoadPlanner
class _ReaderWithOffset(DefaultLoadPlanner):
    translation: Dict[MetadataIndex, MetadataIndex]  # 翻译字典，映射元数据索引
    state_dict: STATE_DICT_TYPE  # 状态字典类型
    metadata: Metadata   # 元数据

    def __init__(self, fqn_to_offset: Dict[str, Sequence[int]]) -> None:
        super().__init__()   # 调用父类构造函数
        self.fqn_to_offset = fqn_to_offset   # 设置全限定名到偏移量的映射
        self.metadata = Metadata({})   # 初始化元数据为空字典
        self.state_dict = {}   # 初始化状态字典为空
        self.translation = {}   # 初始化翻译字典为空
    # 创建本地加载计划的方法，返回一个LoadPlan对象
    def create_local_plan(self) -> LoadPlan:
        # 初始化一个空列表，用于存放读取请求
        requests = []
        # 初始化一个空字典，用于存放索引的翻译映射
        self.translation = {}
        
        # 遍历状态字典中的每个键值对
        for fqn, obj in self.state_dict.items():
            # 获取当前对象的元数据
            md = self.metadata.state_dict_metadata[fqn]
            
            # 如果当前对象不是ShardedTensor类型
            if not isinstance(obj, ShardedTensor):
                # 调用函数创建读取项，并将结果添加到requests列表中
                requests += _create_read_items(fqn, md, obj)
                continue

            # 如果当前对象的全限定名不在fqn_to_offset字典中
            if fqn not in self.fqn_to_offset:
                # 调用函数创建读取项，并将结果添加到requests列表中
                requests += _create_read_items(fqn, md, obj)
                continue

            # 获取当前对象在fqn_to_offset字典中的偏移量
            offset = self.fqn_to_offset[fqn]

            # 断言当前对象的本地碎片数量为1
            assert len(obj.local_shards()) == 1
            # 获取原始碎片数据
            original_shard = obj.local_shards()[0]
            # 创建本地碎片存储元数据列表
            local_chunks = [
                ChunkStorageMetadata(
                    # 计算偏移后的碎片偏移量
                    offsets=torch.Size(
                        _element_wise_add(original_shard.metadata.shard_offsets, offset)
                    ),
                    # 获取原始碎片的碎片大小
                    sizes=torch.Size(original_shard.metadata.shard_sizes),
                )
            ]

            # 调用函数创建用于碎片列表的读取项
            reqs = create_read_items_for_chunk_list(
                fqn, cast(TensorStorageMetadata, md), local_chunks
            )

            # TODO: 读取项将具有一个被替换的MetadataIndex，需要修复它。
            # TODO: 我们应该修改_create_sharded_read_items以提供更方便的API。

            # 遍历读取项列表
            for ri in reqs:
                # 断言读取项的目标索引偏移量不为None
                assert ri.dest_index.offset is not None
                # 计算原始偏移量
                original_offset = _element_wise_sub(ri.dest_index.offset, offset)
                # 使用新的偏移量替换读取项的目标索引
                original_index = dataclasses.replace(
                    ri.dest_index, offset=torch.Size(original_offset)
                )
                # 将读取项的目标索引及其对应的原始索引添加到翻译映射字典中
                self.translation[ri.dest_index] = original_index

            # 将当前读取项列表添加到requests列表中
            requests += reqs
        
        # 返回包含所有读取请求的加载计划对象
        return LoadPlan(requests)

    # 查找张量的方法，根据索引获取张量数据
    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        # 调用父类方法查找索引对应的张量数据，如果索引不存在则返回原始索引对应的张量数据
        return super().lookup_tensor(self.translation.get(index, index))
    """
    Load a state_dict in conjunction with FSDP sharded optimizer state.

    This is the current recommended way to checkpoint FSDP.
    >>> # xdoctest: +SKIP
    >>> import torch.distributed.checkpoint as dist_cp
    >>> # Save
    >>> model: torch.nn.Model
    >>> optim_params = model.parameters()
    >>> optim = torch.optim.SGD(optim_params, lr=0.01)
    >>> # Save
    >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    >>>     state_dict = {
    >>>         "optimizer": FSDP.optim_state_dict(model, optim),
    >>>         "model": model.state_dict()
    >>>     }
    >>>     dist_cp.save_state_dict(
    >>>         state_dict=optim_state,
    >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
    >>>         planner=dist_cp.DefaultSavePlanner(),
    >>>     )
    >>>
    >>> # Load
    >>> with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
    >>>     model_state_dict = model_tp.state_dict()
    >>>     checkpoint = {
    >>>         "model": model_state_dict
    >>>     }
    >>>     dist_cp.load_state_dict(
    >>>         state_dict=checkpoint,
    >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
    >>>         planner=dist_cp.DefaultLoadPlanner(),
    >>>     )
    >>>     model.load_state_dict(checkpoint["model_state"])
    >>>
    >>>     optim_state = dist_cp.load_sharded_optimizer_state_dict(
    >>>         model_state_dict,
    >>>         optimizer_key="optimizer",
    >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
    >>>     )
    >>>
    >>>     flattened_osd = FSDP.optim_state_dict_to_load(
    >>>        model, optim, optim_state["optimizer"]
    >>>     )
    >>>
    >>>     optim.load_state_dict(flattened_osd)
    """
    metadata = storage_reader.read_metadata()

    # 获取模型状态字典的二维布局规范和分布式过程组（dp_pg）
    layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
    # 获取分布式过程组（dp_pg）的默认设备类型
    dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
    # 获取与设备类型相关联的设备模块
    device_module = _get_device_module(dp_pg_device_type)

    # 如果分布式过程组（dp_pg）为None，则设置空的放置信息列表
    if dp_pg is None:
        placements = []
        # 对于每个进程，确定设备信息并规范化，添加到放置信息列表中
        for i in range(dist.get_world_size()):
            device_info = _normalize_device_info(
                dp_pg_device_type, i % device_module.device_count()
            )
            placements.append(f"rank:{i}/{device_info}")
        # 创建基于块的分片规范（ChunkShardingSpec），在第0维度上分片
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
    else:
        # 根据分布式过程组（dp_pg）创建列式分片规范
        sharding_spec = _create_colwise_spec(dp_pg)

    # 创建用于优化器状态的state_dict
    state_dict: STATE_DICT_TYPE = {}

    # 创建完全限定名到偏移序列的字典
    fqn_to_offset: Dict[str, Sequence[int]] = {}
    # 遍历状态字典元数据中的每个键值对
    for key, value in metadata.state_dict_metadata.items():
        # 获取键对应的路径信息
        key_path = metadata.planner_data[key]
        
        # 如果路径的第一个元素不是优化器键，跳过当前循环
        if key_path[0] != optimizer_key:
            continue
        
        # 如果值是 BytesStorageMetadata 类型
        if isinstance(value, BytesStorageMetadata):
            # 将状态字典中该键对应的值设置为 "<bytes_io>"
            state_dict[key] = "<bytes_io>"
            continue
        
        # 如果值是 TensorStorageMetadata 类型
        # value: TensorStorageMetadata
        if value.size.numel() == 1:
            # 根据属性、大小和设备类型分配张量，并将其设置为状态字典的值
            state_dict[key] = _alloc_tensor(
                value.properties, value.size, dp_pg_device_type
            )
        elif dp_pg is None:
            # 如果分布式处理组不存在，创建分块分片张量，并将其设置为状态字典的值
            state_dict[key] = _create_chunk_sharded_tensor(
                _alloc_tensor(value.properties, value.size, dp_pg_device_type),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=device_module.device_count(),
                pg=_get_default_group(),
            )
        else:
            # 否则，根据指定键从布局规范中获取分配大小
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]
            
            # 构建分片张量的属性
            properties = ShardTensorProperties(
                dtype=value.properties.dtype,
                layout=value.properties.layout,
                requires_grad=value.properties.requires_grad,
                memory_format=value.properties.memory_format,
                pin_memory=value.properties.pin_memory,
            )
            
            # 使用分片规范构建元数据
            st_md = sharding_spec.build_metadata(torch.Size(alloc_size), properties)
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            
            # 遍历分片元数据中的每个分片元数据
            for shard_md in st_md.shards_metadata:
                # 如果当前分片的放置位置的排名不等于当前排名，则跳过当前循环
                if cast(_remote_device, shard_md.placement).rank() != current_rank:
                    continue
                # 将本地分片和元数据添加到本地分片列表中
                local_shards.append(
                    Shard(
                        tensor=_alloc_tensor(
                            value.properties, shard_md.shard_sizes, dp_pg_device_type
                        ),
                        metadata=shard_md,
                    )
                )
            
            # 使用本地分片和全局元数据初始化分片张量
            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )
            
            # 如果规范键在布局规范中存在且不为 None，则将完全限定名称映射到偏移量列表中
            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = cast(Sequence[int], layout_specs[spec_key][0])
            
            # 将分片张量设置为状态字典的值
            state_dict[key] = st
    
    # 无论我们在展开状态字典之前还是之后执行，效果都是一样的
    # 调用 load_state_dict 函数加载状态字典
    load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        # FIXME the type of planner is wrong in load_state_dict
        planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
    )
    
    # 根据规划器数据对状态字典进行展开
    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)
    
    # 返回展开后的状态字典
    return state_dict
```