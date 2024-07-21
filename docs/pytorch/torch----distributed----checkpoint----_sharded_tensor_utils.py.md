# `.\pytorch\torch\distributed\checkpoint\_sharded_tensor_utils.py`

```
# 版权声明和导入模块
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的模块
import copy  # 导入copy模块，用于对象的深拷贝
from typing import TYPE_CHECKING  # 导入类型检查模块

# 导入分布式相关模块
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor, ShardMetadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.remote_device import _remote_device

# 导入内部模块和函数
from ._traverse import OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict
from .utils import _element_wise_add, _normalize_device_info

# 如果是类型检查，导入ShardedTensorMetadata用于类型提示
if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata


# TODO: We need to refactor this code.
# 定义一个函数用于扁平化分布式张量（ShardedTensor）
def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    r"""
    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    """
    # 初始化一个新的状态字典，用于存储扁平化后的结果
    new_state_dict: STATE_DICT_TYPE = {}
    # 定义一个函数 `rewrite_dict`，接受两个参数，一个是路径 `path`，一个是状态字典项 `value`，返回空值
    def rewrite_dict(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # 如果 `value` 不是 `ShardedTensor` 类型，将其设置到 `new_state_dict` 中并返回
        if not isinstance(value, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        
        # 获取 `value` 的本地分片
        shards = value.local_shards()

        # 如果没有分片，直接返回
        if len(shards) == 0:
            return
        
        # 如果分片数量不为 1，将 `value` 设置到 `new_state_dict` 中并返回
        if len(shards) != 1:
            set_element(new_state_dict, path, value)
            return
        
        # 获取外部分片
        outer_shard = shards[0]

        # 获取内部 `ShardedTensor`
        inner_st = outer_shard.tensor
        
        # 如果内部 `ShardedTensor` 不是 `ShardedTensor` 类型，将 `value` 设置到 `new_state_dict` 中并返回
        if not isinstance(inner_st, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        
        # 如果内部 `ShardedTensor` 的本地分片数量不为 1，抛出异常
        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        
        # 获取内部分片
        inner_shard = inner_st.local_shards()[0]

        # 创建本地分片列表，其中包含单个分片
        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    # 计算分片偏移量之和
                    shard_offsets=_element_wise_add(
                        outer_shard.metadata.shard_offsets,
                        inner_shard.metadata.shard_offsets,
                    ),
                    # 使用内部分片的分片大小
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    # 设置分片的位置信息，包括当前进程的排名和设备信息
                    placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}",
                ),
            )
        ]

        # 深度复制 `value` 的元数据，存储到 `st_meta` 中
        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())

        # 设置其他进程的排名，如果当前进程排名大于 0，则设为 0，否则设为 1
        other_rank = 0 if dist.get_rank() > 0 else 1
        # 规范化内部分片张量的设备信息
        device_info = _normalize_device_info(inner_shard.tensor.device.type, 0)

        # 移除内部 `ShardedTensor` 覆盖的外部 `ST` 分片
        for i, shard_md in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break

        # 为其他分片设置进程排名
        for shard_md in st_meta.shards_metadata:
            shard_md.placement = _remote_device(f"rank:{other_rank}/{device_info}")

        # 添加内部张量的其他内部分片
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=_element_wise_add(
                            outer_shard.metadata.shard_offsets,
                            inner_md.shard_offsets,
                        ),
                        shard_sizes=inner_md.shard_sizes,
                        placement=f"rank:{other_rank}/{device_info}",
                    )
                )

        # 最后添加这个分片
        st_meta.shards_metadata.append(local_shards[0].metadata)

        # 使用本地分片和全局元数据初始化 `ShardedTensor`
        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=local_shards,
            sharded_tensor_metadata=st_meta,
        )
        
        # 将 `st` 设置到 `new_state_dict` 中
        set_element(new_state_dict, path, st)

    # 遍历 `state_dict` 并应用 `rewrite_dict` 函数
    traverse_state_dict(state_dict, rewrite_dict)
    
    # 返回更新后的 `new_state_dict`
    return new_state_dict
```