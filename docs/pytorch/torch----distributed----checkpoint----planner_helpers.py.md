# `.\pytorch\torch\distributed\checkpoint\planner_helpers.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数，包括类型定义和类型转换函数
from typing import Any, cast, List

# 导入 PyTorch 库
import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from torch.utils._pytree import tree_map_only_

# 导入本地定义的模块和类
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from .planner import (
    LoadItemType,
    ReadItem,
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from .resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)

# 声明模块对外公开的函数和类名列表
__all__: List[str] = ["create_read_items_for_chunk_list"]


# 根据输入的张量创建对应的块存储元数据
def _create_chunk_from_tensor(tensor: torch.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size([0] * len(tensor.size())), sizes=tensor.size()
    )


# 根据分片元数据创建块存储元数据
def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes),
    )


# 根据分片张量和分片元数据创建张量写入数据
def _sharded_tensor_metadata(
    sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> TensorWriteData:
    # 获取分片张量的属性
    shard_properties = sharded_tensor.metadata().tensor_properties

    # 根据分片张量的属性创建张量属性对象
    properties = TensorProperties(
        dtype=shard_properties.dtype,
        layout=shard_properties.layout,
        requires_grad=shard_properties.requires_grad,
        memory_format=shard_properties.memory_format,
        pin_memory=shard_properties.pin_memory,
    )

    # 返回张量写入数据对象
    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=properties,
        size=sharded_tensor.metadata().size,
    )


# 根据 DTensor 创建写入项
def _create_write_items_for_dtensor(fqn: str, tensor: DTensor) -> WriteItem:
    # 计算本地形状和全局偏移
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)

    # 返回写入项对象
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=sizes,
            ),
            properties=TensorProperties.create_from_tensor(tensor.to_local()),
            size=tensor.size(),
        ),
    )


# 根据分片张量和分片元数据创建写入项
def _create_write_item_for_shard(
    fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> WriteItem:
    # 获取分片偏移
    offsets = torch.Size(shard_md.shard_offsets)

    # 返回写入项对象
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md),
    )
# 创建一个写入操作的条目，用于存储一个张量（Tensor）
def _create_write_item_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    # 初始化偏移量，与张量维度相同的全零张量尺寸
    offsets = torch.Size([0] * len(tensor.size()))
    # 返回一个写入条目对象，包括元数据索引、条目类型（张量）、张量写入数据
    return WriteItem(
        index=MetadataIndex(fqn, offsets),  # 元数据索引对象
        type=WriteItemType.TENSOR,  # 条目类型为张量
        tensor_data=TensorWriteData(  # 张量写入数据对象
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=tensor.size()),  # 数据块存储元数据
            properties=TensorProperties.create_from_tensor(tensor),  # 张量属性
            size=tensor.size(),  # 张量尺寸
        ),
    )


# 创建一个写入操作的条目，用于存储 BytesIO 对象
def _create_write_item_for_bytesio(fqn: str, bytes: Any):
    # 返回一个写入条目对象，包括元数据索引、条目类型（BytesIO）
    return WriteItem(
        index=MetadataIndex(fqn),  # 元数据索引对象
        type=WriteItemType.BYTE_IO,  # 条目类型为 BytesIO
    )


# 创建一个读取操作的条目，用于读取 ByteIO 对象
def _create_read_item_for_byteio(
    dest_index, dest_offset, storage_index, storage_offset, length
):
    # 返回一个读取条目对象，包括读取类型（ByteIO）、目标索引及偏移、存储索引及偏移、长度信息
    return ReadItem(
        type=LoadItemType.BYTE_IO,  # 读取类型为 ByteIO
        dest_index=dest_index,  # 目标索引
        dest_offsets=torch.Size((dest_offset,)),  # 目标偏移量
        storage_index=storage_index,  # 存储索引
        storage_offsets=torch.Size((storage_offset,)),  # 存储偏移量
        lengths=torch.Size((length,)),  # 读取长度
    )


# 创建一个读取操作的条目，用于读取张量（Tensor）
def _create_read_item_for_tensor(
    dest_index, dest_offsets, storage_index, storage_offsets, lengths
):
    # 返回一个读取条目对象，包括读取类型（Tensor）、目标索引及偏移、存储索引及偏移、长度信息
    return ReadItem(
        type=LoadItemType.TENSOR,  # 读取类型为张量
        dest_index=dest_index,  # 目标索引
        dest_offsets=torch.Size(dest_offsets),  # 目标偏移量
        storage_index=storage_index,  # 存储索引
        storage_offsets=torch.Size(storage_offsets),  # 存储偏移量
        lengths=torch.Size(lengths),  # 读取长度
    )


# 创建用于加载块列表的读取操作条目列表
def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: List[ChunkStorageMetadata],
) -> List[ReadItem]:
    """
    根据检查点和本地数据块列表创建一系列读取条目（ReadItem）。

    这里使用重分片算法计算需要满足本地数据块（local_chunks）的读取操作，
    其中检查点由 checkpoint_md 描述。

    Args:
        fqn (str) : 传递给 ``ReadItem`` 的 state_dict FQN。
        checkpoint_md (TensorStorageMetadata): 来自检查点的张量的元数据。
        local_chunks (List[ChunkStorageMetadata]): 需要加载的本地数据块。

    Returns:
        所有输入数据块所需的 ``ReadItem`` 列表。
    """
    read_items = []
    # 这是一个简单的二次算法，可以稍后优化
    # 对本地分块列表中的每个分块进行枚举迭代
    for idx, shard in enumerate(local_chunks):
        # 对检查点元数据中的每个存储块进行枚举迭代
        for storage_idx, storage_md in enumerate(checkpoint_md.chunks):
            # 如果当前分块和存储块的元数据没有重叠部分，则继续下一个迭代
            if not _check_shard_metadata_pair_overlap(shard, storage_md):
                continue

            # 初始化存储偏移列表、目标偏移列表和长度列表
            storage_offsets = []
            dest_offsets = []
            lengths = []
            # 使用特定函数计算当前分块和存储块之间重叠的区域
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=storage_md, current_shard=shard
            ):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            # 为当前重叠区域创建读取项目，并添加到读取项目列表中
            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, shard.offsets, idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(fqn, storage_md.offsets, storage_idx),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    # 返回包含所有读取项目的列表
    return read_items
# 创建一个默认的仅包含元数据的保存计划对象
def _create_default_metadata_only_plan(state_dict: STATE_DICT_TYPE) -> SavePlan:
    # 存储所有保存请求的列表
    requests = []
    # 遍历状态字典中的每个项目
    for fqn, obj in state_dict.items():
        # 如果对象是 DTensor 类型
        if isinstance(obj, DTensor):
            # 创建用于 DTensor 的写入项并添加到请求列表中
            requests.append(_create_write_items_for_dtensor(fqn, obj))
        # 如果对象是 ShardedTensor 类型
        elif isinstance(obj, ShardedTensor):
            # 遍历 ShardedTensor 中每个分片的元数据，为每个分片创建写入项并添加到请求列表中
            for shard_md in obj.metadata().shards_metadata:
                requests.append(_create_write_item_for_shard(fqn, obj, shard_md))
        # 如果对象是 torch.Tensor 类型
        elif isinstance(obj, torch.Tensor):
            # 创建用于 Tensor 的写入项并添加到请求列表中
            requests.append(_create_write_item_for_tensor(fqn, obj))
        # 对于其他类型的对象
        else:
            # 创建用于 BytesIO 的写入项并添加到请求列表中
            requests.append(_create_write_item_for_bytesio(fqn, obj))
    # 返回一个保存计划对象，包含所有的保存请求
    return SavePlan(requests)


# 创建写入项列表的函数，根据对象的类型进行不同的处理
def _create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    # 如果对象有 __create_write_items__ 方法，通常是 DTensor 类型
    if hasattr(object, "__create_write_items__"):
        # 调用对象的 __create_write_items__ 方法生成写入项列表
        return object.__create_write_items__(fqn, object)
    # 如果对象是 ShardedTensor 类型
    elif isinstance(object, ShardedTensor):
        # 为 ShardedTensor 中的每个本地分片创建对应的写入项并返回列表
        return [
            _create_write_item_for_shard(fqn, object, shard.metadata)
            for shard in object.local_shards()
        ]
    # 如果对象是 torch.Tensor 类型
    elif isinstance(object, torch.Tensor):
        # 创建用于 Tensor 的写入项并返回列表
        return [_create_write_item_for_tensor(fqn, object)]
    # 对于其他类型的对象
    else:
        # 创建用于 BytesIO 的写入项并返回列表
        return [_create_write_item_for_bytesio(fqn, object)]


# 从 DTensor 创建块存储元数据的函数
def _create_chunk_from_dtensor(tensor: DTensor) -> ChunkStorageMetadata:
    # 计算本地形状和全局偏移
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)
    # 返回块存储元数据对象
    return ChunkStorageMetadata(
        offsets=offsets,
        sizes=sizes,
    )


# 从 Tensor 创建块存储元数据列表的函数
def _create_chunk_list(tensor: torch.Tensor) -> List[ChunkStorageMetadata]:
    # 如果 Tensor 有 __create_chunk_list__ 方法，通常是 DTensor 类型
    if hasattr(tensor, "__create_chunk_list__"):
        # 调用对象的 __create_chunk_list__ 方法生成本地块列表
        local_chunks = tensor.__create_chunk_list__()  # type: ignore[attr-defined]
    # 如果对象是 ShardedTensor 类型
    elif isinstance(tensor, ShardedTensor):
        # 为 ShardedTensor 中的每个本地分片创建对应的块存储元数据并返回列表
        local_chunks = [
            _chunk_for_shard(shard.metadata) for shard in tensor.local_shards()
        ]
    # 如果对象是 torch.Tensor 类型
    elif isinstance(tensor, torch.Tensor):
        # 从 Tensor 创建单个块存储元数据并返回列表
        local_chunks = [_create_chunk_from_tensor(tensor)]
    # 对于其他类型的对象，抛出异常
    else:
        raise ValueError(
            "Unsupported Type, expecting one of [Tensor, DTensor, ShardedTensor] "
            f",but got {type(tensor)}"
        )

    # 返回本地块存储元数据列表
    return local_chunks


# 创建读取项列表的函数，根据对象的类型进行不同的处理
def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    # 如果元数据不是 BytesStorageMetadata 类型
    if not isinstance(md, BytesStorageMetadata):
        try:
            # 创建对象的块存储元数据列表
            local_chunks = _create_chunk_list(obj)
        # 捕获可能的值错误异常
        except ValueError as ex:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, "
                + f"expected BytesStorageMetadata but found {type(md)}",
            ) from ex

        # 使用块存储元数据列表创建读取项列表
        return create_read_items_for_chunk_list(fqn, md, local_chunks)
    else:
        return [
            _create_read_item_for_byteio(
                dest_index=MetadataIndex(fqn),  # 设置目标索引为完全限定名称的元数据索引
                dest_offset=0,  # 设置目标偏移量为0
                storage_index=MetadataIndex(fqn),  # 设置存储索引为完全限定名称的元数据索引
                storage_offset=0,  # 设置存储偏移量为0
                length=0,  # 设置长度为0
            )
        ]
def _init_state_dict(state_dict: STATE_DICT_TYPE) -> None:
    # 调用 tree_map_only_ 函数，初始化 state_dict 中所有 torch.Tensor 类型的值
    tree_map_only_(torch.Tensor, _init_meta_tensor, state_dict)


def _init_meta_tensor(value: Any) -> Any:
    """
    初始化张量，将其移动到 meta 设备上，适用于 torch.Tensor 或 DTensor 类型。
    """

    # 获取对象的设备属性
    device = getattr(value, "device", None)
    # 如果设备是 meta 设备
    if device == torch.device("meta"):
        # 获取默认的分布式设备类型
        device_type = dist.distributed_c10d._get_pg_default_device().type
        # 获取当前设备
        device = cast(torch.device, _get_device_module(device_type).current_device())
        # 如果对象是 DTensor 类型
        if isinstance(value, DTensor):
            # 根据新的本地张量创建一个新的本地 DTensor
            new_local_tensor = torch.empty_like(value.to_local(), device=device)
            # 需要显式传递形状和步幅，因为 DTensor 可能分片不均匀
            dtensor = DTensor.from_local(
                new_local_tensor,
                device_mesh=value.device_mesh,
                placements=value.placements,
                shape=value.size(),
                stride=value.stride(),
            )
            return dtensor
        # 如果对象是 torch.Tensor 类型
        elif isinstance(value, torch.Tensor):
            # 根据指定设备创建一个与输入张量同样类型的空张量
            tensor = torch.empty_like(value, device=device)
            return tensor
        else:
            # 如果发现不支持的类型，则引发运行时错误
            raise RuntimeError(
                f"Found unsupported type {type(value)} for meta device loading."
            )
    else:
        # 如果设备不是 meta 设备，则直接返回输入值
        return value
```