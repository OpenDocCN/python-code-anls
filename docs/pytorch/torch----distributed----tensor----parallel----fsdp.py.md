# `.\pytorch\torch\distributed\tensor\parallel\fsdp.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型
import copy
from typing import Any, cast, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)

# 指定可以通过导入的符号
__all__ = ["DTensorExtensions"]


# 获取包围框的偏移量和局部张量大小
def _get_box(tensor: DTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(mesh_dim=0)

    if tensor.placements[0].is_shard():
        shard_dim = cast(DShard, placement).dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())


# 根据索引获取张量的包围框
def _get_box_for(tensor: DTensor, idx: int) -> Tuple[torch.Size, torch.Size]:
    offsets, size = _get_box(tensor)
    return (torch.Size([val * idx for val in offsets]), size)


# 获取本地坐标的包围框
def _get_local_box(tensor: DTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    coord = device_mesh.get_coordinate()
    assert coord is not None
    return _get_box_for(tensor, coord[0])


# 从 DTensor 创建 ShardMetadata
def _create_shard_md_from_dt(dt: DTensor, current_rank: int) -> ShardMetadata:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    offsets, sizes = _get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}",
    )


# 从 DTensor 和进程组创建 ShardedTensorMetadata
def _create_sharded_tensor_md_from_dt(
    dt: DTensor, dt_pg: c10d.ProcessGroup
) -> ShardedTensorMetadata:
    # 这里是关键点，我们必须生成一个具有完整覆盖范围但只有当前排名有效分片的 ShardedTensor。

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1
    # 遍历每个分片数量范围
    for i in range(shard_count):
        # 调用 _get_box_for 函数获取第 i 个分片的偏移量和大小
        offsets, sizes = _get_box_for(dt, i)
        # 将分片的偏移量和大小作为列表添加到 shards_md 中
        shards_md.append(
            ShardMetadata(
                shard_offsets=list(offsets),
                shard_sizes=list(sizes),
                # 确定分片的放置位置字符串，根据 i 的值决定使用 scapegoat_rank 或者 my_rank
                placement=(
                    f"rank:{scapegoat_rank if i > 0 else my_rank}/{dt._local_tensor.device}"
                ),
            )
        )

    # 返回一个包含分片元数据的 ShardedTensorMetadata 对象
    return ShardedTensorMetadata(
        shards_metadata=shards_md,  # 分片的元数据列表
        size=dt.size(),             # 张量的大小
        tensor_properties=TensorProperties(
            dtype=dt.dtype,             # 张量的数据类型
            layout=dt.layout,           # 张量的布局方式
            requires_grad=dt.requires_grad,  # 张量是否需要梯度
            # 忽略 memory_format 和 pin_memory，因为这些特性不被 DT 支持
        ),
    )
# 获取与给定 DTensor 相关的进程组
    # 如果输入的张量是 DTensor 类型，则执行以下逻辑
    elif type(tensor) is DTensor:
        # 获取张量的设备网格信息
        device_mesh = tensor.device_mesh
        # 断言设备网格的维度为1，当前仅处理1维的设备网格
        assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

        # 获取内部参数（局部张量）
        inner_param = tensor._local_tensor

        # 创建分块分片张量（_create_chunk_sharded_tensor 是一个函数）
        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        # 获取分布式进程组
        dt_pg = _get_dt_pg(tensor)

        # 在这里我们采用不同的方法，创建一个没有本地分片的分片张量，然后对其进行修补
        # 创建包含单个分片的分片列表
        shards = [
            Shard(inner_st, _create_shard_md_from_dt(tensor, dist.get_rank(dt_pg)))
        ]

        # 创建分片张量的元数据
        st_meta = _create_sharded_tensor_md_from_dt(tensor, dt_pg)
        # 设置张量属性，禁用梯度计算
        st_meta.tensor_properties.requires_grad = False

        # 从本地分片和全局元数据初始化分片张量
        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False,
        )

        # 返回创建的外部分片张量
        return st_outer
    else:
        # 如果输入张量不是 DTensor 类型，则调用函数创建分块分片张量
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )
def _chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension.

    The local rank will get its corresponding chunk as the local tensor to create a DTensor.
    """
    # 获取父设备网格
    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    # 如果找不到父设备网格，则抛出运行时错误
    if parent_mesh is None:
        raise RuntimeError("No parent device_mesh is found for FSDP device_mesh.")
    # 如果父设备网格维度小于2，则抛出运行时错误
    if parent_mesh.ndim < 2:
        raise RuntimeError(
            f"Found parent device_mesh of ndim={parent_mesh.ndim},",
            "but meshes must be at least 2D.",
        )

    # 克隆并分离张量，以返回一个从当前计算图中分离的新张量
    tensor = tensor.clone().detach()

    # 当张量不是 DTensor 类型时，执行以下操作
    if isinstance(tensor, torch.Tensor) and not isinstance(tensor, DTensor):
        # 对于张量，它在 TP 维度上进行复制，并在 FSDP 维度上进行分片
        # TP 是内部维度，而 FSDP 是外部维度
        # 因此，张量的分片放置为 (Shard(0), Replicate())
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        shard_placements[0] = DShard(0)  # 设置第一个维度的分片为 DShard(0) 类型

        # 从本地创建 DTensor，不运行检查，然后重新分配到指定的设备网格和分片位置
        return DTensor.from_local(
            tensor, parent_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=parent_mesh,
            placements=shard_placements,
        )
    else:
        # 获取张量的放置信息
        tp_placements = tensor.placements
        # 获取第一个放置信息
        tp_placement = tp_placements[0]

        # 将张量转换为本地张量
        tensor = tensor.to_local()

        # 对于 DTensor，首先沿着 tp 维度分片，然后沿着 FSDP 维度分片。
        # TP 是内部维度，FSDP 是外部维度。
        # 因此，张量的分片放置为 (Shard(0), tp_placement)。
        # 对于更高维度的网格，它会在其他维度上复制。例如，使用 HSDP，张量的分片放置为 (Replicate, Shard(0), tp_placement)。
        
        # 创建一个列表，用于存放在父网格每个维度上的复制放置对象
        replicate_placements = [Replicate() for _ in range(parent_mesh.ndim)]
        # 将最后一个维度的复制放置对象设置为 tp_placement
        replicate_placements[-1] = tp_placement  # type: ignore[call-overload]
        
        # 创建一个列表，用于存放在每个维度上的分片放置对象
        shard_placements = [Replicate() for i in range(parent_mesh.ndim)]  # type: ignore[misc]
        # 将倒数第二个维度的分片放置对象设置为 DShard(0)
        shard_placements[-2] = DShard(0)  # type: ignore[call-overload]
        # 将最后一个维度的分片放置对象设置为 tp_placement
        shard_placements[-1] = tp_placement  # type: ignore[call-overload]

        # 使用本地张量创建一个 DTensor 对象，并进行分布式重分配
        return DTensor.from_local(
            tensor, parent_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=parent_mesh,
            placements=shard_placements,
        )
# 预加载状态字典的函数，接受一个 torch.Tensor 参数，并返回包含该张量和分片列表的元组
def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[Shard]]:
    # 获取 tensor 对应的 ShardedTensor 对象的本地分片列表
    shards = cast(ShardedTensor, tensor).local_shards()
    # 如果只有一个分片且该分片的 tensor 类型是 ShardedTensor
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        # 获取内部的 ShardedTensor 对象
        inner_tensor = shards[0].tensor
        # 获取内部 ShardedTensor 对象的本地分片列表
        shards = inner_tensor.local_shards()  # pyre-ignore[16]
        # 将 tensor 更新为内部的 ShardedTensor 对象
        tensor = inner_tensor

    # 返回更新后的 tensor 和分片列表（如果有分片的话）
    return (tensor, shards if len(shards) > 0 else [])


# 所有聚合 DTensor 的函数，根据 parent_mesh 参数进行 FSDP 维度的全局聚合，并返回本地张量
def _all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: Optional[DeviceMesh],
) -> torch.Tensor:
    """All gather a DTensor in its FSDP dimension and return the local tensor."""
    # 断言 parent_mesh 和 tensor 的设备网格一致
    assert parent_mesh == tensor.device_mesh

    # 深拷贝 tensor 的 placements 属性列表
    placements = list(copy.deepcopy(tensor.placements))
    # 对 placements 列表中除了最后一个元素以外的所有元素进行替换为 Replicate() 对象
    # FSDP + TP: [Shard(0), tp_placement] -> [Replicate(), tp_placement]
    # HSDP + TP: [Replicate(), Shard(0), tp_placement] -> [Replicate(), Replicate(), tp_placement]
    for i in range(0, len(placements) - 1):
        placements[i] = Replicate()
    
    # 使用 tensor 的 redistribute 方法重新分布张量
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    # 将 tensor 转换为本地张量并返回
    return tensor.to_local()


class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """

    def __init__(self, device_handle) -> None:
        super().__init__()
        self.compute_stream = None
        self.device_handle = device_handle
        # 使用 torch._dynamo.disable 方法来禁用 dynamo，以避免 torch deploy 中的构建失败
        self.post_unflatten_transform = torch._dynamo.disable(self.post_unflatten_transform)  # type: ignore[method-assign]

    # 对输入的 tensor 进行预扁平化转换的方法，返回转换后的 tensor 和可选的参数
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        return _flatten_tensor(tensor)

    # 对输入的 tensor 进行后解扁平化转换的方法，返回转换后的 tensor
    # param_extension 参数是任意扩展参数
    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor:
        # 获取计算流或当前设备处理器流
        stream = self.compute_stream or self.device_handle.current_stream()
        # 在设备处理器流上执行后解扁平化 tensor 的调用
        with self.device_handle.stream(stream):
            # TODO: 这是一个短期修复，应该直接在计算流中执行 get_unflat_views
            result = _unflatten_tensor(
                tensor,
                param_extension,
                device_handle=self.device_handle,
                compute_stream=self.compute_stream,
            )
            # 设置 result 的 FSDP 扁平化属性
            _set_fsdp_flattened(result)
            return result
    # 将输入的张量分块（chunk）以便在分布式环境中使用，返回分块后的张量
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    # 将输入的分布式张量分块（chunk），返回分块后的张量
    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return _chunk_dtensor(tensor, rank, device_mesh)

    # 在加载状态字典前进行的张量转换操作，返回转换后的张量和分片（Shard）列表
    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Shard]]:
        return _pre_load_state_dict(tensor)

    # 聚合分布式张量（DTensor），返回聚合后的张量
    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: Optional[DeviceMesh],
    ) -> torch.Tensor:
        return _all_gather_dtensor(tensor, parent_mesh)
```