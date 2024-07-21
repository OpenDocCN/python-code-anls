# `.\pytorch\torch\distributed\_tensor\_utils.py`

```py
# 从 typing 模块导入类型提示的相关类和函数
from typing import cast, List, Sequence, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.distributed._tensor.api as dtensor
from torch._prims_common import ShapeType
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh


# TODO: 审查现有代码库，看看是否可以安全地移除此 API。
# 计算当前 DTensor 在其所在网格坐标上的本地分片的形状
def compute_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    # 获取当前进程的网格坐标
    my_coordinate = mesh.get_coordinate()

    # 如果当前进程不在网格中，则返回空形状
    if my_coordinate is None:
        return (0,)
    else:
        # 复制全局形状作为本地形状的起始点
        local_shape = list(global_shape)
        ndim = len(global_shape)
        
        # 遍历每个放置对象，调整对应维度的本地形状
        for idx, placement in enumerate(placements):
            # 获取当前维度在网格中的大小
            mesh_dim_size = mesh.size(idx)
            
            # 如果放置对象是 Shard 类型
            if isinstance(placement, Shard):
                # 获取分片的维度
                shard_dim = placement.dim
                # 确保分片维度不超过张量的维度数
                assert shard_dim < ndim, f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                
                # 调用对象方法计算当前维度上的本地分片大小和偏移量
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                assert isinstance(local_shard_size, int)
                
                # 更新本地形状中的对应维度大小
                local_shape[shard_dim] = local_shard_size
        
        # 返回调整后的本地形状作为元组
        return tuple(local_shape)


# 计算当前 DTensor 在其全局等级上的本地张量形状和原始张量的全局偏移量
def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.
    """
    # 获取当前进程的网格坐标
    my_coordinate = mesh.get_coordinate()

    # 如果当前进程不在网格中，则返回空形状和空偏移量元组
    if my_coordinate is None:
        return (tuple(global_shape), tuple(global_shape))
    else:
        # 初始化本地形状和全局偏移量
        local_shape = []
        global_offset = []

        # 遍历每个放置对象
        for idx, placement in enumerate(placements):
            # 获取当前维度在网格中的大小
            mesh_dim_size = mesh.size(idx)
            
            # 如果放置对象是 Shard 类型
            if isinstance(placement, Shard):
                # 获取分片的维度
                shard_dim = placement.dim
                # 确保分片维度不超过张量的维度数
                assert shard_dim < len(global_shape), f"Sharding dim {shard_dim} greater than tensor ndim {len(global_shape)}"
                
                # 调用对象方法计算当前维度上的本地分片大小和偏移量
                local_shard_size, global_shard_offset = placement._local_shard_size_on_dim(
                    global_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                
                # 将本地分片大小和全局偏移量添加到对应列表中
                local_shape.append(local_shard_size)
                global_offset.append(global_shard_offset)
            else:
                # 如果不是 Shard 类型，则维持全局形状和全局偏移量一致
                local_shape.append(global_shape[idx])
                global_offset.append(0)

        # 返回本地形状和全局偏移量作为元组
        return tuple(local_shape), tuple(global_offset)
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1,], global_offset:[0,]
    rank1 -- local_shape:[1,], global_offset:[1,]
    rank2 -- local_shape:[0,], global_offset:[2,]
    rank5 -- local_shape:[0,], global_offset:[2,]
    rank3 -- local_shape:[0,], global_offset:[2,]
    rank4 -- local_shape:[0,], global_offset:[2,]
    rank6 -- local_shape:[0,], global_offset:[2,]
    rank7 -- local_shape:[0,], global_offset:[2,]
    ```
    Get the coordinate of the current process from the mesh object.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # If the current rank is not in the mesh, return empty shapes and offsets
        return ((), ())
    else:
        # Initialize local shape with global shape and global offset with zeros
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        # Iterate over placements and adjust local shape and global offset accordingly
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                
                # Ensure the shard dimension is valid
                assert shard_dim < len(local_shape), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                
                # Get the shard size and offset for the current placement
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                # Update local shape and local offset for the shard dimension
                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # Adjust global offset for the shard dimension considering existing sharding
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        # Return tuple of adjusted local shape and global offset
        return tuple(local_shape), tuple(global_offset)
def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[List[int], List[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplied by `world_size` per Sharding dim.
    The local stride is multiplied by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())  # 获取张量的尺寸并转换为列表形式
    tensor_stride = list(tensor.stride())  # 获取张量的步幅并转换为列表形式
    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)  # 获取网格在给定维度上的大小
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                raise AssertionError(
                    "Shard placements should have negative dims normalized in "
                    f"the user-facing APIs: {shard_placement}"
                )
            shard_dim = shard_placement.dim  # 获取分片维度

            assert (
                shard_dim < tensor.ndim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."

            local_dim_size = tensor_shape[shard_dim]  # 获取本地维度的大小
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size  # 更新全局尺寸

            # 恢复张量的步幅，修改大于当前分片维度的步幅
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")
    return tensor_shape, tensor_stride


def try_find_mesh_from_args(
    op_call: torch._ops.OpOverload, args: Sequence[object]
) -> DeviceMesh:
    """
    Find the device mesh object from args.
    It returns None if no mesh is found.
    NOTE: we can optimize this search if needed
    """
    # 对于参数列表中的每个参数进行遍历
    for arg in args:
        # 检查参数是否为 dtensor.DTensor 或 DTensorSpec 类型的实例
        if isinstance(arg, (dtensor.DTensor, DTensorSpec)):
            # 如果是，则返回其设备网格属性
            return arg.device_mesh
        # 如果参数是列表或元组，并且长度大于0，并且第一个元素是 dtensor.DTensor 或 DTensorSpec 的实例
        elif (
            isinstance(arg, (list, tuple))
            and len(arg) > 0
            and isinstance(arg[0], (dtensor.DTensor, DTensorSpec))
        ):
            # 返回第一个元素的设备网格属性
            return arg[0].device_mesh

    # 如果没有找到符合条件的参数，抛出异常
    raise ValueError(f"Cannot find device mesh from args for op : {op_call}.")
def compute_local_stride(
    global_stride: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride of the DTensor.
    NOTE: Currently this function is assuming the DTensor is evenly shardable.
    """
    # 初始化一个全为1的列表，用于存储每个维度的步长除数
    stride_divisors = [1] * len(global_stride)
    
    # 遍历所有的放置位置
    for mesh_idx, p in enumerate(placements):
        # 如果当前位置是分片的位置
        if p.is_shard():
            # 获取分片的维度
            i = cast(Shard, p).dim
            # tensor 的维度 i 被分片到 mesh 的维度 mesh_idx 上，
            # 因此需要将所有大于 global_stride[i] 的步长除以子网格大小
            for j in range(len(global_stride)):
                if global_stride[j] > global_stride[i]:
                    stride_divisors[j] *= mesh.size(mesh_idx)
    
    # 计算本地张量分片的步长，返回元组
    return tuple(
        global_stride[i] // stride_divisors[i] for i in range(len(global_stride))
    )
```