# `.\pytorch\torch\distributed\_tensor\debug\visualize_sharding.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import List, Sequence, Tuple

# 导入 NumPy 库，并重命名为 np
import numpy as np

# 从 torch 库中导入相关模块
from torch._prims_common import ShapeType
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Placement, Shard


def _mesh_to_coordinate(mesh, device_type):
    """
    将 n 维设备网格列表转换为设备和其坐标的映射
    """
    # 将设备网格列表转换为 NumPy 数组
    np_mesh = np.array(mesh.mesh.tolist())

    # 创建一个字典，将每个值映射到其坐标
    device_to_coordinate_map = {}
    for coord, value in np.ndenumerate(np_mesh):
        # 每个设备在设备网格中是唯一的
        device_to_coordinate_map[f"{device_type}:{str(value)}"] = list(coord)

    return device_to_coordinate_map


def _convert_offset_to_ranges(all_offsets):
    """
    使用 tabulate 包创建表格时，通过指定行和列范围更容易
    将偏移量转换为范围
    """
    converted_blocks = []

    for offset in all_offsets:
        shape, offset, value = offset

        # 计算行范围和列范围
        row_range = (offset[0], offset[0] + shape[0] - 1)
        column_range = (offset[1], offset[1] + shape[1] - 1)

        # 将值转换为字符串，以匹配所需的格式
        converted_block = {
            "row_range": row_range,
            "column_range": column_range,
            "value": str(value),
        }
        converted_blocks.append(converted_block)

    return converted_blocks


def _create_table(blocks):
    """
    根据行和列范围以及设备名称创建一个 tabulate 表格
    """
    try:
        # 尝试导入 tabulate 库
        from tabulate import tabulate
    except ImportError as e:
        # 如果导入失败，则抛出 ImportError
        raise ImportError("tabulate package is required to visualize sharding") from e

    # 提取唯一的行和列范围
    row_ranges = sorted({block["row_range"] for block in blocks})
    col_ranges = sorted({block["column_range"] for block in blocks})

    # 创建一个用空字符串初始化的矩阵
    matrix = [["" for _ in col_ranges] for _ in row_ranges]

    # 填充矩阵的值
    for block in blocks:
        row_index = row_ranges.index(block["row_range"])
        col_index = col_ranges.index(block["column_range"])
        if matrix[row_index][col_index] == "":
            matrix[row_index][col_index] = block["value"]
        else:
            matrix[row_index][col_index] += ", " + block["value"]

    # 准备表头
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]

    # 使用 tabulate 函数生成表格并返回
    return tabulate(matrix, headers=col_headers, showindex=row_headers)


def compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    my_coordinate: List[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    计算本地形状和全局偏移量
    """
    # 定义一个函数，与 torch.distributed._tensor._utils.compute_local_shape_and_global_offset 功能类似，
    # 但接受自定义的 my_coordinate 输入。这是为 visualize_sharding 修改后的实现。
    
    if my_coordinate is None:
        # 如果 my_coordinate 为空，表示当前进程在 mesh 中不存在，返回空的偏移量
        return ((), ())
    else:
        # 复制全局形状到本地形状
        local_shape = list(global_shape)
        # 初始化全局偏移为零数组
        global_offset = [0] * len(global_shape)
    
        # 遍历 placements 列表中的索引和元素
        for idx, placement in enumerate(placements):
            # 获取当前维度的 mesh 大小
            mesh_dim_size = mesh.size(idx)
            
            # 如果 placement 是 Shard 类的实例
            if isinstance(placement, Shard):
                # 获取 shard 的维度
                shard_dim = placement.dim
                # 初始化本地偏移为零数组
                local_offset = [0] * len(global_shape)
                
                # 断言 shard_dim 必须小于 local_shape 的维度数，否则抛出异常
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                
                # 调用 placement 的 _local_shard_size_on_dim 方法，获取 shard 在当前维度上的大小和偏移量
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )
    
                # 将 shard_size 赋值给 local_shape 的相应维度
                local_shape[shard_dim] = shard_size
                # 将 shard_offset 赋值给 local_offset 的相应维度
                local_offset[shard_dim] = shard_offset
    
                # 对于给定的维度，如果 local_offset[shard_dim] 小于等于 global_offset[shard_dim]，
                # 表示此维度已经在先前的 placement 中进行了分片。
                # 因此，不能简单地用 local_offset[shard_dim] 替换 global_offset[shard_dim]，
                # 而是需要将 local_offset[shard_dim] 添加到现有的 global_offset[shard_dim]。
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]
    
        # 返回转换为元组的 local_shape 和 global_offset
        return tuple(local_shape), tuple(global_offset)
def visualize_sharding(dtensor, header=""):
    """
    Visualizes sharding in 1D-2D dtensors
    Requires tabulate, install with `pip install tabulate`

    note: no sharding info will be printed for empty tensors
    """
    # 如果输入的 dtensor 是空的，则不打印分片信息
    if dtensor.numel() == 0:  # we do not print for empty dtensors
        return

    # 检查 dtensor 的维度是否大于等于3，只能处理1维或2维的情况
    if len(dtensor.shape) >= 3:
        raise RuntimeError(
            "visualize sharding is only implemented for 1D or 2D dtensor"
        )

    # 获取 dtensor 的分片信息和设备网格信息
    placements = dtensor.placements
    device_mesh = dtensor.device_mesh
    device_type = dtensor.device_mesh.device_type

    # 如果当前的 rank 不在设备网格中，不进行打印
    if device_mesh.get_coordinate() is None:  # current rank is not in the mesh
        return

    # 只在每个维度的坐标都为0的 rank 上显示一次可视化表格。
    # 例如，如果网格是一个完整网格，我们只在 rank 0 上打印。
    local_rank_zero_on_all_dim = all(
        device_mesh.get_local_rank(mesh_dim=dim) == 0 for dim in range(device_mesh.ndim)
    )
    if not local_rank_zero_on_all_dim:
        return

    # 将设备网格映射为坐标形式的字典
    device_map = _mesh_to_coordinate(device_mesh, device_type)

    # 计算所有设备的本地形状和全局偏移，并将结果存入 all_offsets 列表
    all_offsets = []
    for device in device_map:
        local_shape, global_offset = compute_local_shape_and_global_offset(
            dtensor.shape, device_mesh, placements, device_map[device]
        )
        all_offsets.append([local_shape, global_offset, device])

    # 将偏移转换为包含行范围的块，以便用于 tabulate
    blocks = _convert_offset_to_ranges(all_offsets)

    # 打印表格
    print(header)
    print(_create_table(blocks))
```