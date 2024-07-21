# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_common.py`

```
# 设置 mypy 选项，允许未标记类型的函数
mypy: allow-untyped-defs

# 导入所需的模块
import math
import traceback
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, cast, List, Optional

import torch
import torch._dynamo.compiled_autograd as ca
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import DTensorSpec

# 定义数据类 DataParallelMeshInfo
@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    # 初始化方法，确保 shard_mesh_dim 和 replicate_mesh_dim 至少有一个不为 None
    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )

# 定义数据类 FSDPMeshInfo，继承自 DataParallelMeshInfo
@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    # 初始化方法，调用父类的初始化方法，并确保 shard_mesh_dim 不为 None
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = self.shard_process_group.rank()

# 定义数据类 DDPMeshInfo，继承自 DataParallelMeshInfo
@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    # 初始化方法，调用父类的初始化方法，并确保 replicate_mesh_dim 不为 None
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()

# 定义数据类 HSDPMeshInfo，继承自 FSDPMeshInfo 和 DDPMeshInfo
@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    # 初始化方法，依次调用 FSDPMeshInfo、DDPMeshInfo 和 DataParallelMeshInfo 的初始化方法
    def __post_init__(self):
        super().__post_init__()

# 定义枚举类 TrainingState，描述 FSDP 状态/参数组的训练状态
class TrainingState(Enum):
    FORWARD = auto()  # 进入前向传播状态
    PRE_BACKWARD = auto()  # 进入反向传播前状态
    POST_BACKWARD = auto()  # 进入反向传播后状态
    IDLE = auto()  # 空闲状态

# 定义函数 _raise_assert_with_print，用于打印带有当前进程编号的断言错误信息
def _raise_assert_with_print(*args: Any, **kwargs: Any):
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)

# 定义函数 _is_composable_with_fsdp，判断模块是否与 FSDP 兼容
def _is_composable_with_fsdp(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    return "replicate" not in registry

# 定义函数 _get_dim0_padded_size，计算在维度 0 上填充后的大小
def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    # 计算第一维度的填充后的大小，确保能够被 dim0_factor 整除
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    # 返回一个新的 torch.Size 对象，第一个维度为 padded_dim0，其余维度与原始 tensor_size 保持一致
    return cast(torch.Size, torch.Size([padded_dim0]) + tensor_size[1:])
# 将输入的张量按指定维度分块，返回分块后的张量列表
def _chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> List[torch.Tensor]:
    # 使用 torch.chunk 函数将张量按指定维度分块，并转换为列表
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    # 如果分块后的列表长度不足 num_chunks，补充空张量
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


# 根据给定的块和未分块的大小，返回新的大小 torch.Size 对象
def _get_dim0_chunked_size(
    chunk: torch.Tensor, unchunked_size: torch.Size
) -> torch.Size:
    if chunk.numel() > 0:
        return chunk.size()
    # 对于零元素的情况，保留未分块的大小的其余维度以供 DTensor API 使用
    return cast(torch.Size, torch.Size([0]) + unchunked_size[1:])


# 在无梯度计算环境下，使用本地张量和分片规范创建 DTensor 对象
def _from_local_no_grad(
    local_tensor: torch.Tensor,
    sharding_spec: DTensorSpec,
) -> DTensor:
    """
    This method is similar to ``DTensor.from_local()`` except that in eager mode
    it avoids some CPU overhead by avoiding default args and not being differentiable.
    """
    if not ca.compiled_autograd_enabled:
        return DTensor(
            # 直接使用本地张量而不是通过 `view_as()` 等方法创建新的张量变量，因为这不是可微的
            local_tensor,
            sharding_spec,
            requires_grad=local_tensor.requires_grad,
        )
    else:
        return DTensor.from_local(
            local_tensor,
            sharding_spec.mesh,
            sharding_spec.placements,
            shape=sharding_spec.shape,
            stride=sharding_spec.stride,
        )


# 如果需要，将张量转换为指定的数据类型
def _to_dtype_if_needed(
    tensor: torch.Tensor, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


# 如果输入张量是浮点型且不是目标数据类型，则将其转换为目标数据类型
def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)
```