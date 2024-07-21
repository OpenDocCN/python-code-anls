# `.\pytorch\torch\distributed\tensor\parallel\_utils.py`

```
# mypy: allow-untyped-defs
# 导入警告模块，用于生成警告信息
import warnings
# 引入类型提示相关模块
from typing import Tuple, Union

# 导入需要的模块和类
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources

# 尝试导入 torch._dynamo.external_utils 模块中的 is_compiling 函数，若失败则定义一个空函数
try:
    from torch._dynamo.external_utils import is_compiling as is_torchdynamo_compiling
except Exception:
    
    def is_torchdynamo_compiling():  # type: ignore[misc]
        return False

# 定义一个类型别名，包括 Placement 类型和 Placement 类型的元组
LayoutsType = Union[Placement, Tuple[Placement, ...]]


def _deprecate_warnings(func_name: str, extra_msg: str) -> None:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either a :class:`Tensor` or :class:`DTensor`
    and only 1D :class:`DeviceMesh` is passed in.
    """
    # TODO: Will follow up with dynamo POC to make warnings.warn working with dynamo.
    # 如果不是在 torchdynamo 编译环境下，则发出 FutureWarning 警告
    if not is_torchdynamo_compiling():
        warnings.warn(
            f"{func_name} is deprecated and will be removed soon. {extra_msg}",
            FutureWarning,
            stacklevel=3,
        )


def _validate_tp_mesh_dim(
    device_mesh: DeviceMesh,
) -> None:
    """
    Check whether TP mesh dimension is valid or not.

    Args:
        device_mesh (:class:`DeviceMesh`):
            The `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        `True` if the mesh dimension
        is valid, `False` otherwise.
    """
    # 检查设备网格的维度是否超过了1，若超过则抛出 ValueError 异常
    if device_mesh.ndim > 1:
        raise ValueError(
            f"Tensor Parallel only accepts a 1D DeviceMesh, but found {device_mesh.ndim}D!"
            'If you have a 2-D or N-D device_mesh, consider passing in device_mesh["tp"]'
        )

    # 获取设备网格的父网格，并进行相关验证
    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh:
        tp_mesh_dim_in_parent = _mesh_resources.get_parent_mesh_dim(device_mesh)
        # 如果 TP 网格维度不等于其父网格的维度减一，则抛出 RuntimeError 异常
        if tp_mesh_dim_in_parent != parent_mesh.ndim - 1:
            raise RuntimeError(
                f"Found TP device_mesh on the {tp_mesh_dim_in_parent} dimension of its parent mesh.",
                "Currently we only support intranode TP and TP needs to be the innermost dimension on its parent mesh.",
            )
```