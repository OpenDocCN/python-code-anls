# `.\pytorch\torch\distributed\device_mesh.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging  # 导入日志模块
import math  # 导入数学模块
import threading  # 导入线程模块
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
from torch.distributed import is_available  # 导入分布式模块的可用性检查函数
from torch.utils._typing_utils import not_none  # 导入类型提示相关的模块

# 定义导出的模块列表
__all__ = ["init_device_mesh", "DeviceMesh"]

# 如果分布式不可用，则创建一个桩（stub）以避免在文档测试时出现导入错误
if not is_available():
    import sys

    class _DeviceMeshStub:
        pass

    def _init_device_mesh_stub():
        pass

    # 设置模块属性为桩
    sys.modules["torch.distributed.device_mesh"].DeviceMesh = _DeviceMeshStub  # type: ignore[attr-defined]
    sys.modules[
        "torch.distributed.device_mesh"
    ].init_device_mesh = _init_device_mesh_stub  # type: ignore[attr-defined]

else:
    # 导入实际的分布式C10D模块中的相关函数和类
    from torch.distributed.distributed_c10d import (
        _find_pg_by_ranks_and_tag,
        _get_default_group,
        _get_group_tag,
        get_process_group_ranks,
        get_rank,
        get_world_size,
        init_process_group,
        is_initialized,
        new_group,
        ProcessGroup,
    )

    logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

    # 仅在类型检查时导入numpy的类型提示，用于数组的类型提示
    if TYPE_CHECKING:
        try:
            from numpy.typing import ArrayLike  # 导入NumPy的数组类型提示
        except ImportError:
            logger.warning(
                "DeviceMesh requires numpy >= 1.21 to be installed for type checking"
            )

    _mesh_resources: _MeshEnv = _MeshEnv()  # 初始化一个私有变量用于存储网格资源的环境信息

    def _get_device_handle(device_type: str = "cuda"):
        """
        获取与给定设备类型对应的模块，例如当设备类型是cuda时返回`torch.cuda`模块。
        如果没有对应的模块，则返回None。
        """
        return getattr(torch, device_type, None)  # 使用getattr动态获取torch中的设备类型对应的模块

    def init_device_mesh(
        device_type: str,
        mesh_shape: Tuple[int, ...],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        # 初始化设备网格的函数，指定设备类型、网格形状及网格维度名称
```