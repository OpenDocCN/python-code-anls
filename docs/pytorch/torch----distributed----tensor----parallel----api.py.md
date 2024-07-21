# `.\pytorch\torch\distributed\tensor\parallel\api.py`

```
# 导入必要的模块和类
from fnmatch import fnmatch  # 导入 fnmatch 模块，用于文件名匹配
from typing import Dict, Union  # 导入类型提示模块中的 Dict 和 Union 类型

import torch  # 导入 PyTorch 模块
import torch.distributed._tensor.random as random  # 导入随机数生成相关的分布式张量模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._tensor import DeviceMesh  # 导入分布式张量中的设备网格类
from torch.distributed._tensor.random import (
    is_rng_supported_mesh,  # 导入判断是否支持的随机数生成网格函数
    TensorParallelRNGTracker,  # 导入张量并行随机数生成器追踪器类
)
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim  # 导入验证张量并行网格维度的函数
from torch.distributed.tensor.parallel.style import ParallelStyle  # 导入张量并行风格类


__all__ = [
    "parallelize_module",  # 将 parallelize_module 函数添加到 __all__ 中，使其能够通过 from ... import * 导入
]


def parallelize_module(  # 定义函数 parallelize_module，用于在 PyTorch 中应用张量并行
    module: nn.Module,  # 函数参数 module，类型为 nn.Module，表示要并行化的模块
    device_mesh: DeviceMesh,  # 函数参数 device_mesh，类型为 DeviceMesh，描述设备的网格拓扑结构
    parallelize_plan: Union[ParallelStyle, Dict[str, ParallelStyle]],  # 函数参数 parallelize_plan，类型为 Union[ParallelStyle, Dict[str, ParallelStyle]]，表示并行化计划
) -> nn.Module:  # 返回类型为 nn.Module，返回并行化后的模块
    """
    Apply Tensor Parallelism in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

    We parallelize module or sub_modules based on a parallelize_plan. The parallelize_plan contains
    :class:`ParallelStyle`, which indicates how user wants the module or sub_module
    to be parallelized.

    User can also specify different parallel style per module fully qualified name (FQN).

    Note that ``parallelize_module`` only accepts a 1-D :class:`DeviceMesh`, if you have a 2-D or N-D :class:`DeviceMesh`,
    slice the DeviceMesh to a 1-D sub DeviceMesh first then pass to this API(i.e. ``device_mesh["tp"]``)

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        parallelize_plan (Union[:class:`ParallelStyle`, Dict[str, :class:`ParallelStyle`]]):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how
            we prepare input/output for Tensor Parallelism or it can be a
            dict of module FQN and its corresponding :class:`ParallelStyle` object.
    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>> m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
        >>>

    .. note:: For complex module architecture like Attention, MLP layers, we recommend composing
        different ParallelStyles together (i.e. ``ColwiseParallel`` and ``RowwiseParallel``) and pass
        as a parallelize_plan, to achieves the desired sharding computation.
    """
    torch._C._log_api_usage_once("torch.distributed.tensor.parallel.parallelize_module")  # 记录一次 API 使用情况

    _validate_tp_mesh_dim(device_mesh)  # 验证张量并行网格的维度是否合法

    # instantiate a TP RNG state tracker if it's not there
    # 检查设备是否支持 RNG 支持的网格，并且当前的 RNG 追踪器不是 TensorParallelRNGTracker 类型的实例
    if is_rng_supported_mesh(device_mesh) and not isinstance(
        random._rng_tracker, TensorParallelRNGTracker
    ):
        # 如果不是，则创建一个新的 TensorParallelRNGTracker 实例并赋给 random._rng_tracker
        random._rng_tracker = TensorParallelRNGTracker(device_mesh.device_type)
        # TODO: 应该允许用户从配置中传递默认种子
        # 使用指定的基础种子（base_seed=1234）手动设置 RNG 追踪器的种子
        random._rng_tracker._manual_seed(device_mesh, base_seed=1234)
        # 默认情况下，在非张量并行区域执行随机操作。如果用户希望在张量并行区域执行，
        # 可以手动将 distribute_region_enabled 字段设置为 True
        random._rng_tracker.distribute_region_enabled = False

    # 如果 parallelize_plan 是 ParallelStyle 类型的实例，则调用其 _apply 方法并返回结果
    if isinstance(parallelize_plan, ParallelStyle):
        return parallelize_plan._apply(module, device_mesh)
    # 如果 parallelize_plan 是字典类型
    elif isinstance(parallelize_plan, dict):
        # 遍历 parallelize_plan 中的每个模块路径和并行化风格
        for module_path, parallelize_style in parallelize_plan.items():
            # 将模块路径按 "." 分割成路径组成部分
            path_splits = module_path.split(".")
            # 如果路径分割后为空，则抛出数值错误异常
            if len(path_splits) == 0:
                raise ValueError(
                    "Expect module path to be non-empty, but got empty string!"
                )
            # 循环处理路径组成部分
            while path_splits:
                # 从路径组成部分中弹出第一个部分作为当前处理的部分
                atom = path_splits.pop(0)
                # 过滤出模块中与当前部分匹配的子模块
                matched_children = filter(
                    # `t[0]` 是子模块的名称
                    lambda t: fnmatch(t[0], atom),
                    module.named_children(),
                )
                # 针对所有匹配的子模块应用给定的并行化计划
                for _, submodule in matched_children:
                    if path_splits:
                        # 如果还没有到达叶子模块，则以字典方式应用并行化风格到当前子模块的剩余路径
                        leaf_path = ".".join(
                            path_splits
                        )  # `atom` 后的剩余路径部分
                        parallelize_module(
                            submodule, device_mesh, {leaf_path: parallelize_style}
                        )
                    else:
                        # 否则，直接将并行化风格应用到当前子模块
                        parallelize_module(submodule, device_mesh, parallelize_style)
        # 处理完成后返回模块本身
        return module
    else:
        # 如果 parallelize_plan 类型既不是 ParallelStyle 类型也不是字典类型，则抛出类型错误异常
        raise TypeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )
```