# `.\pytorch\torch\distributed\algorithms\model_averaging\utils.py`

```
# mypy: allow-untyped-defs
# flake8: noqa C101
# 引入 itertools 库，用于高效迭代操作
import itertools
# 引入类型提示相关库
from typing import Dict, Iterable, Iterator, Union

# 引入 PyTorch 库
import torch
# 引入分布式训练相关库
import torch.distributed as dist

# 下面两个导入语句根据 USE_DISTRIBUTED 编译标志的不同可能会导致导入错误。
# 如果使用了这些导入但未设置该标志，将引发 ImportError。
from torch.distributed import group, ProcessGroup

# 定义模块对外暴露的函数和类列表
__all__ = [
    "average_parameters",
    "get_params_to_average",
    "average_parameters_or_parameter_groups",
]

# 平均参数函数
def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: ProcessGroup
):
    """
    Averages all the given parameters.

    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    # 确定要使用的分组，如果未提供 process_group，则使用默认的 WORLD 分组
    group_to_use = process_group if process_group is not None else group.WORLD
    # 如果当前进程不在指定的分组中，则不更新任何参数
    if dist._rank_not_in_group(group_to_use):
        return

    # 将参数迭代器分为两个相同的迭代器，用于后续操作
    params_it1, params_it2 = itertools.tee(params)
    # 如果输入的参数具有不同的数据类型，打包这些参数将触发隐式类型提升。
    # 原始参数数据类型将在后续解包过程中恢复。
    flat_params = torch.cat([p.data.reshape(-1) for p in params_it1])
    flat_params /= dist.get_world_size(group_to_use)
    # 确保 allreduce 操作不会与其他正在进行的进程组冲突
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 执行全局平均操作，将结果存储在 flat_params 中
    dist.all_reduce(flat_params, group=group_to_use)

    # 根据偏移量将 flat_params 中的数据分配给每个参数
    offset = 0
    for p in params_it2:
        p.data = flat_params[offset : offset + p.numel()].view_as(p).type_as(p)
        offset += p.numel()

# 获取需要进行平均的参数列表
def get_params_to_average(
    params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]
):
    """
    Return a list of parameters that need to average.

    This filters out the parameters that do not contain any gradients.
    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, torch.nn.Parameter):
            # 对于 model.parameters() 的输入参数
            param_data = param
            if param_data.grad is not None:
                filtered_params.append(param_data)
        elif isinstance(param, dict):
            # 对于 optimizer.param_groups 的输入参数
            for param_data in param["params"]:
                if param_data.grad is not None:
                    filtered_params.append(param_data)
        else:
            # 不支持的参数类型，抛出 NotImplementedError 异常
            raise NotImplementedError(
                f"Parameter input of type {type(param)} is not supported"
            )
    return filtered_params

# 平均参数或参数组的函数定义
def average_parameters_or_parameter_groups(
    params: Union[
        Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]
    ],
    process_group: ProcessGroup,
):
    """
    对模型或优化器参数组进行平均化处理。
    """
    # 调用 get_params_to_average 函数返回的迭代器，获取需要平均化的参数
    average_parameters(iter(get_params_to_average(params)), process_group)
```