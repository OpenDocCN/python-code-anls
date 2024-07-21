# `.\pytorch\torch\distributed\_shard\api.py`

```
# mypy: allow-untyped-defs
# 引入上下文管理器和类型提示所需的库
from contextlib import contextmanager
from typing import Optional

import torch  # 引入PyTorch库
import torch.distributed as dist  # 引入PyTorch分布式通信模块
import torch.nn as nn  # 引入PyTorch神经网络模块
from torch.distributed import distributed_c10d  # 引入PyTorch分布式C10d模块
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 引入ShardedTensor类

from .sharder import Sharder  # 从当前目录下的sharder模块导入Sharder类
from .sharding_plan import ShardingPlan  # 从当前目录下的sharding_plan模块导入ShardingPlan类
from .sharding_spec import ChunkShardingSpec, ShardingSpec  # 从当前目录下的sharding_spec模块导入特定类

def _shard_tensor(
    tensor: torch.Tensor, sharding_spec: ShardingSpec, src_rank=0, process_group=None
) -> ShardedTensor:
    """
    Given a :class:`torch.Tensor`, it shards that tensor according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        A :class:`ShardedTensor` sharded from the given tensor.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    # 检查输入的张量是否是连续的，否则抛出异常
    if not tensor.is_contiguous():
        raise ValueError("input tensor is not a contiguous Tensor")

    # 获取要操作的进程组，默认为默认进程组
    pg = (
        process_group
        if process_group is not None
        else distributed_c10d._get_default_group()
    )
    # 获取当前进程组的全局大小和当前进程的排名
    world_size = dist.get_world_size(pg)
    current_rank = dist.get_rank(pg)

    # 验证在所有进程中，src_rank和sharding_spec是否一致
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, (src_rank, sharding_spec), group=pg)

    for idx, entry in enumerate(gathered_list):
        # 检查src_rank是否在所有进程中一致
        if src_rank != entry[0]:  # type: ignore[index]
            raise ValueError(
                f"src_rank={src_rank} on rank: {current_rank} does not "
                f"match with src_rank={entry[0]} on rank: {idx}"  # type: ignore[index]
            )
        # 检查sharding_spec是否在所有进程中一致
        if sharding_spec != entry[1]:  # type: ignore[index]
            raise ValueError(
                f"sharding_spec={sharding_spec} on rank: {current_rank} does not "
                f"match with sharding_spec={entry[1]} on rank: {idx}"  # type: ignore[index]
            )

    # 使用给定的sharding_spec对张量进行分片
    st = sharding_spec.shard(tensor, src_rank=src_rank, process_group=pg)

    return st


def shard_parameter(
    module: torch.nn.Module,
    param_name: str,
    sharding_spec: ShardingSpec,
    src_rank=0,
    process_group=None,
):
    """
    Placeholder function for sharding a parameter within a module.

    Args:
        module (torch.nn.Module): The module containing the parameter to shard.
        param_name (str): The name of the parameter within the module to shard.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the parameter tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    """
    # 此函数作为对模块中参数进行分片的占位符函数，具体实现未给出
    pass  # Placeholder function, implementation not provided
    # 首先进行一些验证。
    # 检查模块是否具有指定的参数名，如果没有则抛出异常。
    if not hasattr(module, param_name):
        raise AttributeError(f"{module._get_name()} has no attribute `{param_name}`")

    # 获取指定参数名在模块中的张量对象。
    tensor = getattr(module, param_name)
    # 检查获取到的对象是否为 torch.Tensor 类型，如果不是则抛出异常。
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}"
        )

    # 检查张量是否是连续的，如果不是则抛出异常。
    if not tensor.is_contiguous():
        raise ValueError(f"param: {param_name} is not a contiguous Tensor")

    # 使用 _shard_tensor 函数对张量进行分片操作，返回一个 ShardedTensor 对象。
    st = _shard_tensor(tensor, sharding_spec, src_rank, process_group)

    # 将原始模块中的参数替换为新创建的 ShardedTensor 对象。
    # 注册新的参数，使其成为模块的一个属性。
    module.register_parameter(param_name, nn.Parameter(st))
# 当前进程组的跟踪器，在加载上下文管理器中使用。
_CURRENT_PROCESS_GROUP: Optional[dist.ProcessGroup] = None

# 加载上下文管理器，用于设置加载 ShardedTensor 时使用的进程组。
@contextmanager
def load_with_process_group(process_group):
    """
    设置加载 ShardedTensor 时使用的进程组的上下文管理器。
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            "context manager"
        )
    _CURRENT_PROCESS_GROUP = process_group
    try:
        yield process_group
    finally:
        _CURRENT_PROCESS_GROUP = None

# 获取当前由 ``load_with_process_group`` 设置的进程组。
def _get_current_process_group():
    """
    获取由 ``load_with_process_group`` 设置的当前进程组。
    如果未设置，则返回默认组。
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is None:
        return distributed_c10d._get_default_group()
    else:
        return _CURRENT_PROCESS_GROUP

# 将模块的输出根据给定的 ``resharding_spec`` 钩住并进行输出重分片。
def _reshard_output(
    module: torch.nn.Module, resharding_spec: ShardingSpec
) -> torch.nn.Module:
    """
    在前向传递中使用给定的 ``resharding_spec`` 钩住模块的输出进行输出重分片。

    Args:
        module (:class:`torch.nn.Module`): 需要重分片输出的模块。
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            描述模块输出将如何进行重分片的规范。

    Returns:
        一个带有重分片 API 钩住的 :class:`torch.nn.Module` 对象。
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            return output.reshard(resharding_spec)
        return output

    module.register_forward_hook(hook_func)
    return module

# 钩住模块，在前向传递中收集本地分片。
def _collect_local_shard(module: torch.nn.Module) -> torch.nn.Module:
    """
    钩住模块，在前向传递中收集本地分片。

    该 API 通常用于将分片表示转换回数据并行表示。特别地，它返回此分片的本地张量。
    如果本地张量沿分片维度的大小为 1，则从最终结果中删除此维度。
    例如，跨 4 个秩的 [4, 16] ShardedTensor 通常是在每个秩上大小为 [16] 的本地张量，而不是在每个秩上大小为 [1, 16]。

    Args:
        module (:class:`torch.nn.Module`): 输出为 ShardedTensor 的模块，需要返回本地张量值。

    Returns:
        一个带有收集 API 钩住的 :class:`torch.nn.Module` 对象。
    """
    # 注册一个前向钩子函数，用于处理模块的输出
    def hook_func(_module, _input, output):
        # 检查输出是否为 ShardedTensor 类型
        if isinstance(output, ShardedTensor):
            # 获取本地张量，用于分片
            local_tensor = output.local_tensor()
            # 获取分片规范
            sharding_spec = output._sharding_spec
            # 如果分片规范为 ChunkShardingSpec，并且在指定的维度上本地张量的大小为 1
            if (
                isinstance(sharding_spec, ChunkShardingSpec)
                and local_tensor.size(sharding_spec.dim) == 1  # type: ignore[attr-defined, arg-type]
            ):
                # 在指定的维度上挤压张量，只适用于 ChunkShardingSpec
                local_tensor = local_tensor.squeeze(
                    output._sharding_spec.dim  # type: ignore[attr-defined]
                )
            # 返回处理后的本地张量
            return local_tensor

    # 将 hook_func 注册为模块的前向钩子函数
    module.register_forward_hook(hook_func)
    # 返回注册后的模块
    return module
    # 记录 Sharder 的路径，用于检查计划中的项是否与 Sharder 工作的子模块树冲突
    sharder_paths = []
    # 遍历 sharding plan 中的每一个项，包括参数名和对应的 ShardingSpec
    for name, spec in plan.plan.items():
        # 如果当前项的规范是 Sharder 类型的实例
        if isinstance(spec, Sharder):
            # 将当前项的名称添加到 sharder_paths 列表中
            sharder_paths.append(name)

    # 根据 ShardingPlan 对参数进行分片处理
    # 遍历计划中的每个项目，其中每个项目包含名称和规格
    for name, spec in plan.plan.items():
        # 检查规格是否为ShardingSpec实例
        if isinstance(spec, ShardingSpec):
            # 如果发现了ShardingSpec，则尝试对参数进行分片
            # 将名称分割为模块路径、点号和参数名
            module_path, _, param_name = name.rpartition(".")

            # 遍历分片器路径列表，检查是否有模块路径以分片器路径开头
            for sharder_path in sharder_paths:
                if module_path.startswith(sharder_path):
                    # 如果找到模块路径以分片器路径开头，则引发运行时错误
                    raise RuntimeError(
                        f"ShardingPlan is in-valid, trying to shard a parameter: {name},"
                        f" but there's already a Sharder entry for module {sharder_path},"
                        f" parameter sharding should not conflict with the submodule tree"
                        f" that a Sharder is working with!"
                    )

            # 获取模块路径对应的子模块对象
            mod = module.get_submodule(module_path)
            # 调用shard_parameter函数，对模块的参数进行分片
            shard_parameter(
                mod, param_name, spec, src_rank=src_rank, process_group=process_group
            )
        # 如果规格是Sharder实例
        elif isinstance(spec, Sharder):
            # 将名称分割为父模块路径、点号和模块名
            parent_mod_path, _, mod_name = name.rpartition(".")
            # 如果名称为空，则引发关键错误，模块路径不能为空
            if name == "":
                raise KeyError("Module path must not be empty for custom sharder!")
            # 获取模块路径对应的子模块对象
            mod = module.get_submodule(name)
            # 获取父模块路径对应的子模块对象
            parent_mod = module.get_submodule(parent_mod_path)
            # 使用Sharder对象对模块进行分片，并将其替换为分片后的模块
            sharded_mod = spec.shard(mod)
            # 将父模块中的子模块名替换为分片后的模块
            parent_mod.mod_name = sharded_mod
        # 如果规格既不是ShardingSpec也不是Sharder实例，则引发类型错误
        else:
            raise TypeError(
                f"Only `ShardingSpec` and `Sharder` are supported to shard '{name}'"
            )

    # 如果计划中存在输出计划
    if plan.output_plan is not None:
        # 遍历输出计划中的每个模块路径和输出规格
        for module_path, output_spec in plan.output_plan.items():
            # 检查输出规格是否为ShardingSpec实例
            if isinstance(output_spec, ShardingSpec):
                # 获取模块路径对应的子模块对象
                mod = module.get_submodule(module_path)
                # 调用_reshard_output函数，重新分片输出
                _reshard_output(mod, output_spec)
            # 如果输出规格不是ShardingSpec实例，则引发类型错误
            else:
                raise TypeError(
                    f"Only `ShardingSpec` is supported as output_plan for '{module_path}'"
                )

    # 如果计划中存在return_local_tensor条目
    if plan.return_local_tensor is not None:
        # 遍历return_local_tensor列表中的每个模块路径
        for module_path in plan.return_local_tensor:
            # 获取模块路径对应的子模块对象
            mod = module.get_submodule(module_path)
            # 调用_collect_local_shard函数，收集模块的本地张量
            _collect_local_shard(mod)
```