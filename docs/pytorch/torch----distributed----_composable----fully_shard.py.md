# `.\pytorch\torch\distributed\_composable\fully_shard.py`

```
# 导入必要的类型定义
from typing import Callable, Iterable, Optional, Union
from typing_extensions import deprecated

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import contract
from torch.distributed._composable_state import _get_module_state, _insert_module_state
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
)
from torch.distributed.fsdp._runtime_utils import (
    _register_post_forward_hook,
    _register_pre_forward_hook,
    _register_root_pre_forward_hook,
)
from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import _Policy

# 定义函数 fully_shard，应用 FullyShardedDataParallel (FSDP) 语义到给定的模块
@contract(state_cls=_FSDPState)
@deprecated(
    "`torch.distributed._composable.fully_shard` is being deprecated. "
    "You can continue to use the wrapper based FSDP. "
    "See usage in: https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py. "
    "`torch.distributed._composable.fully_shard` will be removed after PyTorch 2.5.",
    category=FutureWarning,
)
def fully_shard(
    module: nn.Module,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    policy: Optional[_Policy] = None,
    strategy: Optional[ShardingStrategy] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    cpu_offload: Optional[CPUOffload] = None,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
    device_id: Optional[Union[int, torch.device]] = None,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    sync_module_states: bool = False,
    forward_prefetch: bool = False,
    ignored_states: Union[
        Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
    ] = None,
) -> nn.Module:
    """Applies ``FullyShardedDataParallel`` (FSDP) semantics to ``module``."""
    # 记录 API 使用情况
    torch._C._log_api_usage_once("torch.distributed.fully_shard")
    
    # 强制执行新的自动包装策略
    if policy is not None and not isinstance(policy, _Policy):
        raise ValueError(f"Expects a `_Policy` but got {policy}")
    
    # 获取模块的状态对象
    state = fully_shard.state(module)
    
    # 初始化被忽略的模块状态
    state = _init_ignored_module_states(state, module, ignored_modules, ignored_states)
    
    # 初始化设备句柄
    state = _init_device_handle(state, module, state._ignored_params, device_id)
    
    # 对模块进行标注以适配动态分片
    _annotate_modules_for_dynamo(module, state._ignored_modules, True)
    # 初始化处理组状态，返回更新后的状态对象
    state = _init_process_group_state(state, process_group, strategy, policy)
    
    # 如果策略不为None，设置根参数字典root_kwargs，包含各种初始化参数
    if policy is not None:
        root_kwargs = {
            "process_group": process_group,
            "strategy": strategy,
            "mixed_precision": mixed_precision,
            "cpu_offload": cpu_offload,
            "ignored_modules": ignored_modules,
            "device_id": device_id,
            "param_init_fn": param_init_fn,
            "sync_module_states": sync_module_states,
            "forward_prefetch": forward_prefetch,
            "ignored_states": ignored_states,
        }
        # 如果策略在HYBRID_SHARDING_STRATEGIES中，则更新process_group参数
        if strategy in HYBRID_SHARDING_STRATEGIES:
            root_kwargs["process_group"] = (state.process_group, state._inter_node_pg)
        # 对module进行自动包装处理
        _auto_wrap(
            module,
            policy,
            state._ignored_modules,
            state._ignored_params,
            root_kwargs,
            fully_shard,
        )
    
    # 初始化核心状态state，如果策略为None，则使用默认的FULL_SHARD策略
    state = _init_core_state(
        state,
        strategy or ShardingStrategy.FULL_SHARD,
        mixed_precision,
        cpu_offload,
        limit_all_gathers=True,
        use_orig_params=True,
        backward_prefetch_limit=1,
        forward_prefetch_limit=1,
    )
    
    # 初始化运行时状态state
    state = _init_runtime_state(state)
    
    # 初始化预取状态state，使用BackwardPrefetch.BACKWARD_PRE和前向预取参数forward_prefetch
    state = _init_prefetching_state(
        state, BackwardPrefetch.BACKWARD_PRE, forward_prefetch=forward_prefetch
    )
    
    # 初始化缓冲状态state，使用当前module和状态state
    state = _init_buffer_state(state, module)
    
    # 从module初始化参数处理状态state，设置设备ID、参数初始化函数param_init_fn和同步模块状态sync_module_states
    state = _init_param_handle_from_module(
        state, module, device_id, param_init_fn, sync_module_states
    )
    
    # 初始化状态字典状态state
    state = _init_state_dict_state(state)
    
    # 注册所有状态字典钩子函数到state
    _register_all_state_dict_hooks(state)
    
    # 注册前向钩子函数到state和module
    _register_pre_forward_hook(state, module)
    
    # 注册后向钩子函数到state和module
    _register_post_forward_hook(state, module)
    
    # 注册根前向钩子函数到state和module（在最前面插入）
    _register_root_pre_forward_hook(state, module)
    
    # 始终插入传入module的状态，即使它没有管理的参数，此时没有句柄并且不出现在`_fully_sharded_module_to_handles`中
    _insert_module_state(module, state)
    
    # 遍历module的所有子模块，如果子模块在`state._fully_sharded_module_to_handle`中并且其状态为None，则插入该子模块的状态
    for submodule in module.modules():
        if (
            submodule in state._fully_sharded_module_to_handle
            and _get_module_state(submodule) is None
        ):
            _insert_module_state(submodule, state)
    
    # 返回经过所有初始化和注册后的module
    return module
```