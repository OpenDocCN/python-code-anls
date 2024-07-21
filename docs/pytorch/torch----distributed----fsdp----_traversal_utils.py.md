# `.\pytorch\torch\distributed\fsdp\_traversal_utils.py`

```py
"""
NOTE: This file must be imported like
``import torch.distributed.fsdp._traversal_utils`` and not like
``from torch.distirbuted.fsdp._traversal_utils import ...`` to avoid circular
imports. For brevity, we may import the file as ``traversal_utils``.
"""

# 引入必要的库
import collections
from typing import Deque, List, Set, Tuple

import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state


"""
[Note: FSDP State Traversal]
For the wrapper code path, ``_FSDPState`` is the ``FullyShardedDataParallel``
module wrapping a fully sharded module, and for the non-wrapper code path,
``_FSDPState`` is an object that gets embedded on a fully sharded module.
See [Note: Fully Sharded Module] for the definition.

There are three common traversal idioms: Given a root module,
- ``_get_fsdp_states()`` returns all ``_FSDPState`` s in the tree.
- ``get_fsdp_root_states()`` returns all local root ``_FSDPState`` s in the
tree (i.e. those with ``_is_root == True``).
- ``_get_fsdp_handles()``returns all ``FlatParamHandle`` s in the tree.

All of these methods must take in the root module (i.e. an ``nn.Module``) and
not a general ``_FSDPState`` because ``_FSDPState`` does not support a graph
traversal, whereas ``nn.Module`` has ``nn.Module.modules()`` for traversal.
"""


def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
    # 获取模块的注册表
    registry = _get_registry(module)
    # 如果注册表为空，说明可以与 fully_shard 组合
    if registry is None:
        return True
    # 如果注册表中不包含 'replicate'，也说明可以与 fully_shard 组合
    return "replicate" not in registry


# TODO (awgu): We may be able to remove this function if we retired the
# `use_orig_params=False` code path since so far we only need the module for
# `FlatParameter` registration, which is not needed for `use_orig_params=True`.
def _get_fsdp_states_with_modules(
    module: nn.Module,
) -> Tuple[List[_FSDPState], List[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the modules owning the states in the first list.

    For the wrapper code path, both returned lists are the same, each
    containing all ``FullyShardedDataParallel`` instances. For the composable
    code path, this returns a list of all composable state instances and a list
    of the corresponding fully sharded modules. See [Note: Fully Sharded
    Module].

    NOTE: The traversal does not proceed into any module annotated by an
    incompatible API (e.g. ``replicate``).
    """
    # 初始化空列表用于存储 FSDP 状态和模块
    fsdp_states: List[_FSDPState] = []
    fsdp_modules: List[nn.Module] = []
    # 跟踪已访问的 FSDP 状态，因为多个模块可能共享相同的状态，我们需要返回去重后的列表
    # 创建一个空的集合，用于跟踪已访问的_FSDPState对象
    visited_fsdp_states: Set[_FSDPState] = set()
    
    # 创建一个空的集合，用于跟踪已访问的nn.Module对象
    visited_modules: Set[nn.Module] = set()

    # 创建一个双端队列，起始包含参数传入的module，用于实现深度优先搜索
    deque: Deque[nn.Module] = collections.deque([module])
    
    # 当队列不为空时进行循环
    while deque:
        # 从队列左侧取出一个子模块
        submodule = deque.popleft()
        
        # 将当前子模块标记为已访问
        visited_modules.add(submodule)
        
        # 如果子模块不兼容，则跳过
        if not _composable(submodule):
            continue
        
        # 对当前子模块的每个子模块进行反向遍历
        for child_module in reversed(list(submodule.children())):
            # 如果子模块未被访问过，则加入队列左侧
            if child_module not in visited_modules:
                deque.appendleft(child_module)
        
        # 获取当前子模块的FSDP状态
        optional_state = _get_module_fsdp_state(submodule)
        
        # 如果FSDP状态不为空且未被访问过，则加入已访问的FSDP状态集合，并分别添加到fsdp_states和fsdp_modules列表中
        if optional_state is not None and optional_state not in visited_fsdp_states:
            visited_fsdp_states.add(optional_state)
            fsdp_states.append(optional_state)
            fsdp_modules.append(submodule)
    
    # 返回所有收集到的FSDP状态和对应的模块列表
    return fsdp_states, fsdp_modules
# 获取与给定模块关联的所有 FSDP 状态列表
def _get_fsdp_states(module: nn.Module) -> List[_FSDPState]:
    """See :func:`_get_fsdp_states_with_modules`."""
    # 调用 _get_fsdp_states_with_modules 函数获取模块及其子模块的 FSDP 状态列表和模块引用
    fsdp_states, _ = _get_fsdp_states_with_modules(module)
    # 返回仅包含 FSDP 状态的列表
    return fsdp_states


# 获取给定模块及其子模块中所有 FlatParamHandle 的列表
def _get_fsdp_handles(module: nn.Module) -> List:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_fsdp_state`.
    """
    # 通过遍历每个 FSDP 状态，提取其关联的 FlatParamHandle
    handles = [
        fsdp_state._handle
        for fsdp_state in _get_fsdp_states(module)  # 遍历模块及其子模块的 FSDP 状态列表
        if fsdp_state._handle is not None  # 仅选择具有非空 _handle 的 FSDP 状态
    ]
    # 返回所有 FlatParamHandle 组成的列表
    return handles
```