# `.\pytorch\torch\distributed\_composable_state.py`

```
# 引入必要的模块和类型声明
from typing import cast, Dict, Optional
import torch.nn as nn

# 定义一个空类 `_State`，用于表示模块的状态
class _State:
    pass

# 创建一个字典 `_module_state_mapping`，用于映射 nn.Module 到 _State 对象
_module_state_mapping: Dict[nn.Module, _State] = {}

# 插入模块状态的函数，将指定模块和其状态存入全局映射 `_module_state_mapping`
def _insert_module_state(module: nn.Module, state: _State) -> None:
    global _module_state_mapping
    # 断言确保插入的模块不在映射中，避免重复插入
    assert module not in _module_state_mapping, f"Inserting {module} more than once."
    _module_state_mapping[module] = state

# 获取模块状态的函数，根据给定模块返回其对应的 _State 对象或 None
def _get_module_state(module: nn.Module) -> Optional[_State]:
    """
    Return the ``_State`` in ``model``.

    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State`` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """
    global _module_state_mapping
    # 如果模块本身就是 _State 类型，则直接返回
    if isinstance(module, _State):
        return cast(_State, module)
    else:
        # 否则，查找模块是否在全局映射中，返回其对应的状态对象或 None
        if module in _module_state_mapping:
            return _module_state_mapping[module]
        else:
            return None
```