# `.\pytorch\torch\distributed\_composable\contract.py`

```
# mypy: allow-untyped-defs
# 引入 uuid 模块，用于生成唯一标识符
import uuid
# 引入 OrderedDict，用于按顺序存储注册的 API
from collections import OrderedDict
# 引入 wraps，用于装饰器函数，使得被 contract 装饰的函数可序列化（pickleable）
from functools import wraps
# 引入 Callable、Dict、List、Optional、Type，用于类型提示
from typing import Callable, Dict, List, Optional, Type

# 引入 PyTorch 的神经网络模块 nn
import torch.nn as nn
# 引入 torch.distributed._composable_state 中的 _State 类
from torch.distributed._composable_state import _State

# 定义一个函数 generate_state_key，生成状态键的唯一标识符
def generate_state_key(string="__composable_api_state_key"):
    return f"{string}_{str(uuid.uuid4())}"

# 生成全局唯一的状态键和注册键
STATE_KEY = generate_state_key()
REGISTRY_KEY = generate_state_key()

# TODO: we can add additional info to RegistryItem to share across APIs. E.g.,
# we can add args and kwargs here, and then we can detect whether fully_shard
# is combined with reentrant activation checkpointing and error out with a clear
# message.
# 定义一个空的类 RegistryItem，用于作为注册项的基类
class RegistryItem:
    pass

# 定义装饰器函数 contract，用于装饰一个函数作为可组合的分布式 API
def contract(state_cls: Type[_State] = _State):
    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance. The
    decorator verifies that the wrapped function does not modify parameter,
    buffer or sub-module fully-qualified names (FQN).

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract()
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    """
    
    # wraps will make functions decorated with contract() pickleable - needed for integration with torch.package
    # 使用 wraps 装饰器使得 state_cls 可以序列化，并返回内部函数 inner
    @wraps(state_cls)
    def inner(func):
        return func
    
    return inner

# 定义一个函数 _get_registry，用于获取模块上已应用的可组合 API 的有序字典
def _get_registry(module: nn.Module) -> Optional[Dict[str, RegistryItem]]:
    r"""
    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    """
    # 返回模块上的注册项字典，如果没有应用任何 API，则返回 None
    return getattr(module, REGISTRY_KEY, None)
```